"""Run a fixed, resumable HealthBench Hard sample against a local harness."""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "data/healthbench/healthbench_hard.jsonl"
DEFAULT_REPORT = ROOT / "data/evals/healthbench_hard_fixed_100.json"
DEFAULT_GRADER_SOURCE = ROOT / "reference/simple-evals/healthbench_eval.py"

GRADE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "explanation": {"type": "string"},
        "criteria_met": {"type": "boolean"},
    },
    "required": ["explanation", "criteria_met"],
}


def fixed_sample(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Select a stable UUID-ordered sample independent of JSONL row order."""
    if count < 1:
        raise ValueError("count must be positive")
    if len(rows) < count:
        raise ValueError(f"dataset has only {len(rows)} rows; cannot select {count}")
    return sorted(rows, key=lambda row: row["prompt_id"])[:count]


def calculate_score(rubrics: list[dict[str, Any]], grades: list[dict[str, Any]]) -> float:
    possible = sum(item["points"] for item in rubrics if item["points"] > 0)
    if possible <= 0:
        raise ValueError("example has no positive rubric points")
    achieved = sum(
        rubric["points"]
        for rubric, grade in zip(rubrics, grades, strict=True)
        if grade["criteria_met"]
    )
    return achieved / possible


def load_grader_template(path: Path) -> str:
    """Safely extract the official constant without importing optional simple-evals deps."""
    module = ast.parse(path.read_text(encoding="utf-8"))
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "GRADER_TEMPLATE" for target in node.targets):
            continue
        value = node.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "strip"
            and not value.args
            and isinstance(value.func.value, ast.Constant)
            and isinstance(value.func.value.value, str)
        ):
            return value.func.value.value.strip()
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
        raise ValueError("GRADER_TEMPLATE has an unexpected expression")
    raise ValueError("GRADER_TEMPLATE not found")


def resolve_openai_key() -> str:
    value = os.getenv("OPENAI_API_KEY", "").strip()
    if value:
        return value
    path = ROOT / "submission_openai_key"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


def response_text(data: dict[str, Any]) -> str:
    direct = data.get("output_text")
    if isinstance(direct, str) and direct:
        return direct
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                return content["text"]
    raise ValueError("grader response contained no output text")


class Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.key = resolve_openai_key()
        if not self.key:
            raise ValueError("set OPENAI_API_KEY or provide submission_openai_key")
        self.template = load_grader_template(Path(args.grader_source))
        self.grade_slots = asyncio.Semaphore(args.grader_concurrency)

    async def grade_rubric(
        self,
        client: httpx.AsyncClient,
        conversation: list[dict[str, str]],
        rubric: dict[str, Any],
    ) -> dict[str, Any]:
        conversation_text = "\n\n".join(
            f"{message['role']}: {message['content']}" for message in conversation
        )
        rubric_text = f"[{rubric['points']}] {rubric['criterion']}"
        prompt = self.template.replace("<<conversation>>", conversation_text).replace(
            "<<rubric_item>>", rubric_text
        )
        payload = {
            "model": self.args.grader_model,
            "input": prompt,
            "reasoning": {"effort": self.args.grader_reasoning_effort},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "healthbench_rubric_grade",
                    "strict": True,
                    "schema": GRADE_SCHEMA,
                }
            },
            "store": False,
        }
        headers = {"Authorization": f"Bearer {self.key}"}
        async with self.grade_slots:
            for attempt in range(3):
                try:
                    response = await client.post(self.args.openai_url, headers=headers, json=payload)
                    response.raise_for_status()
                    grade = json.loads(response_text(response.json()))
                    if isinstance(grade.get("criteria_met"), bool):
                        return {**rubric, **grade}
                except (httpx.HTTPError, ValueError, json.JSONDecodeError):
                    if attempt == 2:
                        raise
                    await asyncio.sleep(2**attempt)
        raise RuntimeError("grader failed to return a boolean result")

    async def run(self) -> dict[str, Any]:
        dataset_path = Path(self.args.dataset)
        with dataset_path.open(encoding="utf-8") as source:
            rows = [json.loads(line) for line in source]
        selected = fixed_sample(rows, self.args.count)
        selected_ids = [row["prompt_id"] for row in selected]

        report_path = Path(self.args.output)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report: dict[str, Any]
        if report_path.exists() and not self.args.reset:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("selected_prompt_ids") != selected_ids:
                raise ValueError("existing report uses a different fixed sample; pass --reset")
        else:
            report = {
                "status": "running",
                "split": "hard",
                "selection": "lexicographically-smallest prompt_id values",
                "count": self.args.count,
                "selected_prompt_ids": selected_ids,
                "generator_endpoint": self.args.endpoint,
                "grader_model": self.args.grader_model,
                "examples": [],
            }
            self.save(report_path, report)

        existing = {item["prompt_id"]: item for item in report["examples"]}
        timeout = httpx.Timeout(self.args.timeout, connect=15)
        async with httpx.AsyncClient(timeout=timeout) as client:
            for position, row in enumerate(selected, 1):
                item = existing.get(row["prompt_id"])
                if item and item.get("status") == "graded":
                    print(f"SKIP {position}/{self.args.count} score={item['score']:.4f}", flush=True)
                    continue
                if item is None:
                    print(f"GENERATE {position}/{self.args.count} id={row['prompt_id']}", flush=True)
                    started = time.monotonic()
                    response = await client.post(
                        self.args.endpoint,
                        json={"model": self.args.request_model, "messages": row["prompt"], "stream": False},
                    )
                    response.raise_for_status()
                    completion = response.json()["choices"][0]["message"]["content"]
                    item = {
                        "position": position,
                        "prompt_id": row["prompt_id"],
                        "prompt": row["prompt"],
                        "completion": completion,
                        "generation_seconds": time.monotonic() - started,
                        "status": "generated",
                        "rubric_grades": [],
                    }
                    report["examples"].append(item)
                    existing[row["prompt_id"]] = item
                    self.save(report_path, report)

                print(f"GRADE {position}/{self.args.count} rubrics={len(row['rubrics'])}", flush=True)
                conversation = row["prompt"] + [{"role": "assistant", "content": item["completion"]}]
                grades = await asyncio.gather(
                    *(self.grade_rubric(client, conversation, rubric) for rubric in row["rubrics"])
                )
                possible = sum(rubric["points"] for rubric in row["rubrics"] if rubric["points"] > 0)
                achieved = sum(grade["points"] for grade in grades if grade["criteria_met"])
                item.update(
                    status="graded",
                    positive_possible_points=possible,
                    achieved_points=achieved,
                    score=calculate_score(row["rubrics"], grades),
                    rubric_grades=grades,
                )
                self.save(report_path, report)
                print(
                    f"RESULT {position}/{self.args.count} score={item['score']:.4f} "
                    f"points={achieved}/{possible}",
                    flush=True,
                )

        scores = [item["score"] for item in report["examples"] if item.get("status") == "graded"]
        report["mean_score_clipped"] = min(1.0, max(0.0, sum(scores) / len(scores)))
        report["status"] = "complete" if len(scores) == self.args.count else "partial"
        self.save(report_path, report)
        return report

    @staticmethod
    def save(path: Path, report: dict[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".part")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--grader-source", default=str(DEFAULT_GRADER_SOURCE))
    parser.add_argument("--output", default=str(DEFAULT_REPORT))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--request-model", default="active-strategy")
    parser.add_argument("--grader-model", default="gpt-5.6-sol")
    parser.add_argument("--grader-reasoning-effort", default="medium")
    parser.add_argument("--grader-concurrency", type=int, default=5)
    parser.add_argument("--openai-url", default="https://api.openai.com/v1/responses")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--reset", action="store_true", help="discard the prior report")
    return parser.parse_args()


def main() -> None:
    try:
        report = asyncio.run(Runner(parse_args()).run())
    except (ValueError, OSError, httpx.HTTPError) as exc:
        sys.exit(f"benchmark failed: {exc}")
    print(
        f"FINAL status={report['status']} n={len(report['examples'])} "
        f"mean_score_clipped={report['mean_score_clipped']:.4f}"
    )


if __name__ == "__main__":
    main()
