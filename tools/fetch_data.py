#!/usr/bin/env python3
"""Manifest-driven downloader for large data assets.

This repo does not use git-lfs. Anything big (datasets, model dumps, reference
corpora) is declared in a JSON manifest and fetched on demand, verified by size
and SHA-256. Stdlib only -- no pip install needed, works in a bare container.

    python tools/fetch_data.py                       # fetch everything missing
    python tools/fetch_data.py status                # what's here, what's not
    python tools/fetch_data.py fetch healthbench     # fetch one group (or asset name)
    python tools/fetch_data.py fetch --force hard    # re-download even if present
    python tools/fetch_data.py verify                # re-hash everything on disk
    python tools/fetch_data.py list                  # show the manifest
    python tools/fetch_data.py add URL data/foo.bin --group mygroup
    python tools/fetch_data.py clean healthbench-extra --yes

Behaviour worth knowing:
  * Downloads stream to `<dest>.part` and resume with an HTTP Range request if
    interrupted, so a dropped 136 MB transfer doesn't start over.
  * An asset with a `sha256` in the manifest is verified after download and the
    file is only moved into place if it matches.
  * An asset with `"sha256": null` is unpinned (e.g. a file tracking a git
    branch); it is re-downloaded on `--force` only and never hash-checked.
  * `auth_env` on an asset sends `Authorization: Bearer $<that env var>`.

Manifest schema (tools/assets.json):
    {"version": 1, "assets": [
      {"name": "...", "group": "...", "url": "...", "dest": "relative/path",
       "size": 123 | null, "sha256": "hex" | null, "description": "...",
       "auth_env": "SOME_TOKEN"            # optional
      }
    ]}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "tools" / "assets.json"
CHUNK = 1 << 20  # 1 MiB
USER_AGENT = "fetch_data.py/1.0"


# --------------------------------------------------------------------------- io


def human(n: int | None) -> str:
    if n is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n}"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------- manifest


class Asset:
    def __init__(self, d: dict, root: Path):
        self.name: str = d["name"]
        self.group: str = d.get("group", "default")
        self.url: str = d["url"]
        self.dest: Path = root / d["dest"]
        self.rel: str = d["dest"]
        self.size: int | None = d.get("size")
        self.sha256: str | None = d.get("sha256")
        self.description: str = d.get("description", "")
        self.auth_env: str | None = d.get("auth_env")

    @property
    def pinned(self) -> bool:
        return self.sha256 is not None

    def present(self) -> bool:
        if not self.dest.exists():
            return False
        if self.size is not None and self.dest.stat().st_size != self.size:
            return False
        return True

    def headers(self) -> dict:
        h = {"User-Agent": USER_AGENT}
        if self.auth_env:
            token = os.environ.get(self.auth_env)
            if not token:
                raise SystemExit(
                    f"{self.name}: needs credentials, but ${self.auth_env} is not set"
                )
            h["Authorization"] = f"Bearer {token}"
        return h


def load_manifest(path: Path) -> tuple[dict, list[Asset]]:
    if not path.exists():
        raise SystemExit(f"manifest not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw, [Asset(d, ROOT) for d in raw["assets"]]


def select(assets: list[Asset], patterns: list[str]) -> list[Asset]:
    if not patterns:
        return assets
    chosen, unknown = [], []
    for pat in patterns:
        hits = [a for a in assets if a.name == pat or a.group == pat]
        if not hits:
            hits = [a for a in assets if pat in a.name or pat in a.group]
        if not hits:
            unknown.append(pat)
        for a in hits:
            if a not in chosen:
                chosen.append(a)
    if unknown:
        names = sorted({a.name for a in assets} | {a.group for a in assets})
        raise SystemExit(f"no asset or group matches {unknown}. known: {', '.join(names)}")
    return chosen


# --------------------------------------------------------------------- download


def download(asset: Asset, *, force: bool, verify: bool, dry_run: bool) -> str:
    """Returns one of: skipped, fetched, failed."""
    if asset.present() and not force:
        if verify and asset.pinned:
            actual = sha256_of(asset.dest)
            if actual != asset.sha256:
                log(f"  ! {asset.name}: on-disk hash mismatch, re-downloading")
                force = True
            else:
                log(f"  = {asset.name} ({human(asset.size)}) present, hash ok")
                return "skipped"
        else:
            log(f"  = {asset.name} ({human(asset.size)}) present")
            return "skipped"

    if dry_run:
        log(f"  + {asset.name} would download {human(asset.size)} -> {asset.rel}")
        return "fetched"

    asset.dest.parent.mkdir(parents=True, exist_ok=True)
    part = asset.dest.with_suffix(asset.dest.suffix + ".part")
    if force and part.exists():
        part.unlink()

    resume_at = part.stat().st_size if part.exists() else 0
    headers = asset.headers()
    if resume_at:
        headers["Range"] = f"bytes={resume_at}-"

    req = urllib.request.Request(asset.url, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if resume_at and e.code in (416, 501):  # range not satisfiable/supported
            part.unlink()
            return download(asset, force=True, verify=verify, dry_run=dry_run)
        log(f"  ! {asset.name}: HTTP {e.code} {e.reason}")
        return "failed"
    except OSError as e:
        log(f"  ! {asset.name}: {e}")
        return "failed"

    mode = "ab"
    if resume_at and resp.status != 206:  # server ignored our Range
        resume_at, mode = 0, "wb"
    elif not resume_at:
        mode = "wb"

    remaining = resp.headers.get("Content-Length")
    total = (int(remaining) + resume_at) if remaining else asset.size
    got, started, last = resume_at, time.time(), 0.0
    verb = "resuming" if resume_at else "downloading"
    log(f"  > {asset.name}: {verb} {human(total)} -> {asset.rel}")

    try:
        with resp, part.open(mode) as f:
            while True:
                block = resp.read(CHUNK)
                if not block:
                    break
                f.write(block)
                got += len(block)
                now = time.time()
                if now - last > 2.0:
                    last = now
                    rate = (got - resume_at) / max(now - started, 1e-6) / 1e6
                    pct = f"{got / total:5.1%}" if total else human(got)
                    log(f"    {asset.name}: {pct} ({rate:.1f} MB/s)")
    except OSError as e:
        log(f"  ! {asset.name}: transfer failed ({e}) - rerun to resume")
        return "failed"

    if asset.size is not None and got != asset.size:
        log(f"  ! {asset.name}: size mismatch (got {got}, expected {asset.size})")
        return "failed"
    if asset.pinned:
        actual = sha256_of(part)
        if actual != asset.sha256:
            log(f"  ! {asset.name}: sha256 mismatch\n      expected {asset.sha256}\n      got      {actual}")
            part.rename(part.with_suffix(".bad"))
            return "failed"

    os.replace(part, asset.dest)
    log(f"  + {asset.name}: done ({human(got)})")
    return "fetched"


# --------------------------------------------------------------------- commands


def cmd_list(args, assets: list[Asset]) -> int:
    by_group: dict[str, list[Asset]] = {}
    for a in assets:
        by_group.setdefault(a.group, []).append(a)
    for group, items in by_group.items():
        total = sum(a.size or 0 for a in items)
        print(f"\n[{group}]  {len(items)} asset(s), {human(total)}")
        for a in items:
            pin = "pinned" if a.pinned else "unpinned"
            print(f"  {a.name:<36} {human(a.size):>8}  {pin:<8} {a.rel}")
            if a.description:
                print(f"  {'':<36} {a.description}")
    return 0


def cmd_status(args, assets: list[Asset]) -> int:
    missing = 0
    for a in assets:
        state = "present" if a.present() else "MISSING"
        if not a.present():
            missing += 1
        print(f"  {state:<8} {a.name:<36} {human(a.size):>8}  {a.rel}")
    total_missing = sum(a.size or 0 for a in assets if not a.present())
    print(f"\n{len(assets) - missing}/{len(assets)} present; {human(total_missing)} to download")
    return 1 if missing else 0


def cmd_fetch(args, assets: list[Asset]) -> int:
    chosen = select(assets, args.targets)
    log(f"fetching {len(chosen)} asset(s) with {args.jobs} worker(s)")
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(
            pool.map(
                lambda a: download(
                    a, force=args.force, verify=not args.no_verify, dry_run=args.dry_run
                ),
                chosen,
            )
        )
    counts = {k: results.count(k) for k in ("fetched", "skipped", "failed")}
    log(f"\ndone: {counts['fetched']} fetched, {counts['skipped']} skipped, {counts['failed']} failed")
    return 1 if counts["failed"] else 0


def cmd_verify(args, assets: list[Asset]) -> int:
    bad = 0
    for a in select(assets, args.targets):
        if not a.dest.exists():
            print(f"  MISSING  {a.name}")
            bad += 1
        elif not a.pinned:
            print(f"  unpinned {a.name} ({human(a.dest.stat().st_size)})")
        elif sha256_of(a.dest) == a.sha256:
            print(f"  ok       {a.name}")
        else:
            print(f"  CORRUPT  {a.name} -- delete it and refetch")
            bad += 1
    return 1 if bad else 0


def cmd_add(args, assets: list[Asset]) -> int:
    """Download a new URL, hash it, and append it to the manifest."""
    manifest_path = Path(args.manifest)
    raw, _ = load_manifest(manifest_path)
    dest_rel = args.dest.replace("\\", "/")
    name = args.name or Path(dest_rel).stem.replace("_", "-")
    if any(d["name"] == name for d in raw["assets"]):
        raise SystemExit(f"asset '{name}' already in manifest")

    entry = {
        "name": name,
        "group": args.group,
        "url": args.url,
        "dest": dest_rel,
        "size": None,
        "sha256": None,
        "description": args.description,
    }
    if args.auth_env:
        entry["auth_env"] = args.auth_env

    asset = Asset(entry, ROOT)
    if download(asset, force=True, verify=False, dry_run=False) == "failed":
        return 1

    entry["size"] = asset.dest.stat().st_size
    entry["sha256"] = sha256_of(asset.dest)
    raw["assets"].append(entry)
    manifest_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log(f"added '{name}' to {manifest_path} ({human(entry['size'])}, sha256 {entry['sha256'][:12]}...)")
    return 0


def cmd_clean(args, assets: list[Asset]) -> int:
    chosen = select(assets, args.targets)
    freed = sum(a.dest.stat().st_size for a in chosen if a.dest.exists())
    if not args.yes:
        print(f"would delete {len(chosen)} asset(s), freeing {human(freed)}:")
        for a in chosen:
            if a.dest.exists():
                print(f"  {a.rel}")
        print("\nrerun with --yes to actually delete")
        return 0
    for a in chosen:
        if a.dest.exists():
            a.dest.unlink()
            log(f"  - removed {a.rel}")
    log(f"freed {human(freed)}")
    return 0


# ------------------------------------------------------------------------- main


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("list").set_defaults(func=cmd_list)
    sub.add_parser("status").set_defaults(func=cmd_status)

    s = sub.add_parser("fetch", help="download missing (or --force all) assets")
    s.add_argument("targets", nargs="*", help="asset names or group names; empty = all")
    s.add_argument("--force", action="store_true")
    s.add_argument("--no-verify", action="store_true", help="skip re-hashing files already on disk")
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--jobs", type=int, default=3)
    s.set_defaults(func=cmd_fetch)

    s = sub.add_parser("verify", help="re-hash on-disk files against the manifest")
    s.add_argument("targets", nargs="*")
    s.set_defaults(func=cmd_verify)

    s = sub.add_parser("add", help="download a new URL and record it in the manifest")
    s.add_argument("url")
    s.add_argument("dest", help="path relative to repo root, e.g. data/foo/bar.jsonl")
    s.add_argument("--name")
    s.add_argument("--group", default="default")
    s.add_argument("--description", default="")
    s.add_argument("--auth-env", dest="auth_env")
    s.set_defaults(func=cmd_add)

    s = sub.add_parser("clean", help="delete downloaded assets")
    s.add_argument("targets", nargs="*")
    s.add_argument("--yes", action="store_true")
    s.set_defaults(func=cmd_clean)

    args = p.parse_args()
    if args.cmd is None:  # bare invocation == fetch everything missing
        args = p.parse_args(["fetch"])

    _, assets = load_manifest(Path(args.manifest))
    return args.func(args, assets)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("\ninterrupted -- rerun to resume partial downloads")
        sys.exit(130)
