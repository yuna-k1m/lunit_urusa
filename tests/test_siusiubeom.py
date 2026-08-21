import unittest
from unittest.mock import patch

from chat_models.base import ChatRequest
from chat_models.siusiubeom import SiusiubeomH4Strategy


class SiusiubeomStrategyTest(unittest.IsolatedAsyncioTestCase):
    async def test_adapts_engine_output_and_preserves_messages(self):
        client = object()
        messages = [{"role": "user", "content": "질문"}]
        output = {
            "answer": "답변",
            "plan": {"language": "ko"},
            "notes": {"revised": False},
            "review": None,
            "timings": {"plan": 1.0},
        }
        with patch("chat_models.siusiubeom.engine.answer", return_value=output) as answer:
            result = await SiusiubeomH4Strategy(client=client, planner=None).complete(
                ChatRequest(messages=messages, original_payload={"temperature": 0.2, "max_tokens": 900})
            )
        self.assertEqual(result.content, "답변")
        self.assertEqual(result.metadata["plan"]["language"], "ko")
        # The evaluator's max_tokens is deliberately NOT forwarded (it would truncate
        # L2's reasoning+answer); the engine applies its own tuned budget.
        answer.assert_called_once_with(
            client, messages, temperature=0.2, max_tokens=None, planner=None,
            local_first=False,
        )


if __name__ == "__main__":
    unittest.main()
