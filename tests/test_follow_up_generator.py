"""Unit tests for the grounded Follow-Up Question Generator powered by Qwen2.5 7B."""

import json
import unittest

from src.generation.follow_up_generator import (
    extract_follow_up_json,
    generate_follow_up_questions,
)
from src.generation.llm_provider import BaseLLMProvider


class _MockFollowUpProvider(BaseLLMProvider):
    provider_name = "mock_ollama"

    def __init__(self, response_text: str | None = None):
        super().__init__(model_name="qwen2.5:7b")
        self.response_text = response_text or json.dumps({
            "follow_up_questions": [
                "How does the electron transport chain generate an electrochemical gradient?",
                "What role does ATP synthase play in oxidative phosphorylation?",
            ]
        })
        self.calls: list[dict] = []

    def is_configured(self) -> bool:
        return True

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> str:
        self.calls.append({
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "kwargs": kwargs,
        })
        return self.response_text


class TestFollowUpGenerator(unittest.TestCase):
    def test_extract_follow_up_json_direct(self):
        raw = json.dumps({
            "follow_up_questions": [
                "What is the difference between light and dark reactions?",
                "Where does the Calvin cycle take place?",
            ]
        })
        questions = extract_follow_up_json(raw)
        self.assertEqual(len(questions), 2)
        self.assertEqual(questions[0], "What is the difference between light and dark reactions?")

    def test_extract_follow_up_json_fenced(self):
        raw = "```json\n" + json.dumps({
            "follow_up_questions": ["Can you explain cyclic photophosphorylation?"]
        }) + "\n```"
        questions = extract_follow_up_json(raw)
        self.assertEqual(len(questions), 1)

    def test_extract_follow_up_fallback_bullets(self):
        raw = (
            "Here are some follow up questions:\n"
            "1. What is the role of NADPH in the stroma?\n"
            "2. How does temperature affect Rubisco enzyme activity?\n"
        )
        questions = extract_follow_up_json(raw)
        self.assertEqual(len(questions), 2)
        self.assertIn("What is the role of NADPH in the stroma?", questions)

    def test_extract_follow_up_empty(self):
        self.assertEqual(extract_follow_up_json(""), [])

    def test_generate_follow_up_questions_end_to_end(self):
        provider = _MockFollowUpProvider()
        chunks = [
            {
                "chunk_id": "c1",
                "text": "Photosynthesis produces ATP and NADPH.",
                "metadata": {"filename": "doc.pdf", "page": 1, "page_range": "1"},
            }
        ]
        questions = generate_follow_up_questions(
            "What is photosynthesis?",
            "It is the process of converting light energy into chemical energy.",
            chunks,
            count=2,
            provider=provider,
        )
        self.assertEqual(len(questions), 2)
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(provider.calls[0]["kwargs"].get("format"), "json")

    def test_generate_follow_up_empty_inputs(self):
        provider = _MockFollowUpProvider()
        self.assertEqual(generate_follow_up_questions("", "answer", [{"text": "abc"}], provider=provider), [])
        self.assertEqual(generate_follow_up_questions("question", "answer", [], provider=provider), [])


if __name__ == "__main__":
    unittest.main()
