"""Unit tests for the grounded Document Concept & Excerpt Explanation Generator."""

import unittest
from unittest.mock import MagicMock

import src.config as config
from src.generation.explanation_generator import (
    EXPLANATION_STYLES,
    EXPLANATION_SYSTEM_PROMPT,
    explain_concept,
)
from src.generation.llm_provider import BaseLLMProvider


class _MockExplainerProvider(BaseLLMProvider):
    provider_name = "mock_ollama"

    def __init__(self, response_text: str = "Detailed explanation grounded in context."):
        super().__init__(model_name="qwen2.5:7b")
        self.response_text = response_text
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

    def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ):
        self.calls.append({
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        })
        for word in self.response_text.split():
            yield word + " "


class TestExplanationGenerator(unittest.TestCase):
    def setUp(self):
        self.sample_chunks = [
            {
                "chunk_id": "c1",
                "text": "Photosystem II contains P680 reaction center that absorbs 680 nm light.",
                "metadata": {
                    "filename": "botany.pdf",
                    "page": 3,
                    "page_range": "3",
                    "chunk_index": 0,
                    "chunk_id": "c1",
                },
            }
        ]

    def test_explain_concept_standard(self):
        provider = _MockExplainerProvider("Photosystem II initiates the light-dependent reactions [Page 3].")
        result = explain_concept(
            "Photosystem II",
            self.sample_chunks,
            style="standard",
            provider=provider,
            stream=False,
        )
        self.assertIn("Photosystem II initiates", result)
        self.assertEqual(len(provider.calls), 1)
        self.assertIn("Explain clearly", provider.calls[0]["user_prompt"])

    def test_explain_concept_simplified(self):
        provider = _MockExplainerProvider("Think of Photosystem II like a solar collector.")
        result = explain_concept(
            "Photosystem II",
            self.sample_chunks,
            style="simplified",
            provider=provider,
            stream=False,
        )
        self.assertIn("solar collector", result)
        self.assertIn("simple, everyday language", provider.calls[0]["user_prompt"])

    def test_explain_concept_academic(self):
        provider = _MockExplainerProvider("Photochemical excitation of the P680 dimer drives water oxidation.")
        result = explain_concept(
            "Photosystem II",
            self.sample_chunks,
            style="academic",
            provider=provider,
            stream=False,
        )
        self.assertIn("Photochemical excitation", result)
        self.assertIn("rigorous academic explanation", provider.calls[0]["user_prompt"])

    def test_explain_concept_streaming(self):
        provider = _MockExplainerProvider("Token1 Token2 Token3")
        stream_iter = explain_concept(
            "Photosystem II",
            self.sample_chunks,
            style="standard",
            provider=provider,
            stream=True,
        )
        tokens = list(stream_iter)
        self.assertEqual(len(tokens), 3)
        self.assertEqual("".join(tokens).strip(), "Token1 Token2 Token3")
        self.assertTrue(provider.calls[0].get("stream"))

    def test_explain_concept_empty_inputs_raise(self):
        provider = _MockExplainerProvider()
        with self.assertRaises(ValueError):
            explain_concept("", self.sample_chunks, provider=provider)
        with self.assertRaises(ValueError):
            explain_concept("Photosystem II", [], provider=provider)


if __name__ == "__main__":
    unittest.main()
