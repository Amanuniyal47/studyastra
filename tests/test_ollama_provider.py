"""Automated unit tests for local Ollama and Qwen2.5 7B integration.

Uses mocked HTTP responses so that tests run quickly and reliably without requiring
a running Ollama daemon or a GPU.
"""

from collections.abc import Iterator
import json
import sys
import unittest
from unittest.mock import MagicMock, patch

import httpx

import src.config as config
from src.generation.answer_generator import generate_answer
from src.generation.llm_provider import (
    BaseLLMProvider,
    LLMConfigurationError,
    LLMProviderError,
    OllamaLLMProvider,
    get_llm_provider,
    list_registered_providers,
)
from src.generation.prompts import (
    GROUNDED_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_context,
    build_user_prompt,
)


class _DummyStreamResponse:
    """Mock HTTP streaming response that yields NDJSON lines."""

    def __init__(self, lines: list[str], status_code: int = 200):
        self.lines = lines
        self.status_code = status_code

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def iter_lines(self) -> Iterator[str]:
        yield from self.lines


class TestOllamaLLMProvider(unittest.TestCase):
    def setUp(self):
        OllamaLLMProvider.invalidate_health_cache()
        self.provider = OllamaLLMProvider(
            base_url="http://localhost:11434",
            model_name="qwen2.5:7b",
            timeout=30.0,
            context_window=4096,
            keep_alive="5m",
        )

    def test_01_configuration_and_defaults(self):
        self.assertTrue(self.provider.is_configured())
        self.assertEqual(self.provider.base_url, "http://localhost:11434")
        self.assertEqual(self.provider.model_name, "qwen2.5:7b")
        self.assertEqual(self.provider.context_window, 4096)
        self.assertEqual(self.provider.keep_alive, "5m")
        self.assertIn("ollama", list_registered_providers())

        # Test empty base_url error
        with self.assertRaises(LLMConfigurationError):
            OllamaLLMProvider(base_url="   ")

    def test_02_ollama_connection_success(self):
        mock_resp = MagicMock(status_code=200)
        with patch("httpx.get", return_value=mock_resp) as mock_get:
            ok, msg = self.provider.check_connection()
            self.assertTrue(ok)
            self.assertIn("reachable", msg)
            mock_get.assert_called_once_with(
                "http://localhost:11434/api/tags", timeout=5.0
            )

    def test_03_ollama_connection_failure_daemon_down(self):
        request = httpx.Request("GET", "http://localhost:11434/api/tags")
        with patch(
            "httpx.get", side_effect=httpx.ConnectError("Connection refused", request=request)
        ):
            ok, msg = self.provider.check_connection()
            self.assertFalse(ok)
            self.assertIn("Cannot connect to Ollama", msg)
            self.assertIn("ollama serve", msg)

    def test_04_ollama_model_availability_found(self):
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {
            "models": [
                {"name": "qwen2.5:7b", "model": "qwen2.5:7b"},
                {"name": "all-minilm:latest", "model": "all-minilm:latest"},
            ]
        }
        with patch("httpx.get", return_value=mock_resp):
            ok, msg = self.provider.check_model_available("qwen2.5:7b")
            self.assertTrue(ok)
            self.assertIn("available", msg)

    def test_05_ollama_model_availability_missing(self):
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {
            "models": [{"name": "llama3:latest", "model": "llama3:latest"}]
        }
        with patch("httpx.get", return_value=mock_resp):
            ok, msg = self.provider.check_model_available("qwen2.5:7b")
            self.assertFalse(ok)
            self.assertIn("not installed", msg)
            self.assertIn("ollama pull qwen2.5:7b", msg)

    def test_06_normal_text_generation_success(self):
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {
            "model": "qwen2.5:7b",
            "message": {"role": "assistant", "content": "  Photosynthesis produces glucose.  "},
            "done": True,
        }

        with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
            result = self.provider.generate(
                "You are an assistant.",
                "What does photosynthesis produce?",
                temperature=0.3,
                max_tokens=500,
            )
            self.assertEqual(result, "Photosynthesis produces glucose.")
            mock_post.assert_called_once()
            call_url = mock_post.call_args[0][0]
            call_json = mock_post.call_args[1]["json"]
            self.assertEqual(call_url, "http://localhost:11434/api/chat")
            self.assertEqual(call_json["model"], "qwen2.5:7b")
            self.assertEqual(call_json["stream"], False)
            self.assertEqual(call_json["options"]["temperature"], 0.3)
            self.assertEqual(call_json["options"]["num_predict"], 500)
            self.assertEqual(call_json["options"]["num_ctx"], 4096)
            self.assertEqual(call_json["keep_alive"], "5m")

    def test_07_streaming_generation_success(self):
        ndjson_lines = [
            json.dumps({"message": {"content": "Photosynthesis "}, "done": False}),
            json.dumps({"message": {"content": "converts light "}, "done": False}),
            json.dumps({"message": {"content": "into glucose."}, "done": True}),
        ]
        dummy_stream = _DummyStreamResponse(ndjson_lines, status_code=200)

        with patch("httpx.Client.stream", return_value=dummy_stream):
            tokens = list(
                self.provider.generate_stream(
                    "You are an assistant.",
                    "Explain photosynthesis.",
                    temperature=0.2,
                )
            )
            self.assertEqual(tokens, ["Photosynthesis ", "converts light ", "into glucose."])
            self.assertEqual("".join(tokens), "Photosynthesis converts light into glucose.")

    def test_08_ollama_unavailable_raises_llm_provider_error(self):
        request = httpx.Request("POST", "http://localhost:11434/api/chat")
        with patch(
            "httpx.Client.post", side_effect=httpx.ConnectError("Failed to connect", request=request)
        ):
            with self.assertRaises(LLMProviderError) as ctx:
                self.provider.generate("sys", "user")
            self.assertIn("Cannot connect to Ollama", str(ctx.exception))
            self.assertIn("ollama serve", str(ctx.exception))

    def test_09_model_not_found_raises_llm_provider_error(self):
        mock_resp = MagicMock(status_code=404)
        mock_resp.text = '{"error": "model \'qwen2.5:7b\' not found"}'
        mock_resp.json.return_value = {"error": "model 'qwen2.5:7b' not found"}

        with patch("httpx.Client.post", return_value=mock_resp):
            with self.assertRaises(LLMProviderError) as ctx:
                self.provider.generate("sys", "user")
            self.assertIn("was not found in Ollama", str(ctx.exception))
            self.assertIn("ollama pull qwen2.5:7b", str(ctx.exception))

    def test_10_timeout_raises_llm_provider_error(self):
        request = httpx.Request("POST", "http://localhost:11434/api/chat")
        with patch(
            "httpx.Client.post", side_effect=httpx.TimeoutException("Timed out", request=request)
        ):
            with self.assertRaises(LLMProviderError) as ctx:
                self.provider.generate("sys", "user")
            self.assertIn("timed out", str(ctx.exception).lower())

    def test_11_invalid_prompt_inputs(self):
        with self.assertRaises(ValueError):
            self.provider.generate("   ", "user")
        with self.assertRaises(ValueError):
            self.provider.generate("sys", "   ")
        with self.assertRaises(ValueError):
            self.provider.generate("sys", "user", max_tokens=-5)

    def test_12_rag_context_passed_correctly_to_qwen(self):
        mock_provider = MagicMock(spec=BaseLLMProvider)
        mock_provider.generate.return_value = "According to Source 1, photosynthesis creates glucose."

        chunks = [
            {
                "text": "Chloroplasts capture light energy to synthesize glucose from CO2.",
                "metadata": {
                    "filename": "biology_notes.pdf",
                    "page": 4,
                    "page_range": "4",
                    "chunk_index": 2,
                },
            }
        ]

        answer = generate_answer("What does photosynthesis make?", chunks, provider=mock_provider)
        self.assertEqual(answer, "According to Source 1, photosynthesis creates glucose.")
        mock_provider.generate.assert_called_once()
        passed_sys, passed_user = (
            mock_provider.generate.call_args[0][0],
            mock_provider.generate.call_args[0][1],
        )

        # Verify grounding rules in system prompt
        self.assertIn("CRITICAL GROUNDING RULES", passed_sys)
        self.assertIn("Strict Context Adherence", passed_sys)
        self.assertIn("I cannot answer this question based on the provided document context", passed_sys)

        # Verify context and metadata formatting in user prompt
        self.assertIn("Source 1 (page 4): (biology_notes.pdf)", passed_user)
        self.assertIn("Chloroplasts capture light energy", passed_user)
        self.assertIn("Question: What does photosynthesis make?", passed_user)

    def test_13_source_metadata_preservation_multidoc(self):
        chunks = [
            {
                "text": "Text from doc A.",
                "metadata": {"filename": "doc_a.pdf", "page": 1, "chunk_index": 1},
            },
            {
                "text": "Text from doc B.",
                "metadata": {"filename": "doc_b.pdf", "page": 12, "chunk_index": 5},
            },
        ]
        context = build_context(chunks)
        self.assertIn("Source 1 (page 1): (doc_a.pdf)\nText from doc A.", context)
        self.assertIn("Source 2 (page 12): (doc_b.pdf)\nText from doc B.", context)

    def test_14_empty_retrieval_chunks_raises(self):
        with self.assertRaises(ValueError):
            generate_answer("Any question?", [])

    def test_15_streaming_answer_generation(self):
        mock_provider = MagicMock(spec=BaseLLMProvider)
        mock_provider.generate_stream.return_value = iter(["Token1 ", "Token2."])

        chunks = [{"text": "Valid text.", "metadata": {"page": 1, "chunk_index": 1}}]
        result = generate_answer("Question?", chunks, provider=mock_provider, stream=True)
        tokens = list(result)
        self.assertEqual(tokens, ["Token1 ", "Token2."])
        mock_provider.generate_stream.assert_called_once()

    def test_16_ollama_format_json_enforced_when_requested_or_prompt_demands_json(self):
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {
            "model": "qwen2.5:7b",
            "message": {"role": "assistant", "content": '{"questions": []}'},
            "done": True,
        }

        with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
            result = self.provider.generate(
                "You are a generator. Return ONLY valid JSON matching schema.",
                "Generate 1 question.",
            )
            self.assertEqual(result, '{"questions": []}')
            mock_post.assert_called_once()
            call_json = mock_post.call_args[1]["json"]
            self.assertEqual(call_json.get("format"), "json")

    def test_17_ollama_provider_integration_with_summarizer_and_questions(self):
        provider = get_llm_provider("ollama")
        self.assertIsInstance(provider, OllamaLLMProvider)
        self.assertEqual(provider.model_name, "qwen2.5:7b")
        self.assertTrue(provider.is_configured())

        # Test generation with mock
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {
            "model": "qwen2.5:7b",
            "message": {"role": "assistant", "content": "Summary text generated by Qwen."},
            "done": True,
        }
        with patch("httpx.Client.post", return_value=mock_resp):
            output = provider.generate("System prompt", "User prompt")
            self.assertEqual(output, "Summary text generated by Qwen.")

    def test_18_all_eleven_features_use_centralized_ollama_provider_and_no_cloud_apis(self):
        """Verify all 11 AI features route to centralized OllamaLLMProvider without cloud APIs."""
        from src.generation.explanation_generator import explain_concept
        from src.generation.flashcard_generator import generate_flashcards
        from src.generation.follow_up_generator import generate_follow_up_questions
        from src.generation.question_generator import generate_practice_questions
        from src.generation.quiz_evaluator import evaluate_quiz
        from src.generation.study_notes_generator import generate_study_notes
        from src.generation.summarizer import summarize_document

        provider = get_llm_provider("ollama")
        self.assertIsInstance(provider, OllamaLLMProvider)
        self.assertEqual(provider.model_name, "qwen2.5:7b")

        mock_store = MagicMock()
        mock_store.get_document.return_value = {
            "document_id": "doc_test",
            "filename": "cell.pdf",
            "page_count": 1,
            "chunk_count": 1,
            "pages_indexed": [1],
        }
        chunk = {
            "chunk_id": "c_test1",
            "chunk_index": 1,
            "text": "Cellular respiration produces ATP in the presence of oxygen.",
            "page": 1,
            "page_range": "1",
            "filename": "cell.pdf",
            "document_id": "doc_test",
            "token_count": 12,
            "chunk_type": "text",
        }
        mock_store.get_document_chunks.return_value = [chunk]
        mock_store.get_document_page_numbers.return_value = [1]
        chunks = [chunk]

        # Ensure Groq client is completely unconfigured / not called
        groq_patch_target = "groq.Groq" if "groq" in sys.modules else "src.generation.llm_provider.Groq"
        with (
            patch.object(config, "LLM_PROVIDER", "ollama"),
            patch.object(config, "GROQ_API_KEY", ""),
            patch(groq_patch_target, side_effect=AssertionError("Groq client must NEVER be called")),
        ):
            mock_generate = MagicMock(return_value="Valid mock generation")
            mock_json_generate = MagicMock(return_value=json.dumps({
                "questions": [{
                    "question_id": "q1",
                    "question_type": "mcq",
                    "difficulty": "medium",
                    "question": "What does cellular respiration produce?",
                    "options": ["ATP", "Glucose", "Light", "Chlorophyll"],
                    "correct_answer": "ATP",
                    "explanation": "Cellular respiration produces ATP.",
                    "source_chunk_ids": ["c_test1"],
                }],
                "flashcards": [{
                    "front": "What is produced in cellular respiration?",
                    "back": "Adenosine triphosphate (ATP) is produced.",
                    "source_chunk_ids": ["c_test1"],
                }],
                "follow_up_questions": ["How is ATP utilized?"],
                "evaluations": [{
                    "question_id": "q1",
                    "earned_points": 1.0,
                    "status": "correct",
                    "feedback": "Correct!",
                    "explanation": "Respiration produces ATP.",
                }],
            }))

            mock_prov = MagicMock(spec=OllamaLLMProvider)
            mock_prov.provider_name = "ollama"
            mock_prov.model_name = "qwen2.5:7b"
            mock_prov.generate = mock_generate
            mock_prov.generate_stream = MagicMock(return_value=iter(["streamed ", "answer"]))

            # 1. Summarization
            summarize_document(mock_store, "doc_test", provider=mock_prov)
            self.assertTrue(mock_generate.called)

            # 2. Practice Questions & 3. MCQ & 4. Quiz
            mock_prov.generate = mock_json_generate
            q_res = generate_practice_questions(
                mock_store, "doc_test", count=1, question_type="mcq", provider=mock_prov
            )
            self.assertEqual(len(q_res.questions), 1)

            # 5. Quiz Answer Evaluation (LLM mode)
            eval_res = evaluate_quiz(
                q_res.to_dict(),
                {"q1": "ATP"},
                evaluator_mode="llm",
                provider=mock_prov,
                store=mock_store,
            )
            self.assertEqual(eval_res.earned_points, 1.0)

            # 6. Quick Q&A & 7. RAG Q&A
            ans = generate_answer("What is produced?", chunks, provider=mock_prov)
            self.assertIsNotNone(ans)

            # 8. Follow-up Question Generation
            fus = generate_follow_up_questions("What is produced?", "ATP", chunks, provider=mock_prov)
            self.assertTrue(len(fus) >= 1)

            # 9. Document Concept Explanation
            mock_prov.generate = MagicMock(return_value="Detailed ATP explanation [Page 1].")
            exp = explain_concept("ATP", chunks, provider=mock_prov)
            self.assertIn("Detailed ATP", exp)

            # 10. Study Notes Generation
            mock_prov.generate = MagicMock(return_value="# Study Notes\n## Overview\nCellular respiration.")
            notes = generate_study_notes(mock_store, "doc_test", provider=mock_prov)
            self.assertEqual(notes.filename, "cell.pdf")

            # 11. Flashcard Generation
            mock_prov.generate = mock_json_generate
            fcs = generate_flashcards(mock_store, "doc_test", count=1, provider=mock_prov)
            self.assertEqual(len(fcs.cards), 1)


if __name__ == "__main__":
    unittest.main()
