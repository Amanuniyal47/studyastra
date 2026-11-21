import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pymupdf
from streamlit.testing.v1 import AppTest

import src.config as config
from src.generation.llm_provider import BaseLLMProvider, LLMProviderError
import src.generation.question_generator as question_generator
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.vector_store import FAISSVectorStore

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


class _NamedUpload:
    def __init__(self, data: bytes, name: str):
        self._data = data
        self.name = name

    def getvalue(self) -> bytes:
        return self._data


def _make_pdf(pages: list[str], name: str) -> _NamedUpload:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(50, 50, 545, 780), text, fontsize=10)
    raw = doc.tobytes()
    doc.close()
    return _NamedUpload(raw, name=name)


class _FakeUIEmbedder:
    def __init__(self, dim: int = 4):
        self._dim = dim
        self.encode_calls = 0

    def get_embedding_dimension(self) -> int:
        return self._dim

    def encode(
        self,
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    ):
        self.encode_calls += 1
        rows = []
        for text in texts:
            lower = text.lower()
            rows.append([
                float("photosynthesis" in lower or "plant" in lower or "biology" in lower),
                float("faiss" in lower or "vector" in lower),
                float("history" in lower or "revolution" in lower),
                0.05,
            ])
        return np.array(rows, dtype=np.float32)


class TestStage4StreamlitIntegration(unittest.TestCase):
    def test_01_initial_empty_knowledge_base(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            with patch.object(config, "VECTOR_STORE_DIR", store_dir):
                at = AppTest.from_file(APP_PATH).run()

            self.assertFalse(at.exception)
            self.assertEqual(at.title[0].value, "PDF Chat - RAG Question Answering System")
            self.assertEqual(at.sidebar.metric[0].value, "0")
            self.assertEqual(at.sidebar.metric[1].value, "0")
            self.assertTrue(
                any("Upload a PDF to start asking questions." in info.value for info in at.info)
            )

    def test_02_index_pdf_duplicate_prevention_and_no_rerun_reindexing(self):
        fake_model = _FakeUIEmbedder()
        pdf_bio = _make_pdf(
            [
                "Biology page 1: Photosynthesis in plant cells converts sunlight into glucose.",
                "Biology page 2: Plant chloroplasts store chlorophyll pigments.",
            ],
            name="biology.pdf",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
                patch("streamlit.file_uploader", return_value=pdf_bio),
            ):
                # Initial upload should NOT auto-index before button click
                at = AppTest.from_file(APP_PATH).run()
                self.assertFalse(at.exception)
                self.assertEqual(fake_model.encode_calls, 0)
                self.assertEqual(at.sidebar.metric[0].value, "0")
                index_btn = at.button(key="process_index_pdf_btn")
                self.assertFalse(index_btn.disabled)

                # Explicit click indexes PDF once
                at = index_btn.click().run()
                self.assertFalse(at.exception)
                self.assertEqual(fake_model.encode_calls, 1)
                self.assertEqual(at.sidebar.metric[0].value, "1")
                self.assertEqual(at.sidebar.metric[1].value, "2")

                # Duplicate upload detected BEFORE indexing and button is disabled
                self.assertTrue(any("Duplicate detected" in w.value for w in at.warning))
                self.assertTrue(at.button(key="process_index_pdf_btn").disabled)

                # Another rerun must NOT trigger additional embedding calls
                at = at.run()
                self.assertFalse(at.exception)
                self.assertEqual(fake_model.encode_calls, 1)

    def test_03_multi_pdf_retrieval_filtering_deletion_and_restart_persistence(self):
        fake_model = _FakeUIEmbedder()
        pdf_bio = _make_pdf(
            ["Biology notes: Photosynthesis happens in plant cells."],
            name="biology.pdf",
        )
        pdf_cs = _make_pdf(
            ["FAISS stores dense vector embeddings for fast search."],
            name="faiss_guide.pdf",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"

            # Index PDF 1 via Streamlit UI
            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
                patch("streamlit.file_uploader", return_value=pdf_bio),
            ):
                at = AppTest.from_file(APP_PATH).run()
                at = at.button(key="process_index_pdf_btn").click().run()
                self.assertFalse(at.exception)
                self.assertEqual(at.sidebar.metric[0].value, "1")

            # Index PDF 2 via a fresh Streamlit session (simulates restart + second PDF)
            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
                patch("streamlit.file_uploader", return_value=pdf_cs),
            ):
                at = AppTest.from_file(APP_PATH).run()
                self.assertFalse(at.exception)
                self.assertEqual(at.sidebar.metric[0].value, "1")
                at = at.button(key="process_index_pdf_btn").click().run()
                self.assertFalse(at.exception)
                self.assertEqual(at.sidebar.metric[0].value, "2")
                self.assertEqual(at.sidebar.metric[1].value, "2")

            # Restart Streamlit completely (fresh AppTest) & verify persistence + retrieval across all docs
            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
            ):
                at = AppTest.from_file(APP_PATH).run()
                self.assertFalse(at.exception)
                self.assertEqual(at.sidebar.metric[0].value, "2")
                self.assertEqual(at.sidebar.metric[1].value, "2")

                # Retrieve across all documents
                at.text_input(key="retrieval_query_input").set_value("How does FAISS search vectors?")
                search_btn = [b for b in at.button if b.label == "Search Vectors"][0]
                at = search_btn.click().run()
                self.assertFalse(at.exception)
                labels = [exp.label for exp in at.expander]
                self.assertTrue(
                    any(
                        "faiss_guide.pdf" in lbl
                        and "Page 1" in lbl
                        and "Chunk ID:" in lbl
                        and "Cosine Similarity:" in lbl
                        for lbl in labels
                    ),
                    f"Missing expected metadata in retrieval results: {labels}",
                )

                # Retrieve filtered to biology.pdf only
                docs = at.session_state["vector_store"].list_documents()
                bio_doc_id = next(d["document_id"] for d in docs if d["filename"] == "biology.pdf")
                at.selectbox(key="retrieval_doc_filter").set_value(bio_doc_id)
                at.text_input(key="retrieval_query_input").set_value("FAISS vector search")
                search_btn = [b for b in at.button if b.label == "Search Vectors"][0]
                at = search_btn.click().run()
                self.assertFalse(at.exception)
                filtered_labels = [exp.label for exp in at.expander]
                self.assertEqual(len(filtered_labels), 1)
                self.assertIn("biology.pdf", filtered_labels[0])

                # Intentional two-step delete of biology.pdf in sidebar
                at = at.sidebar.button(key=f"delete_doc_{bio_doc_id}").click().run()
                self.assertFalse(at.exception)
                # Not deleted yet until confirmed
                self.assertEqual(at.sidebar.metric[0].value, "2")
                at = at.sidebar.button(key=f"confirm_delete_{bio_doc_id}").click().run()
                self.assertFalse(at.exception)
                self.assertEqual(at.sidebar.metric[0].value, "1")
                self.assertEqual(at.sidebar.metric[1].value, "1")

                # Verify faiss_guide.pdf remains searchable after deletion & restart
                at_restarted = AppTest.from_file(APP_PATH).run()
                self.assertEqual(at_restarted.sidebar.metric[0].value, "1")
                at_restarted.text_input(key="retrieval_query_input").set_value("FAISS vector")
                search_btn = [b for b in at_restarted.button if b.label == "Search Vectors"][0]
                at_restarted = search_btn.click().run()
                remaining_labels = [exp.label for exp in at_restarted.expander]
                self.assertEqual(len(remaining_labels), 1)
                self.assertIn("faiss_guide.pdf", remaining_labels[0])

    def test_04_groq_qa_compatibility_and_missing_api_key_independence(self):
        fake_model = _FakeUIEmbedder()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="FAISS indexes vectors (Source 1)."))]

        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            with patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model):
                svc = EmbeddingService(model_name="all-MiniLM-L6-v2")
                store = FAISSVectorStore(store_dir=store_dir, embedding_service=svc)
                store.add_pdf(
                    _make_pdf(["FAISS indexes dense vectors for similarity search."], name="cs.pdf"),
                    embedding_service=svc,
                )

            # A. Existing Groq Q&A works when GROQ_API_KEY is present
            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch.object(config, "LLM_PROVIDER", "groq"),
                patch.object(config, "GROQ_API_KEY", "test-key"),
                patch("src.generation.answer_generator.GROQ_API_KEY", "test-key"),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
                patch("src.generation.answer_generator.Groq") as mock_groq_cls,
            ):
                mock_groq_cls.return_value.chat.completions.create.return_value = mock_response
                at = AppTest.from_file(APP_PATH).run()
                at = at.text_input(key="groq_qa_question").set_value("What does FAISS do?").run()
                self.assertFalse(at.exception)
                self.assertTrue(
                    any("FAISS indexes vectors" in m.value for m in at.markdown),
                    f"Groq Q&A answer missing: {[m.value for m in at.markdown]}",
                )

            # B. Retrieval testing still works even when GROQ_API_KEY is missing
            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch.object(config, "LLM_PROVIDER", "groq"),
                patch.object(config, "GROQ_API_KEY", None),
                patch("src.generation.answer_generator.GROQ_API_KEY", None),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
            ):
                at = AppTest.from_file(APP_PATH).run()
                self.assertFalse(at.exception)
                self.assertTrue(any("GROQ_API_KEY is not configured" in w.value for w in at.warning))

                at.text_input(key="retrieval_query_input").set_value("FAISS vector search")
                search_btn = [b for b in at.button if b.label == "Search Vectors"][0]
                at = search_btn.click().run()
                self.assertFalse(at.exception)
                self.assertTrue(any("cs.pdf" in exp.label for exp in at.expander))

    def test_05_corrupted_store_requires_explicit_confirmation_before_reset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            store_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = store_dir / "manifest.json"
            manifest_path.write_text("{corrupted json", encoding="utf-8")

            with patch.object(config, "VECTOR_STORE_DIR", store_dir):
                at = AppTest.from_file(APP_PATH).run()
                self.assertFalse(at.exception)
                self.assertTrue(any("corrupted" in err.value.lower() for err in at.error))
                # Must NOT auto-delete user data
                self.assertTrue(manifest_path.exists())
                # Reset button disabled until confirmation checkbox checked
                self.assertTrue(at.button(key="reset_vector_store_btn").disabled)

                # Explicitly confirm and reset
                at = at.checkbox(key="confirm_reset_vector_store").check().run()
                self.assertFalse(at.button(key="reset_vector_store_btn").disabled)
                at = at.button(key="reset_vector_store_btn").click().run()
                self.assertFalse(at.exception)
                self.assertFalse(manifest_path.exists())
                self.assertEqual(at.sidebar.metric[0].value, "0")

    def test_06_incompatible_embedding_model_detected_without_deleting_data(self):
        fake_model = _FakeUIEmbedder()
        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            with patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model):
                svc = EmbeddingService(model_name="all-MiniLM-L6-v2")
                store = FAISSVectorStore(store_dir=store_dir, embedding_service=svc)
                store.add_pdf(
                    _make_pdf(["Sample indexed study notes."], name="notes.pdf"),
                    embedding_service=svc,
                )

            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch.object(config, "EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
            ):
                at = AppTest.from_file(APP_PATH).run()
                self.assertFalse(at.exception)
                self.assertTrue(any("Incompatible" in err.value for err in at.error))
                # Persisted manifest must still exist untouched
                self.assertTrue((store_dir / "manifest.json").exists())
                self.assertTrue(at.button(key="reset_vector_store_btn").disabled)


class _MockSummaryProvider(BaseLLMProvider):
    provider_name = "mock_groq"

    def __init__(self, *, configured: bool = True, fail: bool = False):
        super().__init__(model_name="mock-summary-model")
        self._configured = configured
        self._fail = fail
        self.calls: list[dict] = []

    def is_configured(self) -> bool:
        return self._configured

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        if self._fail:
            raise LLMProviderError("Groq rate limit reached.")
        return "Grounded mock summary preserving key facts [p. 1]."


class TestStage61cSummaryTabUI(unittest.TestCase):
    def _seed_two_pdfs(self, store_dir: Path, fake_model: _FakeUIEmbedder) -> tuple[str, str]:
        with patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model):
            svc = EmbeddingService(model_name="all-MiniLM-L6-v2")
            store = FAISSVectorStore(store_dir=store_dir, embedding_service=svc)
            # Sparse indexed pages (page 2 intentionally blank -> actual indexed pages are [1, 3])
            r_bio = store.add_pdf(
                _make_pdf(
                    [
                        "Chapter 1: Photosynthesis\nPhotosynthesis in plant cells produces glucose.",
                        "",
                        "Chapter 2: Plant Pigments\nChlorophyll absorbs sunlight for photosynthesis.",
                    ],
                    name="biology.pdf",
                ),
                embedding_service=svc,
            )
            r_cs = store.add_pdf(
                _make_pdf(
                    ["FAISS Guide\nFAISS indexes dense vectors for similarity search."],
                    name="faiss_guide.pdf",
                ),
                embedding_service=svc,
            )
            return r_bio["document_id"], r_cs["document_id"]

    def test_01_summary_tab_missing_api_key_does_not_break_retrieval(self):
        fake_model = _FakeUIEmbedder()
        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            self._seed_two_pdfs(store_dir, fake_model)
            unconfigured = _MockSummaryProvider(configured=False)

            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch.object(config, "GROQ_API_KEY", None),
                patch("src.generation.llm_provider.get_llm_provider", return_value=unconfigured),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
            ):
                at = AppTest.from_file(APP_PATH).run()
                self.assertFalse(at.exception)
                self.assertTrue(
                    any("GROQ_API_KEY is not configured" in w.value for w in at.warning)
                )
                # Retrieval Testing tab still works without API key
                at.text_input(key="retrieval_query_input").set_value("FAISS vector search")
                search_btn = [b for b in at.button if b.label == "Search Vectors"][0]
                at = search_btn.click().run()
                self.assertFalse(at.exception)
                self.assertTrue(any("faiss_guide.pdf" in exp.label for exp in at.expander))

    def test_02_full_summary_sparse_pages_no_rerun_calls_and_delete_invalidates_cache(self):
        fake_model = _FakeUIEmbedder()
        provider = _MockSummaryProvider(configured=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            bio_id, _ = self._seed_two_pdfs(store_dir, fake_model)

            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch("src.generation.llm_provider.get_llm_provider", return_value=provider),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
            ):
                at = AppTest.from_file(APP_PATH).run()
                self.assertFalse(at.exception)
                # Explicit button required: zero LLM calls on initial load
                self.assertEqual(len(provider.calls), 0)

                # Verify Page Range selectboxes use ACTUAL indexed pages [1, 3] (not blank page 2)
                at.selectbox(key="summary_doc_select").set_value(bio_id)
                at = at.selectbox(key="summary_mode_select").set_value("Page Range").run()
                self.assertEqual(at.selectbox(key="summary_start_page").options, ["1", "3"])
                self.assertEqual(at.selectbox(key="summary_end_page").options, ["1", "3"])

                # Switch back to Full Document, Short -> click Generate Summary
                at = at.selectbox(key="summary_mode_select").set_value("Full Document").run()
                at.selectbox(key="summary_length_select").set_value("Short")
                at = at.button(key="generate_summary_btn").click().run()

                self.assertFalse(at.exception)
                self.assertEqual(len(provider.calls), 1)
                # Single-PDF isolation: faiss_guide.pdf must never appear in biology prompt
                self.assertNotIn("faiss_guide.pdf", provider.calls[0]["user_prompt"])
                # Verified source metadata rendered (actual indexed pages 1, 3)
                all_text = " ".join(
                    [m.value for m in at.markdown]
                    + [c.value for c in at.caption]
                    + [s.value for s in at.subheader]
                )
                self.assertIn("Grounded mock summary preserving key facts", all_text)
                self.assertIn("biology.pdf", all_text)
                self.assertIn("Pages 1, 3", all_text)
                self.assertIn("mock_groq", all_text)
                self.assertIn("mock-summary-model", all_text)
                # Summaries are never indexed into FAISS (vector count stays 3)
                self.assertEqual(at.sidebar.metric[1].value, "3")

                # Rerun must NOT re-call the LLM
                at = at.run()
                self.assertEqual(len(provider.calls), 1)

                # Deleting biology.pdf immediately invalidates its cached summary
                self.assertIn(bio_id, at.session_state["cached_summaries"])
                at = at.sidebar.button(key=f"delete_doc_{bio_id}").click().run()
                at = at.sidebar.button(key=f"confirm_delete_{bio_id}").click().run()
                self.assertFalse(at.exception)
                self.assertNotIn(bio_id, at.session_state.filtered_state.get("cached_summaries", {}))
                self.assertEqual(at.sidebar.metric[0].value, "1")

    def test_03_api_failure_clears_stale_summary_and_displays_error(self):
        fake_model = _FakeUIEmbedder()
        good_provider = _MockSummaryProvider(configured=True, fail=False)
        bad_provider = _MockSummaryProvider(configured=True, fail=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            bio_id, _ = self._seed_two_pdfs(store_dir, fake_model)

            # First generate a valid summary
            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch("src.generation.llm_provider.get_llm_provider", return_value=good_provider),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
            ):
                at = AppTest.from_file(APP_PATH).run()
                at.selectbox(key="summary_doc_select").set_value(bio_id)
                at = at.button(key="generate_summary_btn").click().run()
                self.assertIn(bio_id, at.session_state["cached_summaries"])

            # Next generation fails -> stale summary MUST be cleared and NOT displayed
            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch("src.generation.llm_provider.get_llm_provider", return_value=bad_provider),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
            ):
                at = at.button(key="generate_summary_btn").click().run()
                self.assertFalse(at.exception)
                self.assertTrue(
                    any("Summary generation failed" in err.value for err in at.error),
                    f"Expected summary error, got: {[e.value for e in at.error]}",
                )
                self.assertNotIn(bio_id, at.session_state.filtered_state.get("cached_summaries", {}))
                self.assertFalse(
                    any("Grounded mock summary" in m.value for m in at.markdown)
                )

    def test_04_section_semantic_fallback_and_topic_summary_in_ui(self):
        fake_model = _FakeUIEmbedder()
        provider = _MockSummaryProvider(configured=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            bio_id, _ = self._seed_two_pdfs(store_dir, fake_model)

            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch("src.generation.llm_provider.get_llm_provider", return_value=provider),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
            ):
                at = AppTest.from_file(APP_PATH).run()
                at.selectbox(key="summary_doc_select").set_value(bio_id)

                # Section mode semantic fallback (not verbatim heading, high semantic similarity)
                at = at.selectbox(key="summary_mode_select").set_value("Section").run()
                at.text_input(key="summary_section_input").set_value("plant photosynthesis energy mechanism")
                at = at.button(key="generate_summary_btn").click().run()
                self.assertFalse(at.exception)
                self.assertTrue(
                    any("semantic fallback" in info.value.lower() for info in at.info),
                    f"Expected semantic fallback notice, got: {[i.value for i in at.info]}",
                )

                # Section mode with unrelated heading reports clear error instead of unreliable summary
                at.text_input(key="summary_section_input").set_value("quantum mechanics wavefunctions")
                at = at.button(key="generate_summary_btn").click().run()
                self.assertFalse(at.exception)
                self.assertTrue(
                    any("could not be reliably identified" in err.value.lower() for err in at.error),
                    f"Expected unreliable section error, got: {[e.value for e in at.error]}",
                )

                # Topic mode using FAISS retrieval strictly for biology.pdf
                at = at.selectbox(key="summary_mode_select").set_value("Topic").run()
                at.text_input(key="summary_topic_input").set_value("photosynthesis plant")
                at = at.button(key="generate_summary_btn").click().run()
                self.assertFalse(at.exception)
                self.assertEqual(at.session_state["cached_summaries"][bio_id]["mode"], "topic")
                self.assertNotIn("faiss_guide.pdf", provider.calls[-1]["user_prompt"])


class _MockQuestionUIProvider(BaseLLMProvider):
    provider_name = "mock_groq"

    def __init__(
        self,
        *,
        configured: bool = True,
        fail: bool = False,
        shortfall_limit: int | None = None,
    ):
        super().__init__(model_name="mock-question-model")
        self._configured = configured
        self._fail = fail
        self._shortfall_limit = shortfall_limit
        self.calls: list[dict] = []
        self._counter = 0

    def is_configured(self) -> bool:
        return self._configured

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        if self._fail:
            raise LLMProviderError("Groq rate limit reached.")

        if "quiz grader" in system_prompt.lower():
            # Return valid grader JSON evaluating short-answer items
            qids = re.findall(r"Question ID: ([^\n]+)", user_prompt)
            evals = []
            for qid in qids:
                evals.append(
                    {
                        "question_id": qid,
                        "score": 1.0,
                        "feedback": f"Grounded response for {qid}.",
                    }
                )
            return json.dumps({"evaluations": evals})

        chunk_ids = [
            cid.strip()
            for cid in re.findall(r"\[Chunk ID: ([^|\]]+) \| Source:", user_prompt)
        ]

        mixed_match = re.search(r"distributed as: ([^.\n]+)\.", user_prompt)
        plan: list[str] = []
        if mixed_match:
            for part in mixed_match.group(1).split(","):
                m = re.match(r"\s*(\d+)\s+(mcq|short_answer|true_false)\s*", part)
                if m:
                    plan.extend([m.group(2)] * int(m.group(1)))
        else:
            single_match = re.search(
                r"Generate exactly (\d+) '(mcq|short_answer|true_false)'", user_prompt
            )
            if single_match:
                plan = [single_match.group(2)] * int(single_match.group(1))
            else:
                plan = ["mcq"]

        diff_match = re.search(r"Difficulty:\s*(Easy|Medium|Hard):", user_prompt)
        difficulty = diff_match.group(1).lower() if diff_match else "medium"

        stems = [
            "chloroplast sunlight glucose conversion",
            "chlorophyll pigment light absorption",
            "dense vector embedding similarity search",
            "plant leaf photosynthetic energy pathway",
            "cellular biochemical reaction grounding",
        ]

        items = []
        for idx, qtype in enumerate(plan):
            if self._shortfall_limit is not None and self._counter >= self._shortfall_limit:
                break
            self._counter += 1
            cid = chunk_ids[idx % len(chunk_ids)]
            uid = self._counter
            phrase = stems[(uid - 1) % len(stems)]
            if qtype == "mcq":
                items.append(
                    {
                        "question_type": "mcq",
                        "difficulty": difficulty,
                        "question": f"MCQ {uid}: How does {phrase} operate?",
                        "options": [
                            f"Correct choice {uid}",
                            f"Distractor alpha {uid}",
                            f"Distractor beta {uid}",
                            f"Distractor gamma {uid}",
                        ],
                        "correct_answer": f"Correct choice {uid}",
                        "explanation": f"Grounded MCQ explanation {uid} from {cid}.",
                        "source_chunk_ids": [cid],
                    }
                )
            elif qtype == "short_answer":
                items.append(
                    {
                        "question_type": "short_answer",
                        "difficulty": difficulty,
                        "question": f"Short {uid}: Explain {phrase} clearly.",
                        "expected_answer": f"Expected short response {uid}.",
                        "explanation": f"Grounded short explanation {uid} from {cid}.",
                        "source_chunk_ids": [cid],
                    }
                )
            else:
                items.append(
                    {
                        "question_type": "true_false",
                        "difficulty": difficulty,
                        "question": f"TF {uid}: {phrase} happens in reverse.",
                        "correct_answer": "False",
                        "correct_statement": f"TF {uid}: {phrase} happens in the forward direction.",
                        "explanation": f"Grounded TF explanation {uid} from {cid}.",
                        "source_chunk_ids": [cid],
                    }
                )

        return json.dumps({"questions": items})


class TestStage62cPracticeQuestionsTabUI(unittest.TestCase):
    def _seed_two_pdfs(self, store_dir: Path, fake_model: _FakeUIEmbedder) -> tuple[str, str]:
        with patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model):
            svc = EmbeddingService(model_name="all-MiniLM-L6-v2")
            store = FAISSVectorStore(store_dir=store_dir, embedding_service=svc)
            r_bio = store.add_pdf(
                _make_pdf(
                    [
                        "Chapter 1: Photosynthesis\nPhotosynthesis in plant cells produces glucose.",
                        "",
                        "Chapter 2: Plant Pigments\nChlorophyll absorbs sunlight for photosynthesis.",
                    ],
                    name="biology.pdf",
                ),
                embedding_service=svc,
            )
            r_cs = store.add_pdf(
                _make_pdf(
                    ["FAISS Guide\nFAISS indexes dense vectors for similarity search."],
                    name="faiss_guide.pdf",
                ),
                embedding_service=svc,
            )
            return r_bio["document_id"], r_cs["document_id"]

    def test_01_empty_kb_and_missing_api_key_keep_retrieval_working(self):
        fake_model = _FakeUIEmbedder()
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_dir = Path(tmpdir) / "empty_store"
            with patch.object(config, "VECTOR_STORE_DIR", empty_dir):
                at = AppTest.from_file(APP_PATH).run()
            self.assertFalse(at.exception)
            self.assertTrue(
                any(
                    "Index at least one PDF first to generate practice questions." in i.value
                    for i in at.info
                )
            )

            store_dir = Path(tmpdir) / "populated_store"
            self._seed_two_pdfs(store_dir, fake_model)
            unconfigured = _MockQuestionUIProvider(configured=False)
            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch.object(config, "GROQ_API_KEY", None),
                patch("src.generation.llm_provider.get_llm_provider", return_value=unconfigured),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
            ):
                at = AppTest.from_file(APP_PATH).run()
                self.assertFalse(at.exception)
                self.assertTrue(
                    any("practice questions" in w.value.lower() for w in at.warning)
                )
                # Retrieval Testing tab remains fully functional
                at.text_input(key="retrieval_query_input").set_value("FAISS vector search")
                search_btn = [b for b in at.button if b.label == "Search Vectors"][0]
                at = search_btn.click().run()
                self.assertFalse(at.exception)
                self.assertTrue(any("faiss_guide.pdf" in exp.label for exp in at.expander))

    def test_02_mixed_mode_explicit_generation_answer_toggle_and_argument_forwarding(self):
        fake_model = _FakeUIEmbedder()
        provider = _MockQuestionUIProvider(configured=True)
        recorded_calls: list[dict] = []
        real_generate = question_generator.generate_practice_questions

        def _spy_generate(*args, **kwargs):
            recorded_calls.append({"args": args, "kwargs": dict(kwargs)})
            return real_generate(*args, **kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            bio_id, cs_id = self._seed_two_pdfs(store_dir, fake_model)

            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch("src.generation.llm_provider.get_llm_provider", return_value=provider),
                patch(
                    "src.generation.question_generator.generate_practice_questions",
                    side_effect=_spy_generate,
                ),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
            ):
                at = AppTest.from_file(APP_PATH).run()
                self.assertFalse(at.exception)
                # Explicit button required: zero LLM calls on initial render
                self.assertEqual(len(provider.calls), 0)

                at.selectbox(key="questions_doc_select").set_value(bio_id)
                at.selectbox(key="questions_scope_select").set_value("Full Document")
                at.selectbox(key="questions_type_select").set_value("Mixed")
                at.selectbox(key="questions_difficulty_select").set_value("Hard")
                at.number_input(key="questions_count_input").set_value(4)
                at = at.button(key="generate_questions_btn").click().run()

                self.assertFalse(at.exception)
                self.assertEqual(len(recorded_calls), 1)
                fwd = recorded_calls[0]["kwargs"]
                self.assertEqual(recorded_calls[0]["args"][1], bio_id)
                self.assertEqual(fwd["question_type"], "mixed")
                self.assertEqual(fwd["difficulty"], "hard")
                self.assertEqual(fwd["count"], 4)
                self.assertEqual(fwd["scope"], "full")
                self.assertIsNotNone(fwd["progress_callback"])

                calls_after_gen = len(provider.calls)
                self.assertGreaterEqual(calls_after_gen, 1)

                # Verify distribution & verified metadata shown, and answers HIDDEN by default
                all_markdown = "\n".join(m.value for m in at.markdown)
                all_captions = "\n".join(c.value for c in at.caption)
                self.assertIn("MCQ 1:", all_markdown)
                self.assertIn("Actual type distribution:", all_captions)
                self.assertIn("**MCQ:** 2", all_captions)
                self.assertIn("**Short Answer:** 1", all_captions)
                self.assertIn("**True / False:** 1", all_captions)
                self.assertIn("biology.pdf", all_captions)
                self.assertIn("Pages 1, 3", all_captions)
                self.assertNotIn("Grounded MCQ explanation 1", all_markdown)
                # Never indexed in FAISS (vector count stays 3)
                self.assertEqual(at.sidebar.metric[1].value, "3")

                # Toggle Show Answers & Explanations -> reveals answers WITHOUT calling LLM again
                at = at.checkbox(key=f"show_answers_{bio_id}").check().run()
                self.assertFalse(at.exception)
                self.assertEqual(len(provider.calls), calls_after_gen)
                revealed_md = "\n".join(m.value for m in at.markdown)
                self.assertIn("Grounded MCQ explanation 1", revealed_md)
                self.assertIn("Expected Answer:", revealed_md)
                self.assertIn("Correct Statement:", revealed_md)

                # Switching selected PDF MUST NOT display biology.pdf's cached questions
                at = at.selectbox(key="questions_doc_select").set_value(cs_id).run()
                self.assertFalse(at.exception)
                self.assertEqual(len(provider.calls), calls_after_gen)
                cs_md = "\n".join(m.value for m in at.markdown)
                self.assertNotIn("MCQ 1:", cs_md)

    def test_03_page_range_topic_individual_types_and_shortfall_notice(self):
        fake_model = _FakeUIEmbedder()

        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            bio_id, _ = self._seed_two_pdfs(store_dir, fake_model)

            # A. Page Range (sparse actual pages ["1", "3"]) + Short Answer + Shortfall notice
            shortfall_provider = _MockQuestionUIProvider(configured=True, shortfall_limit=2)
            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch(
                    "src.generation.llm_provider.get_llm_provider",
                    return_value=shortfall_provider,
                ),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
            ):
                at = AppTest.from_file(APP_PATH).run()
                at.selectbox(key="questions_doc_select").set_value(bio_id)
                at = at.selectbox(key="questions_scope_select").set_value("Page Range").run()
                self.assertEqual(at.selectbox(key="questions_start_page").options, ["1", "3"])
                self.assertEqual(at.selectbox(key="questions_end_page").options, ["1", "3"])

                at.selectbox(key="questions_type_select").set_value("Short Answer")
                at.selectbox(key="questions_difficulty_select").set_value("Easy")
                at.number_input(key="questions_count_input").set_value(5)
                at = at.button(key="generate_questions_btn").click().run()
                self.assertFalse(at.exception)
                self.assertTrue(
                    any("Requested 5" in w.value and "yielded 2" in w.value for w in at.warning),
                    f"Expected shortfall notice, got warnings: {[w.value for w in at.warning]}",
                )

            # B. Topic scope + True / False + MCQ
            topic_provider = _MockQuestionUIProvider(configured=True)
            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch(
                    "src.generation.llm_provider.get_llm_provider",
                    return_value=topic_provider,
                ),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
            ):
                at = AppTest.from_file(APP_PATH).run()
                at.selectbox(key="questions_doc_select").set_value(bio_id)
                at = at.selectbox(key="questions_scope_select").set_value("Topic").run()
                at.text_input(key="questions_topic_input").set_value("photosynthesis plant")
                at.number_input(key="questions_topic_top_k").set_value(2)
                at.selectbox(key="questions_type_select").set_value("True / False")
                at.number_input(key="questions_count_input").set_value(2)
                at = at.button(key="generate_questions_btn").click().run()
                self.assertFalse(at.exception)
                self.assertEqual(at.session_state["cached_questions"][bio_id]["scope"], "topic")
                self.assertEqual(
                    at.session_state["cached_questions"][bio_id]["question_type"], "true_false"
                )
                self.assertNotIn("faiss_guide.pdf", topic_provider.calls[-1]["user_prompt"])

                # Switch to MCQ and re-generate cleanly
                at.selectbox(key="questions_type_select").set_value("MCQ")
                at = at.button(key="generate_questions_btn").click().run()
                self.assertFalse(at.exception)
                self.assertEqual(
                    at.session_state["cached_questions"][bio_id]["question_type"], "mcq"
                )

    def test_04_delete_invalidation_and_provider_error_clears_stale_questions(self):
        fake_model = _FakeUIEmbedder()
        good_provider = _MockQuestionUIProvider(configured=True, fail=False)
        bad_provider = _MockQuestionUIProvider(configured=True, fail=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            bio_id, _ = self._seed_two_pdfs(store_dir, fake_model)

            # Generate valid questions first
            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch("src.generation.llm_provider.get_llm_provider", return_value=good_provider),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
            ):
                at = AppTest.from_file(APP_PATH).run()
                at.selectbox(key="questions_doc_select").set_value(bio_id)
                at.number_input(key="questions_count_input").set_value(2)
                at = at.button(key="generate_questions_btn").click().run()
                self.assertIn(bio_id, at.session_state["cached_questions"])

            # API failure clears stale questions and shows clean error
            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch("src.generation.llm_provider.get_llm_provider", return_value=bad_provider),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
            ):
                at = at.button(key="generate_questions_btn").click().run()
                self.assertFalse(at.exception)
                self.assertTrue(
                    any("Question generation failed" in e.value for e in at.error),
                    f"Expected error banner, got: {[e.value for e in at.error]}",
                )
                self.assertNotIn(bio_id, at.session_state.filtered_state.get("cached_questions", {}))
                self.assertFalse(any("MCQ 1:" in m.value for m in at.markdown))

            # Regenerate then delete PDF -> cached questions immediately invalidated
            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch("src.generation.llm_provider.get_llm_provider", return_value=good_provider),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
            ):
                at = at.button(key="generate_questions_btn").click().run()
                self.assertIn(bio_id, at.session_state["cached_questions"])
                at = at.sidebar.button(key=f"delete_doc_{bio_id}").click().run()
                at = at.sidebar.button(key=f"confirm_delete_{bio_id}").click().run()
                self.assertFalse(at.exception)
                self.assertNotIn(bio_id, at.session_state.filtered_state.get("cached_questions", {}))
                self.assertEqual(at.sidebar.metric[0].value, "1")

    def test_05_store_reset_invalidates_cached_questions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            store_dir.mkdir(parents=True, exist_ok=True)
            (store_dir / "manifest.json").write_text("{corrupted", encoding="utf-8")

            with patch.object(config, "VECTOR_STORE_DIR", store_dir):
                at = AppTest.from_file(APP_PATH).run()
                at.session_state["cached_questions"] = {"stale_doc": {"questions": []}}
                at = at.checkbox(key="confirm_reset_vector_store").check().run()
                at = at.button(key="reset_vector_store_btn").click().run()
                self.assertFalse(at.exception)
                self.assertNotIn("cached_questions", at.session_state)


class TestStage63bInteractiveQuizUI(unittest.TestCase):
    def _seed_two_pdfs(self, store_dir: Path, fake_model: _FakeUIEmbedder) -> tuple[str, str]:
        with patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model):
            svc = EmbeddingService(model_name="all-MiniLM-L6-v2")
            store = FAISSVectorStore(store_dir=store_dir, embedding_service=svc)
            r_bio = store.add_pdf(
                _make_pdf(
                    [
                        "Chapter 1: Photosynthesis\nPhotosynthesis in plant cells produces glucose.",
                        "",
                        "Chapter 2: Plant Pigments\nChlorophyll absorbs sunlight for photosynthesis.",
                    ],
                    name="biology.pdf",
                ),
                embedding_service=svc,
            )
            r_cs = store.add_pdf(
                _make_pdf(
                    ["FAISS Guide\nFAISS indexes dense vectors for similarity search."],
                    name="faiss_guide.pdf",
                ),
                embedding_service=svc,
            )
            return r_bio["document_id"], r_cs["document_id"]

    def test_01_study_view_preserved_and_interactive_quiz_mode_switch(self):
        fake_model = _FakeUIEmbedder()
        provider = _MockQuestionUIProvider(configured=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            bio_id, _ = self._seed_two_pdfs(store_dir, fake_model)

            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch("src.generation.llm_provider.get_llm_provider", return_value=provider),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
            ):
                at = AppTest.from_file(APP_PATH).run()
                at.selectbox(key="questions_doc_select").set_value(bio_id)
                at = at.button(key="generate_questions_btn").click().run()
                self.assertFalse(at.exception)
                self.assertIn("cached_questions", at.session_state)

                # Study View is default: Show Answers & Explanations toggle exists and works cleanly
                self.assertTrue(any(f"show_answers_{bio_id}" in chk.key for chk in at.checkbox))
                at = at.checkbox(key=f"show_answers_{bio_id}").check().run()
                self.assertFalse(at.exception)
                self.assertTrue(any("Expected Answer:" in m.value or "Correct Answer:" in m.value for m in at.markdown))

                # Switch to Interactive Quiz: Study checkbox disappears, answer inputs appear, answers HIDDEN
                at = at.radio(key=f"questions_view_mode_{bio_id}").set_value("Interactive Quiz").run()
                self.assertFalse(at.exception)
                self.assertFalse(any(f"show_answers_{bio_id}" in chk.key for chk in at.checkbox))
                self.assertTrue(any(b.label == "Submit Answers" for b in at.button))
                self.assertTrue(any(b.label == "Retry Quiz" for b in at.button))

    def test_02_reruns_preserve_answers_and_submit_renders_score_and_breakdown(self):
        fake_model = _FakeUIEmbedder()
        provider = _MockQuestionUIProvider(configured=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            bio_id, cs_id = self._seed_two_pdfs(store_dir, fake_model)

            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch("src.generation.llm_provider.get_llm_provider", return_value=provider),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
            ):
                at = AppTest.from_file(APP_PATH).run()
                at.selectbox(key="questions_doc_select").set_value(bio_id)
                at.selectbox(key="questions_type_select").set_value("Mixed")
                at.number_input(key="questions_count_input").set_value(3)
                at = at.button(key="generate_questions_btn").click().run()

                # Switch to Interactive Quiz
                at = at.radio(key=f"questions_view_mode_{bio_id}").set_value("Interactive Quiz").run()
                qset = at.session_state["cached_questions"][bio_id]
                questions = qset["questions"]
                qset_id = qset["question_set_id"]

                # Fill answers for the 3 questions (MCQ, Short Answer, True/False)
                for q in questions:
                    qid = q["question_id"]
                    w_key = f"quiz_ans_{bio_id}_{qset_id}_a1_{qid}"
                    if q["question_type"] == "mcq":
                        at.radio(key=w_key).set_value(q["correct_answer"])
                    elif q["question_type"] == "true_false":
                        at.radio(key=w_key).set_value("False")
                    else:
                        at.text_area(key=w_key).set_value("Chloroplasts absorb sunlight.")

                calls_before_rerun = len(provider.calls)

                # Rerun: answers MUST be preserved in saved_answers & inputs, and zero new LLM generation calls
                at = at.run()
                self.assertFalse(at.exception)
                self.assertEqual(len(provider.calls), calls_before_rerun)
                session_answers = at.session_state["quiz_sessions"][bio_id]["answers"]
                self.assertEqual(len(session_answers), 3)

                # Submit Answers: grades quiz, reveals score, percentage, breakdown, and feedback
                submit_btn = [b for b in at.button if b.label == "Submit Answers"][0]
                at = submit_btn.click().run()
                self.assertFalse(at.exception)

                eval_data = at.session_state["quiz_sessions"][bio_id]["result"]
                self.assertIsNotNone(eval_data)
                self.assertEqual(eval_data["total_points"], 3.0)
                self.assertIn("percentage", eval_data)
                self.assertIn("type_breakdown", eval_data)

                all_text = " ".join(
                    [m.value for m in at.markdown]
                    + [c.value for c in at.caption]
                    + [s.value for s in at.subheader]
                )
                self.assertIn("Quiz Results — biology.pdf", all_text)
                self.assertIn("Detailed Feedback", all_text)
                self.assertIn("Breakdown by question type:", all_text)

                # Switching document to cs_id MUST NOT show biology.pdf's quiz results
                at = at.selectbox(key="questions_doc_select").set_value(cs_id).run()
                self.assertFalse(at.exception)
                cs_text = " ".join([m.value for m in at.markdown] + [c.value for c in at.caption])
                self.assertNotIn("Quiz Results — biology.pdf", cs_text)

                # Round-trip switching back to biology restores biology's submitted results intact
                at = at.selectbox(key="questions_doc_select").set_value(bio_id).run()
                self.assertFalse(at.exception)
                bio_restored_text = " ".join([m.value for m in at.markdown] + [c.value for c in at.caption])
                self.assertIn("Quiz Results — biology.pdf", bio_restored_text)

                # Ordinary rerun preserves submitted results stably
                at = at.run()
                self.assertFalse(at.exception)
                self.assertTrue(at.session_state["quiz_sessions"][bio_id]["submitted"])
                bio_rerun_text = " ".join([m.value for m in at.markdown] + [c.value for c in at.caption])
                self.assertIn("Quiz Results — biology.pdf", bio_rerun_text)

    def test_03_retry_creates_fresh_attempt_without_regenerating_questions(self):
        fake_model = _FakeUIEmbedder()
        provider = _MockQuestionUIProvider(configured=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            bio_id, _ = self._seed_two_pdfs(store_dir, fake_model)

            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch("src.generation.llm_provider.get_llm_provider", return_value=provider),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
            ):
                at = AppTest.from_file(APP_PATH).run()
                at.selectbox(key="questions_doc_select").set_value(bio_id)
                at = at.button(key="generate_questions_btn").click().run()
                at = at.radio(key=f"questions_view_mode_{bio_id}").set_value("Interactive Quiz").run()

                # Submit empty attempt 1
                submit_btn = [b for b in at.button if b.label == "Submit Answers"][0]
                at = submit_btn.click().run()
                self.assertTrue(at.session_state["quiz_sessions"][bio_id]["submitted"])
                self.assertEqual(at.session_state["quiz_sessions"][bio_id]["attempt_number"], 1)

                gen_calls_before_retry = len(provider.calls)

                # Click Retry Quiz -> increments attempt_number, clears answers/result, ZERO new generation calls!
                retry_btn = [b for b in at.button if b.label == "Retry Quiz"][0]
                at = retry_btn.click().run()
                self.assertFalse(at.exception)
                self.assertEqual(len(provider.calls), gen_calls_before_retry)
                quiz_session = at.session_state["quiz_sessions"][bio_id]
                self.assertEqual(quiz_session["attempt_number"], 2)
                self.assertFalse(quiz_session["submitted"])
                self.assertIsNone(quiz_session["result"])
                self.assertTrue(all(v is None or v == "" for v in quiz_session["answers"].values()))

    def test_04_grading_failure_preserves_answers_and_delete_clears_state(self):
        fake_model = _FakeUIEmbedder()
        good_provider = _MockQuestionUIProvider(configured=True, fail=False)
        bad_provider = _MockQuestionUIProvider(configured=True, fail=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            bio_id, _ = self._seed_two_pdfs(store_dir, fake_model)

            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch("src.generation.llm_provider.get_llm_provider", return_value=good_provider),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
            ):
                at = AppTest.from_file(APP_PATH).run()
                at.selectbox(key="questions_doc_select").set_value(bio_id)
                at.selectbox(key="questions_type_select").set_value("Short Answer")
                at.number_input(key="questions_count_input").set_value(1)
                at = at.button(key="generate_questions_btn").click().run()
                at = at.radio(key=f"questions_view_mode_{bio_id}").set_value("Interactive Quiz").run()

                qset = at.session_state["cached_questions"][bio_id]
                qid = qset["questions"][0]["question_id"]
                qset_id = qset["question_set_id"]
                w_key = f"quiz_ans_{bio_id}_{qset_id}_a1_{qid}"
                at.text_area(key=w_key).set_value("My typed short answer.")

            # Simulate LLM failure during grading
            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch("src.generation.llm_provider.get_llm_provider", return_value=bad_provider),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
            ):
                submit_btn = [b for b in at.button if b.label == "Submit Answers"][0]
                at = submit_btn.click().run()
                self.assertFalse(at.exception)
                self.assertTrue(any("Quiz evaluation failed" in err.value for err in at.error))
                # CRITICAL: Submitted answer MUST remain preserved in session state so user does not re-type!
                self.assertEqual(
                    at.session_state["quiz_sessions"][bio_id]["answers"][qid],
                    "My typed short answer.",
                )
                self.assertFalse(at.session_state["quiz_sessions"][bio_id]["submitted"])

            # Resubmitting with working provider succeeds without losing typed answers
            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch("src.generation.llm_provider.get_llm_provider", return_value=good_provider),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
            ):
                submit_btn = [b for b in at.button if b.label == "Submit Answers"][0]
                at = submit_btn.click().run()
                self.assertFalse(at.exception)
                self.assertTrue(at.session_state["quiz_sessions"][bio_id]["submitted"])
                self.assertIsNotNone(at.session_state["quiz_sessions"][bio_id]["result"])
                self.assertEqual(
                    at.session_state["quiz_sessions"][bio_id]["answers"][qid],
                    "My typed short answer.",
                )

            # Deleting document clears cached questions AND quiz session
            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch("src.generation.llm_provider.get_llm_provider", return_value=good_provider),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
            ):
                at = at.sidebar.button(key=f"delete_doc_{bio_id}").click().run()
                at = at.sidebar.button(key=f"confirm_delete_{bio_id}").click().run()
                self.assertFalse(at.exception)
                self.assertNotIn(bio_id, at.session_state.filtered_state.get("cached_questions", {}))
                self.assertNotIn(bio_id, at.session_state.filtered_state.get("quiz_sessions", {}))

    def test_05_reset_and_regenerate_clear_state_and_deterministic_mode(self):
        fake_model = _FakeUIEmbedder()
        provider = _MockQuestionUIProvider(configured=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            bio_id, _ = self._seed_two_pdfs(store_dir, fake_model)

            # Test 1: Configured evaluator mode == "deterministic" is respected by UI default
            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch.object(config, "QUIZ_SHORT_ANSWER_EVALUATOR", "deterministic"),
                patch("src.generation.llm_provider.get_llm_provider", return_value=provider),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
            ):
                at = AppTest.from_file(APP_PATH).run()
                at.selectbox(key="questions_doc_select").set_value(bio_id)
                at.selectbox(key="questions_type_select").set_value("Short Answer")
                at.number_input(key="questions_count_input").set_value(1)
                at = at.button(key="generate_questions_btn").click().run()
                at = at.radio(key=f"questions_view_mode_{bio_id}").set_value("Interactive Quiz").run()

                # Default selectbox value should be "Deterministic"
                self.assertEqual(at.selectbox(key=f"quiz_eval_mode_{bio_id}").value, "Deterministic")

                qset = at.session_state["cached_questions"][bio_id]
                qid = qset["questions"][0]["question_id"]
                qset_id = qset["question_set_id"]
                w_key = f"quiz_ans_{bio_id}_{qset_id}_a1_{qid}"
                at.text_area(key=w_key).set_value("Chloroplasts absorb sunlight.")

                calls_before_submit = len(provider.calls)
                submit_btn = [b for b in at.button if b.label == "Submit Answers"][0]
                at = submit_btn.click().run()
                self.assertFalse(at.exception)
                # Deterministic mode grading makes ZERO LLM calls!
                self.assertEqual(len(provider.calls), calls_before_submit)
                self.assertTrue(at.session_state["quiz_sessions"][bio_id]["submitted"])

                # Test 2: Generating new questions clears quiz session and widget keys
                at = at.button(key="generate_questions_btn").click().run()
                self.assertFalse(at.exception)
                self.assertNotIn(bio_id, at.session_state.filtered_state.get("quiz_sessions", {}))
                self.assertFalse(any(k.startswith(f"quiz_ans_{bio_id}_") for k in at.session_state.filtered_state.keys()))

                # Switch to interactive quiz again, fill an answer
                at = at.radio(key=f"questions_view_mode_{bio_id}").set_value("Interactive Quiz").run()
                new_qset = at.session_state["cached_questions"][bio_id]
                new_qid = new_qset["questions"][0]["question_id"]
                new_w_key = f"quiz_ans_{bio_id}_{new_qset['question_set_id']}_a1_{new_qid}"
                at.text_area(key=new_w_key).set_value("Some answer")
                at = at.run()
                self.assertIn(bio_id, at.session_state.filtered_state.get("quiz_sessions", {}))

                # Test 3: Resetting vector store clears all cached questions, quiz sessions, and widget keys
                (store_dir / "manifest.json").write_text("{corrupted", encoding="utf-8")
                at_reset = AppTest.from_file(APP_PATH).run()
                at_reset.session_state["cached_questions"] = {bio_id: {"questions": []}}
                at_reset.session_state["quiz_sessions"] = {bio_id: {"attempt_number": 1}}
                at_reset.session_state[f"quiz_ans_{bio_id}_q1"] = "Some answer"
                at_reset = at_reset.checkbox(key="confirm_reset_vector_store").check().run()
                at_reset = at_reset.button(key="reset_vector_store_btn").click().run()
                self.assertFalse(at_reset.exception)
                self.assertNotIn("cached_questions", at_reset.session_state)
                self.assertNotIn("quiz_sessions", at_reset.session_state)
                self.assertFalse(any(k.startswith("quiz_ans_") for k in at_reset.session_state.filtered_state.keys()))


class _MockFlashcardUIProvider(BaseLLMProvider):
    provider_name = "mock_ollama"

    def __init__(self, configured: bool = True):
        super().__init__(model_name="qwen2.5:7b")
        self._configured = configured
        self.calls: list[dict] = []

    def is_configured(self) -> bool:
        return self._configured

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        self.calls.append({"system": system_prompt, "user": user_prompt, "kwargs": kwargs})
        return json.dumps({
            "flashcards": [
                {
                    "front": "What does chlorophyll absorb?",
                    "back": "Sunlight for photosynthesis.",
                    "source_chunk_ids": [],
                },
                {
                    "front": "What do plant cells produce in photosynthesis?",
                    "back": "Glucose and oxygen.",
                    "source_chunk_ids": [],
                },
            ]
        })


class TestStage7FlashcardUI(unittest.TestCase):
    def _seed_two_pdfs(self, store_dir: Path, fake_model: _FakeUIEmbedder) -> tuple[str, str]:
        with patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model):
            svc = EmbeddingService(model_name="all-MiniLM-L6-v2")
            store = FAISSVectorStore(store_dir=store_dir, embedding_service=svc)
            r_bio = store.add_pdf(
                _make_pdf(
                    [
                        "Chapter 1: Photosynthesis\nPhotosynthesis in plant cells produces glucose.",
                        "",
                        "Chapter 2: Plant Pigments\nChlorophyll absorbs sunlight for photosynthesis.",
                    ],
                    name="biology.pdf",
                ),
                embedding_service=svc,
            )
            r_hist = store.add_pdf(
                _make_pdf(
                    [
                        "History Page 1: The industrial revolution transformed European manufacturing.",
                    ],
                    name="history.pdf",
                ),
                embedding_service=svc,
            )
            return r_bio["document_id"], r_hist["document_id"]

    def test_01_flashcard_generation_render_and_card_counter(self):
        fake_model = _FakeUIEmbedder()
        provider = _MockFlashcardUIProvider(configured=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            bio_id, _ = self._seed_two_pdfs(store_dir, fake_model)

            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch("src.generation.llm_provider.get_llm_provider", return_value=provider),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
            ):
                at = AppTest.from_file(APP_PATH).run()
                self.assertFalse(at.exception)

                at.selectbox(key="flashcard_doc_select").set_value(bio_id)
                at.number_input(key="flashcard_count_input").set_value(2)
                at = at.button(key="generate_flashcards_btn").click().run()

                self.assertFalse(at.exception)
                self.assertIn(bio_id, at.session_state["cached_flashcards"])
                cards = at.session_state["cached_flashcards"][bio_id]["cards"]
                self.assertEqual(len(cards), 2)

                all_text = " ".join(
                    [m.value for m in at.markdown]
                    + [c.value for c in at.caption]
                    + [s.value for s in at.subheader]
                )
                self.assertIn("Flashcard Deck — biology.pdf", all_text)
                self.assertIn("Card 1 of 2", all_text)
                self.assertIn("What does chlorophyll absorb?", all_text)

    def test_02_flashcard_flip_and_navigation(self):
        fake_model = _FakeUIEmbedder()
        provider = _MockFlashcardUIProvider(configured=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            bio_id, _ = self._seed_two_pdfs(store_dir, fake_model)

            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch("src.generation.llm_provider.get_llm_provider", return_value=provider),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
            ):
                at = AppTest.from_file(APP_PATH).run()
                at.selectbox(key="flashcard_doc_select").set_value(bio_id)
                at = at.button(key="generate_flashcards_btn").click().run()

                # Step 1: Flip Card to view Answer
                flip_btn = [b for b in at.button if "Flip Card" in b.label or "Show Front" in b.label][0]
                at = flip_btn.click().run()
                self.assertFalse(at.exception)
                self.assertTrue(at.session_state[f"fc_flipped_{bio_id}"])
                all_text = " ".join(
                    [m.value for m in at.markdown]
                    + [c.value for c in at.caption]
                    + [s.value for s in at.subheader]
                )
                self.assertIn("Sunlight for photosynthesis.", all_text)

                # Step 2: Next Card
                next_btn = [b for b in at.button if "Next Card" in b.label][0]
                at = next_btn.click().run()
                self.assertFalse(at.exception)
                self.assertEqual(at.session_state[f"fc_idx_{bio_id}"], 1)
                self.assertFalse(at.session_state[f"fc_flipped_{bio_id}"])
                all_text = " ".join(
                    [m.value for m in at.markdown]
                    + [c.value for c in at.caption]
                    + [s.value for s in at.subheader]
                )
                self.assertIn("Card 2 of 2", all_text)
                self.assertIn("What do plant cells produce in photosynthesis?", all_text)

                # Step 3: Previous Card
                prev_btn = [b for b in at.button if "Previous Card" in b.label][0]
                at = prev_btn.click().run()
                self.assertFalse(at.exception)
                self.assertEqual(at.session_state[f"fc_idx_{bio_id}"], 0)

    def test_03_flashcard_in_place_editing_and_persistence(self):
        fake_model = _FakeUIEmbedder()
        provider = _MockFlashcardUIProvider(configured=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            bio_id, _ = self._seed_two_pdfs(store_dir, fake_model)

            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch("src.generation.llm_provider.get_llm_provider", return_value=provider),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
            ):
                at = AppTest.from_file(APP_PATH).run()
                at.selectbox(key="flashcard_doc_select").set_value(bio_id)
                at = at.button(key="generate_flashcards_btn").click().run()

                # Open editor for card 0
                at = at.checkbox(key=f"fc_show_editor_{bio_id}_0").check().run()
                self.assertFalse(at.exception)

                # Revise front and back
                at.text_input(key=f"fc_input_front_{bio_id}_0").set_value("Revised Question on Chlorophyll")
                at.text_area(key=f"fc_input_back_{bio_id}_0").set_value("Revised Answer on Sunlight Absorption")
                save_btn = at.button(key=f"fc_save_btn_{bio_id}_0")
                at = save_btn.click().run()
                self.assertFalse(at.exception)

                # Verify updated in session_state and marked edited
                card_0 = at.session_state["cached_flashcards"][bio_id]["cards"][0]
                self.assertEqual(card_0["front"], "Revised Question on Chlorophyll")
                self.assertEqual(card_0["back"], "Revised Answer on Sunlight Absorption")
                self.assertTrue(card_0.get("edited"))

                # Rerun: verified preserved across reruns
                calls_before = len(provider.calls)
                at = at.run()
                self.assertFalse(at.exception)
                self.assertEqual(len(provider.calls), calls_before)
                self.assertEqual(at.session_state["cached_flashcards"][bio_id]["cards"][0]["front"], "Revised Question on Chlorophyll")

    def test_04_flashcard_export_controls(self):
        fake_model = _FakeUIEmbedder()
        provider = _MockFlashcardUIProvider(configured=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            bio_id, _ = self._seed_two_pdfs(store_dir, fake_model)

            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch("src.generation.llm_provider.get_llm_provider", return_value=provider),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
            ):
                at = AppTest.from_file(APP_PATH).run()
                at.selectbox(key="flashcard_doc_select").set_value(bio_id)
                at = at.button(key="generate_flashcards_btn").click().run()

                # Verify download buttons exist
                md_dl = [b for b in at.get("download_button") if "Markdown" in b.label]
                csv_dl = [b for b in at.get("download_button") if "Anki CSV" in b.label]
                self.assertTrue(len(md_dl) >= 1)
                self.assertTrue(len(csv_dl) >= 1)

    def test_05_flashcard_invalidation_on_delete_and_reset(self):
        fake_model = _FakeUIEmbedder()
        provider = _MockFlashcardUIProvider(configured=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            bio_id, _ = self._seed_two_pdfs(store_dir, fake_model)

            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch("src.generation.llm_provider.get_llm_provider", return_value=provider),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
            ):
                at = AppTest.from_file(APP_PATH).run()
                at.selectbox(key="flashcard_doc_select").set_value(bio_id)
                at = at.button(key="generate_flashcards_btn").click().run()
                self.assertIn(bio_id, at.session_state["cached_flashcards"])

                # Deleting PDF clears cached flashcards
                at = at.sidebar.button(key=f"delete_doc_{bio_id}").click().run()
                at = at.sidebar.button(key=f"confirm_delete_{bio_id}").click().run()
                self.assertFalse(at.exception)
                self.assertNotIn(bio_id, at.session_state.filtered_state.get("cached_flashcards", {}))


class _MockStudyNotesUIProvider(BaseLLMProvider):
    provider_name = "mock_ollama"

    def __init__(self, configured: bool = True, response_text: str | None = None):
        super().__init__(model_name="qwen2.5:7b")
        self._configured = configured
        self._response_text = response_text or (
            "# Study Notes: Plant Biology and Photosynthesis\n\n"
            "## Executive Overview\n\n"
            "Photosynthesis is the fundamental biological process through which autotrophic organisms convert light energy into chemical energy [Page 1].\n\n"
            "## Chapter 1: Light Reactions\n\n"
            "Chlorophyll molecules in thylakoid membranes absorb sunlight [Page 1].\n\n"
            "## Key Concepts and Definitions\n\n"
            "- **Photosynthesis**: Conversion of carbon dioxide and water into glucose and oxygen [Page 1].\n"
            "- **Chlorophyll**: Green photosynthetic pigment [Page 2].\n\n"
            "## High-Yield Summary and Review Takeaways\n\n"
            "1. Plants convert solar energy into glucose [Page 1].\n"
            "2. Thylakoid membranes harbor the light-dependent reaction machinery [Page 1]."
        )
        self.calls: list[dict] = []

    def is_configured(self) -> bool:
        return self._configured

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        self.calls.append({"system": system_prompt, "user": user_prompt, "kwargs": kwargs})
        return self._response_text


class TestStage8StudyNotesUI(unittest.TestCase):
    def _seed_two_pdfs(self, store_dir: Path, fake_model: _FakeUIEmbedder) -> tuple[str, str]:
        with patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model):
            svc = EmbeddingService(model_name="all-MiniLM-L6-v2")
            store = FAISSVectorStore(store_dir=store_dir, embedding_service=svc)
            r_bio = store.add_pdf(
                _make_pdf(
                    [
                        "Chapter 1: Photosynthesis\nPhotosynthesis in plant cells produces glucose.",
                        "",
                        "Chapter 2: Plant Pigments\nChlorophyll absorbs sunlight for photosynthesis.",
                    ],
                    name="biology.pdf",
                ),
                embedding_service=svc,
            )
            r_hist = store.add_pdf(
                _make_pdf(
                    [
                        "History Page 1: The industrial revolution transformed European manufacturing.",
                    ],
                    name="history.pdf",
                ),
                embedding_service=svc,
            )
            return r_bio["document_id"], r_hist["document_id"]

    def test_01_study_notes_tab_render_and_generation(self):
        fake_model = _FakeUIEmbedder()
        provider = _MockStudyNotesUIProvider(configured=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            bio_id, _ = self._seed_two_pdfs(store_dir, fake_model)

            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch("src.generation.llm_provider.get_llm_provider", return_value=provider),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
            ):
                at = AppTest.from_file(APP_PATH).run()
                self.assertFalse(at.exception)

                at.selectbox(key="notes_doc_select").set_value(bio_id)
                at = at.button(key="generate_notes_btn").click().run()

                self.assertFalse(at.exception)
                self.assertIn(bio_id, at.session_state["cached_notes"])
                notes_dict = at.session_state["cached_notes"][bio_id]
                self.assertEqual(notes_dict["title"], "Academic Study Notes: biology.pdf (Full Document)")
                self.assertIn("Photosynthesis", notes_dict["notes_text"])

                all_text = " ".join(
                    [m.value for m in at.markdown]
                    + [c.value for c in at.caption]
                    + [s.value for s in at.subheader]
                )
                self.assertIn("Academic Study Notes: biology.pdf (Full Document)", all_text)
                self.assertIn("Pages Covered", all_text)

    def test_02_study_notes_scope_page_range_and_topic(self):
        fake_model = _FakeUIEmbedder()
        provider = _MockStudyNotesUIProvider(configured=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            bio_id, _ = self._seed_two_pdfs(store_dir, fake_model)

            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch("src.generation.llm_provider.get_llm_provider", return_value=provider),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
            ):
                # 1. Page Range valid
                at = AppTest.from_file(APP_PATH).run()
                at.selectbox(key="notes_doc_select").set_value(bio_id)
                at = at.selectbox(key="notes_scope_select").set_value("Page Range").run()
                at.selectbox(key="notes_start_page").set_value(1)
                at.selectbox(key="notes_end_page").set_value(3)
                at = at.button(key="generate_notes_btn").click().run()
                self.assertFalse(at.exception)
                self.assertEqual(at.session_state["cached_notes"][bio_id]["scope"], "page_range")
                self.assertEqual(at.session_state["cached_notes"][bio_id]["page_range_label"], "Pages 1, 3")

                # 2. Page Range invalid (start > end)
                at.selectbox(key="notes_start_page").set_value(3)
                at.selectbox(key="notes_end_page").set_value(1)
                at = at.run()
                self.assertTrue(at.button(key="generate_notes_btn").disabled)
                error_texts = [e.value for e in at.error]
                self.assertTrue(any("cannot be greater than end page" in t for t in error_texts))

                # 3. Topic Scope
                at = at.selectbox(key="notes_scope_select").set_value("Topic").run()
                at.text_input(key="notes_topic_input").set_value("Chlorophyll absorption")
                at.number_input(key="notes_topic_top_k").set_value(3)
                at = at.button(key="generate_notes_btn").click().run()
                self.assertFalse(at.exception)
                self.assertEqual(at.session_state["cached_notes"][bio_id]["scope"], "topic")

    def test_03_study_notes_persistence_across_reruns(self):
        fake_model = _FakeUIEmbedder()
        provider = _MockStudyNotesUIProvider(configured=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            bio_id, _ = self._seed_two_pdfs(store_dir, fake_model)

            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch("src.generation.llm_provider.get_llm_provider", return_value=provider),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
            ):
                at = AppTest.from_file(APP_PATH).run()
                at.selectbox(key="notes_doc_select").set_value(bio_id)
                at = at.button(key="generate_notes_btn").click().run()
                self.assertIn(bio_id, at.session_state["cached_notes"])

                calls_before = len(provider.calls)
                at = at.run()
                self.assertFalse(at.exception)
                self.assertEqual(len(provider.calls), calls_before)
                self.assertIn(bio_id, at.session_state["cached_notes"])

    def test_04_study_notes_markdown_download(self):
        fake_model = _FakeUIEmbedder()
        provider = _MockStudyNotesUIProvider(configured=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            bio_id, _ = self._seed_two_pdfs(store_dir, fake_model)

            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch("src.generation.llm_provider.get_llm_provider", return_value=provider),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
            ):
                at = AppTest.from_file(APP_PATH).run()
                at.selectbox(key="notes_doc_select").set_value(bio_id)
                at = at.button(key="generate_notes_btn").click().run()

                download_buttons = [b for b in at.get("download_button") if "Download Notes (.md)" in b.label]
                self.assertTrue(len(download_buttons) >= 1)

    def test_05_study_notes_invalidation_on_delete_and_reset(self):
        fake_model = _FakeUIEmbedder()
        provider = _MockStudyNotesUIProvider(configured=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            bio_id, _ = self._seed_two_pdfs(store_dir, fake_model)

            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch("src.generation.llm_provider.get_llm_provider", return_value=provider),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
            ):
                at = AppTest.from_file(APP_PATH).run()
                at.selectbox(key="notes_doc_select").set_value(bio_id)
                at = at.button(key="generate_notes_btn").click().run()
                self.assertIn(bio_id, at.session_state["cached_notes"])

                # Deleting PDF clears cached notes
                at = at.sidebar.button(key=f"delete_doc_{bio_id}").click().run()
                at = at.sidebar.button(key=f"confirm_delete_{bio_id}").click().run()
                self.assertFalse(at.exception)
                self.assertNotIn(bio_id, at.session_state.filtered_state.get("cached_notes", {}))


class TestStage9PomodoroUI(unittest.TestCase):
    def test_01_pomodoro_tab_initial_render(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            stats_file = Path(tmpdir) / "study_stats.json"
            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch.object(config, "STUDY_STATS_FILE", stats_file),
            ):
                at = AppTest.from_file(APP_PATH).run()
                self.assertFalse(at.exception)

                all_text = " ".join(
                    [m.value for m in at.markdown]
                    + [c.value for c in at.caption]
                    + [s.value for s in at.subheader]
                )
                self.assertIn("Study Timer & Focus", all_text)
                self.assertIn("Focus Session", all_text)
                self.assertIn("25:00", all_text)
                self.assertTrue(len([b for b in at.button if b.key == "pomo_start_btn"]) >= 1)

    def test_02_pomodoro_start_pause_resume_reset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            stats_file = Path(tmpdir) / "study_stats.json"
            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch.object(config, "STUDY_STATS_FILE", stats_file),
            ):
                at = AppTest.from_file(APP_PATH).run()
                self.assertFalse(at.exception)

                # Start
                at = at.button(key="pomo_start_btn").click().run()
                self.assertFalse(at.exception)
                self.assertTrue(len([b for b in at.button if b.key == "pomo_pause_btn"]) >= 1)

                # Pause
                at = at.button(key="pomo_pause_btn").click().run()
                self.assertFalse(at.exception)
                self.assertTrue(len([b for b in at.button if b.key == "pomo_resume_btn"]) >= 1)

                # Resume
                at = at.button(key="pomo_resume_btn").click().run()
                self.assertFalse(at.exception)
                self.assertTrue(len([b for b in at.button if b.key == "pomo_pause_btn"]) >= 1)

                # Reset
                at = at.button(key="pomo_reset_btn").click().run()
                self.assertFalse(at.exception)
                self.assertTrue(len([b for b in at.button if b.key == "pomo_start_btn"]) >= 1)

    def test_03_pomodoro_skip_and_custom_settings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            stats_file = Path(tmpdir) / "study_stats.json"
            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch.object(config, "STUDY_STATS_FILE", stats_file),
            ):
                at = AppTest.from_file(APP_PATH).run()

                # Skip advances from focus to short break
                at = at.button(key="pomo_skip_btn").click().run()
                self.assertFalse(at.exception)
                all_text = " ".join(
                    [m.value for m in at.markdown]
                    + [c.value for c in at.caption]
                    + [s.value for s in at.subheader]
                )
                self.assertIn("Short Break", all_text)
                self.assertIn("05:00", all_text)

                # Customize durations
                at.number_input(key="pomo_input_short").set_value(8)
                at = at.button(key="pomo_apply_settings_btn").click().run()
                self.assertFalse(at.exception)
                all_text2 = " ".join([m.value for m in at.markdown] + [c.value for c in at.caption])
                self.assertIn("08:00", all_text2)

    def test_04_pomodoro_completion_and_stats_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            stats_file = Path(tmpdir) / "study_stats.json"
            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch.object(config, "STUDY_STATS_FILE", stats_file),
            ):
                at = AppTest.from_file(APP_PATH).run()
                timer = at.session_state["pomodoro_timer"]
                # Start timer and set deadline in the past
                timer.start(now_monotonic=100.0)
                timer.deadline_monotonic = 50.0  # Already expired

                # Run app to trigger completion
                at = at.run()
                self.assertFalse(at.exception)

                # Verify transition to short break and stats updated
                stats_mgr = at.session_state["study_stats_mgr"]
                stats = stats_mgr.to_dict()
                self.assertEqual(stats["completed_focus_sessions"], 1)
                self.assertEqual(stats["current_streak"], 1)
                self.assertEqual(len(stats["session_history"]), 1)

    def test_05_pomodoro_audio_chime_toggle_and_trigger(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            stats_file = Path(tmpdir) / "study_stats.json"
            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch.object(config, "STUDY_STATS_FILE", stats_file),
            ):
                at = AppTest.from_file(APP_PATH).run()
                self.assertFalse(at.exception)

                # Audio chime checkbox is present and defaults to True
                sound_toggles = [c for c in at.checkbox if c.key == "pomo_sound_enabled"]
                self.assertEqual(len(sound_toggles), 1)
                self.assertTrue(sound_toggles[0].value)

                # Uncheck sound toggle
                at = sound_toggles[0].uncheck().run()
                self.assertFalse(at.exception)
                self.assertFalse(at.session_state["pomo_sound_enabled"])


class TestStage9ChallengeUI(unittest.TestCase):
    def test_01_challenge_initial_render(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            stats_file = Path(tmpdir) / "study_stats.json"
            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch.object(config, "STUDY_STATS_FILE", stats_file),
            ):
                at = AppTest.from_file(APP_PATH).run()
                self.assertFalse(at.exception)

                all_text = " ".join(
                    [m.value for m in at.markdown]
                    + [c.value for c in at.caption]
                    + [s.value for s in at.subheader]
                )
                self.assertIn("Countdown Study Challenge", all_text)
                self.assertIn("25:00", all_text)

                # Widgets present
                self.assertTrue(len([t for t in at.text_input if t.key == "chal_title_input"]) >= 1)
                self.assertTrue(len([s for s in at.selectbox if s.key == "chal_preset_select"]) >= 1)
                self.assertTrue(len([b for b in at.button if b.key == "chal_start_btn"]) >= 1)
                self.assertTrue(len([b for b in at.button if b.key == "chal_reset_btn"]) >= 1)

    def test_02_challenge_start_pause_resume_reset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            stats_file = Path(tmpdir) / "study_stats.json"
            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch.object(config, "STUDY_STATS_FILE", stats_file),
            ):
                at = AppTest.from_file(APP_PATH).run()
                self.assertFalse(at.exception)

                # Start Challenge
                at.text_input(key="chal_title_input").set_value("Linear Algebra Practice")
                at = at.button(key="chal_start_btn").click().run()
                self.assertFalse(at.exception)
                self.assertTrue(len([b for b in at.button if b.key == "chal_pause_btn"]) >= 1)
                self.assertTrue(len([b for b in at.button if b.key == "chal_cancel_btn"]) >= 1)

                # Pause Challenge
                at = at.button(key="chal_pause_btn").click().run()
                self.assertFalse(at.exception)
                self.assertTrue(len([b for b in at.button if b.key == "chal_resume_btn"]) >= 1)

                # Resume Challenge
                at = at.button(key="chal_resume_btn").click().run()
                self.assertFalse(at.exception)
                self.assertTrue(len([b for b in at.button if b.key == "chal_pause_btn"]) >= 1)

                # Reset Challenge
                at = at.button(key="chal_reset_btn").click().run()
                self.assertFalse(at.exception)
                self.assertTrue(len([b for b in at.button if b.key == "chal_start_btn"]) >= 1)

    def test_03_challenge_cancel_flow_with_confirmation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            stats_file = Path(tmpdir) / "study_stats.json"
            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch.object(config, "STUDY_STATS_FILE", stats_file),
            ):
                at = AppTest.from_file(APP_PATH).run()
                at = at.button(key="chal_start_btn").click().run()

                # Trigger cancel prompt
                at = at.button(key="chal_cancel_btn").click().run()
                self.assertFalse(at.exception)
                self.assertTrue(len([b for b in at.button if b.key == "chal_confirm_cancel_btn"]) >= 1)
                self.assertTrue(len([b for b in at.button if b.key == "chal_cancel_dismiss_btn"]) >= 1)

                # Dismiss cancel prompt (keep going)
                at = at.button(key="chal_cancel_dismiss_btn").click().run()
                self.assertFalse(at.exception)
                if "chal_confirm_cancel" in at.session_state:
                    self.assertFalse(bool(at.session_state["chal_confirm_cancel"]))
                self.assertTrue(len([b for b in at.button if b.key == "chal_pause_btn"]) >= 1)

                # Confirm cancel
                at = at.button(key="chal_cancel_btn").click().run()
                at = at.button(key="chal_confirm_cancel_btn").click().run()
                self.assertFalse(at.exception)

                timer = at.session_state["challenge_timer"]
                self.assertEqual(timer.state, "cancelled")

                # Verify cancelled challenge is NOT recorded in stats
                stats_mgr = at.session_state["study_stats_mgr"]
                self.assertEqual(stats_mgr.to_dict()["completed_challenges"], 0)
                self.assertEqual(len(stats_mgr.to_dict()["challenge_history"]), 0)

    def test_04_challenge_completion_and_stats_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            stats_file = Path(tmpdir) / "study_stats.json"
            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch.object(config, "STUDY_STATS_FILE", stats_file),
            ):
                at = AppTest.from_file(APP_PATH).run()
                timer = at.session_state["challenge_timer"]
                timer.set_challenge("FAISS Retrieval Sprint", 15)
                timer.start(now_monotonic=100.0)
                timer.deadline_monotonic = 50.0  # Already expired

                # Run app to trigger completion
                at = at.run()
                self.assertFalse(at.exception)

                stats_mgr = at.session_state["study_stats_mgr"]
                stats = stats_mgr.to_dict()
                self.assertEqual(stats["completed_challenges"], 1)
                self.assertEqual(stats["total_challenge_minutes"], 15.0)
                self.assertEqual(len(stats["challenge_history"]), 1)
                self.assertEqual(stats["challenge_history"][0]["title"], "FAISS Retrieval Sprint")

                # Pomodoro stats remain 0 (independent tracking)
                self.assertEqual(stats["completed_focus_sessions"], 0)

                # History displayed in UI
                all_text = " ".join([m.value for m in at.markdown] + [c.value for c in at.caption])
                self.assertIn("FAISS Retrieval Sprint", all_text)

    def test_05_challenge_custom_duration_and_preset_selection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            stats_file = Path(tmpdir) / "study_stats.json"
            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch.object(config, "STUDY_STATS_FILE", stats_file),
            ):
                at = AppTest.from_file(APP_PATH).run()

                # Select custom duration preset
                at = at.selectbox(key="chal_preset_select").set_value("Custom").run()
                self.assertFalse(at.exception)
                self.assertTrue(len([n for n in at.number_input if n.key == "chal_custom_mins"]) >= 1)

                at.text_input(key="chal_title_input").set_value("Machine Learning Review")
                at.number_input(key="chal_custom_mins").set_value(45)

                at = at.button(key="chal_start_btn").click().run()
                self.assertFalse(at.exception)

                timer = at.session_state["challenge_timer"]
                self.assertEqual(timer.title, "Machine Learning Review")
                self.assertEqual(timer.duration_seconds, 45 * 60)
                self.assertEqual(timer.format_time(), "45:00")


class TestStage9TimedQuizUI(unittest.TestCase):
    def _seed_test_pdf(self, store_dir: Path, fake_model: _FakeUIEmbedder) -> str:
        with patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model):
            svc = EmbeddingService(model_name="all-MiniLM-L6-v2")
            store = FAISSVectorStore(store_dir=store_dir, embedding_service=svc)
            res = store.add_pdf(
                _make_pdf(
                    [
                        "Cell Biology Fundamentals\nMitochondria generate ATP through cellular respiration.",
                        "",
                        "Plant Biology\nChloroplasts convert solar energy into glucose.",
                    ],
                    name="biology.pdf",
                ),
                embedding_service=svc,
            )
            return res["document_id"]

    def test_01_untimed_mode_unaffected(self):
        fake_model = _FakeUIEmbedder()
        provider = _MockQuestionUIProvider(configured=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            stats_file = Path(tmpdir) / "study_stats.json"
            bio_id = self._seed_test_pdf(store_dir, fake_model)

            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch.object(config, "STUDY_STATS_FILE", stats_file),
                patch("src.generation.llm_provider.get_llm_provider", return_value=provider),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
            ):
                at = AppTest.from_file(APP_PATH).run()
                at.selectbox(key="questions_doc_select").set_value(bio_id)
                at.number_input(key="questions_count_input").set_value(2)
                at = at.button(key="generate_questions_btn").click().run()

                # Switch to Interactive Quiz
                at = at.radio(key=f"questions_view_mode_{bio_id}").set_value("Interactive Quiz").run()
                self.assertFalse(at.exception)

                # Verify Timed Mode toggle exists and is False by default
                timed_toggles = [c for c in at.checkbox if "quiz_timed_toggle_" in c.key]
                self.assertTrue(len(timed_toggles) >= 1)
                self.assertFalse(timed_toggles[0].value)

                # Submit untimed quiz
                at = at.button(key=f"submit_quiz_btn_{bio_id}").click().run()
                self.assertFalse(at.exception)
                self.assertTrue(at.session_state["quiz_sessions"][bio_id]["submitted"])

                # Untimed submission must NOT increment timed quiz stats
                stats_mgr = at.session_state["study_stats_mgr"]
                self.assertEqual(stats_mgr.to_dict()["completed_timed_quizzes"], 0)

    def test_02_timed_mode_controls_and_start(self):
        fake_model = _FakeUIEmbedder()
        provider = _MockQuestionUIProvider(configured=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            stats_file = Path(tmpdir) / "study_stats.json"
            bio_id = self._seed_test_pdf(store_dir, fake_model)

            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch.object(config, "STUDY_STATS_FILE", stats_file),
                patch("src.generation.llm_provider.get_llm_provider", return_value=provider),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
            ):
                at = AppTest.from_file(APP_PATH).run()
                at.selectbox(key="questions_doc_select").set_value(bio_id)
                at.number_input(key="questions_count_input").set_value(2)
                at = at.button(key="generate_questions_btn").click().run()
                at = at.radio(key=f"questions_view_mode_{bio_id}").set_value("Interactive Quiz").run()

                # Enable Timed Quiz Challenge
                toggle = [c for c in at.checkbox if "quiz_timed_toggle_" in c.key][0]
                at = toggle.check().run()
                self.assertFalse(at.exception)

                # Duration preset and Start button render
                presets = [s for s in at.selectbox if "quiz_timed_preset_" in s.key]
                self.assertTrue(len(presets) >= 1)
                start_btns = [b for b in at.button if b.key == f"start_timed_quiz_btn_{bio_id}"]
                self.assertTrue(len(start_btns) >= 1)

                # Start the timed quiz
                at = start_btns[0].click().run()
                self.assertFalse(at.exception)

                # Timer is now running
                session = at.session_state["quiz_sessions"][bio_id]
                timer = session["timer"]
                self.assertIsNotNone(timer)
                self.assertEqual(timer.state, "running")
                self.assertEqual(timer.duration_seconds, 10 * 60)

                # Countdown header is visible in UI
                all_text = " ".join([m.value for m in at.markdown] + [c.value for c in at.caption])
                self.assertIn("Time Remaining:", all_text)

    def test_03_timed_quiz_manual_submit_and_stats(self):
        fake_model = _FakeUIEmbedder()
        provider = _MockQuestionUIProvider(configured=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            stats_file = Path(tmpdir) / "study_stats.json"
            bio_id = self._seed_test_pdf(store_dir, fake_model)

            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch.object(config, "STUDY_STATS_FILE", stats_file),
                patch("src.generation.llm_provider.get_llm_provider", return_value=provider),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
            ):
                at = AppTest.from_file(APP_PATH).run()
                at.selectbox(key="questions_doc_select").set_value(bio_id)
                at.number_input(key="questions_count_input").set_value(2)
                at = at.button(key="generate_questions_btn").click().run()
                at = at.radio(key=f"questions_view_mode_{bio_id}").set_value("Interactive Quiz").run()

                # Enable Timed Quiz and start
                toggle = [c for c in at.checkbox if "quiz_timed_toggle_" in c.key][0]
                at = toggle.check().run()
                start_btn = [b for b in at.button if b.key == f"start_timed_quiz_btn_{bio_id}"][0]
                at = start_btn.click().run()

                # Manual submission
                submit_btn = [b for b in at.button if b.key == f"submit_quiz_btn_{bio_id}"][0]
                at = submit_btn.click().run()
                self.assertFalse(at.exception)

                session = at.session_state["quiz_sessions"][bio_id]
                self.assertTrue(session["submitted"])
                self.assertFalse(session["auto_submitted"])
                self.assertEqual(session["submission_state"], "submitted")

                # Verify stats recording
                stats_mgr = at.session_state["study_stats_mgr"]
                stats = stats_mgr.to_dict()
                self.assertEqual(stats["completed_timed_quizzes"], 1)
                self.assertEqual(len(stats["timed_quiz_history"]), 1)
                self.assertFalse(stats["timed_quiz_history"][0]["auto_submitted"])

                # Success banner shown
                all_text = " ".join([s.value for s in at.success] + [m.value for m in at.markdown])
                self.assertIn("Timed Quiz Completed!", all_text)

    def test_04_timed_quiz_auto_submit_on_expiry(self):
        fake_model = _FakeUIEmbedder()
        provider = _MockQuestionUIProvider(configured=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            stats_file = Path(tmpdir) / "study_stats.json"
            bio_id = self._seed_test_pdf(store_dir, fake_model)

            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch.object(config, "STUDY_STATS_FILE", stats_file),
                patch("src.generation.llm_provider.get_llm_provider", return_value=provider),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
            ):
                at = AppTest.from_file(APP_PATH).run()
                at.selectbox(key="questions_doc_select").set_value(bio_id)
                at.number_input(key="questions_count_input").set_value(2)
                at = at.button(key="generate_questions_btn").click().run()
                at = at.radio(key=f"questions_view_mode_{bio_id}").set_value("Interactive Quiz").run()

                # Enable Timed Quiz and start
                toggle = [c for c in at.checkbox if "quiz_timed_toggle_" in c.key][0]
                at = toggle.check().run()
                start_btn = [b for b in at.button if b.key == f"start_timed_quiz_btn_{bio_id}"][0]
                at = start_btn.click().run()

                # Set deadline in the past to trigger timeout
                timer = at.session_state["quiz_sessions"][bio_id]["timer"]
                timer.deadline_monotonic = 50.0  # Expired

                # Run app to trigger fragment update and auto-submission
                at = at.run()
                self.assertFalse(at.exception)

                session = at.session_state["quiz_sessions"][bio_id]
                self.assertTrue(session["submitted"])
                self.assertTrue(session["auto_submitted"])
                self.assertEqual(session["submission_state"], "submitted")

                # Verify stats recorded as auto_submitted
                stats_mgr = at.session_state["study_stats_mgr"]
                stats = stats_mgr.to_dict()
                self.assertEqual(stats["completed_timed_quizzes"], 1)
                self.assertTrue(stats["timed_quiz_history"][0]["auto_submitted"])

                # Warning banner shown for timeout
                all_text = " ".join([w.value for w in at.warning] + [m.value for m in at.markdown])
                self.assertIn("Time Expired!", all_text)

    def test_05_timed_quiz_retry_fresh_timer(self):
        fake_model = _FakeUIEmbedder()
        provider = _MockQuestionUIProvider(configured=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            stats_file = Path(tmpdir) / "study_stats.json"
            bio_id = self._seed_test_pdf(store_dir, fake_model)

            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch.object(config, "STUDY_STATS_FILE", stats_file),
                patch("src.generation.llm_provider.get_llm_provider", return_value=provider),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
            ):
                at = AppTest.from_file(APP_PATH).run()
                at.selectbox(key="questions_doc_select").set_value(bio_id)
                at.number_input(key="questions_count_input").set_value(2)
                at = at.button(key="generate_questions_btn").click().run()
                at = at.radio(key=f"questions_view_mode_{bio_id}").set_value("Interactive Quiz").run()

                # Start and submit timed quiz
                toggle = [c for c in at.checkbox if "quiz_timed_toggle_" in c.key][0]
                at = toggle.check().run()
                at = [b for b in at.button if b.key == f"start_timed_quiz_btn_{bio_id}"][0].click().run()
                at = [b for b in at.button if b.key == f"submit_quiz_btn_{bio_id}"][0].click().run()

                session1 = at.session_state["quiz_sessions"][bio_id]
                self.assertEqual(session1["attempt_number"], 1)

                # Retry Quiz
                retry_btn = [b for b in at.button if b.key == f"retry_quiz_btn_{bio_id}"][0]
                at = retry_btn.click().run()
                self.assertFalse(at.exception)

                session2 = at.session_state["quiz_sessions"][bio_id]
                self.assertEqual(session2["attempt_number"], 2)
                self.assertFalse(session2["submitted"])
                self.assertEqual(session2["submission_state"], "not_submitted")
                self.assertIsNone(session2["timer"])  # Fresh attempt has no previous timer!


if __name__ == "__main__":
    unittest.main()
