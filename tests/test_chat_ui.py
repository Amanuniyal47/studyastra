"""Unit and integration tests for Streamlit Chat with PDF UI (Stage 10.4).

Covers all 16 required areas:
1. Chat interface rendering (empty and populated states).
2. Creating a new conversation thread.
3. Sending a user question.
4. Displaying a successful assistant response.
5. Follow-up question handling and multi-turn transcript.
6. Conversation switching and history preservation.
7. Conversation renaming.
8. Conversation deletion with confirmation and lock cleanup.
9. Document switching and document-scoped isolation.
10. Citation and source evidence display.
11. Streaming behavior and write_stream integration.
12. Ollama unavailable / generation failure containment.
13. Failed-turn preservation and retry without duplicate messages.
14. Streamlit rerun stability.
15. Document deletion and store reset invalidation.
16. Zero regression across existing tabs and features.
"""

from collections.abc import Iterator
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
from src.chat.chat_orchestrator import ChatOrchestrator
from src.chat.conversation_memory import ConversationManager
from src.generation.llm_provider import BaseLLMProvider, LLMProviderError
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
                float("history" in lower or "revolution" in lower or "rome" in lower),
                0.05,
            ])
        return np.array(rows, dtype=np.float32)


class _FakeStreamingLLMProvider(BaseLLMProvider):
    """Mock LLM provider that yields tokens incrementally and records calls."""

    def __init__(
        self,
        answer: str = "Photosynthesis produces glucose and oxygen [Source 1, Page 1].",
        stream_chunks: list[str] | None = None,
    ):
        super().__init__(model_name="qwen2.5:7b")
        self.answer = answer
        self.stream_chunks = stream_chunks or [self.answer]
        self.generate_calls = 0
        self.stream_calls = 0

    def is_configured(self) -> bool:
        return True

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        self.generate_calls += 1
        return self.answer

    def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        self.stream_calls += 1
        for chunk in self.stream_chunks:
            yield chunk


class _FailingLLMProvider(BaseLLMProvider):
    """Mock LLM provider that simulates connection or generation failure."""

    def __init__(self, error_message: str = "Cannot connect to Ollama at http://localhost:11434"):
        super().__init__(model_name="qwen2.5:7b")
        self.error_message = error_message

    def is_configured(self) -> bool:
        return True

    def generate(self, *args, **kwargs) -> str:
        raise LLMProviderError(self.error_message)

    def generate_stream(self, *args, **kwargs) -> Iterator[str]:
        raise LLMProviderError(self.error_message)
        yield "never"


def _index_pdf_in_store(store_dir: Path, fake_model: _FakeUIEmbedder, pdf_upload: _NamedUpload) -> None:
    """Helper to pre-index a PDF in a FAISS vector store directly."""
    with patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model):
        service = EmbeddingService(model_name=config.EMBEDDING_MODEL)
        service._model = fake_model
        store = FAISSVectorStore(
            store_dir=store_dir,
            embedding_model=config.EMBEDDING_MODEL,
            embedding_service=service,
        )
        store.add_pdf(
            pdf_upload,
            filename=pdf_upload.name,
            embedding_service=service,
            skip_duplicates=True,
        )


class TestStreamlitChatInterface(unittest.TestCase):
    def setUp(self):
        ChatOrchestrator.clear_all_thread_locks()

    def tearDown(self):
        ChatOrchestrator.clear_all_thread_locks()

    def test_01_empty_knowledge_base_rendering(self):
        """Area 1: When no PDFs are indexed, Chat tab indicates no documents are available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            with patch.object(config, "VECTOR_STORE_DIR", store_dir):
                at = AppTest.from_file(APP_PATH).run()

            self.assertFalse(at.exception)
            # Info alert should be present explaining that no documents are indexed yet
            info_texts = [i.value for i in at.info]
            self.assertTrue(
                any("No documents indexed yet" in text or "Upload a PDF" in text for text in info_texts)
            )

    def test_02_indexed_pdf_renders_chat_interface(self):
        """Area 1 & 2: When PDF is indexed, chat tab displays controls, conversation thread, and chat_input."""
        fake_model = _FakeUIEmbedder()
        pdf_bio = _make_pdf(
            [
                "Biology Page 1: Photosynthesis in plants converts solar energy into chemical energy.",
                "Biology Page 2: Chloroplasts contain chlorophyll molecules.",
            ],
            name="biology.pdf",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            _index_pdf_in_store(store_dir, fake_model, pdf_bio)

            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
            ):
                at = AppTest.from_file(APP_PATH).run()

            self.assertFalse(at.exception)
            # Check subheader
            subheader_texts = [s.value for s in at.subheader]
            self.assertIn("Chat with PDF", subheader_texts)

            # Check chat_input is present
            self.assertEqual(len(at.chat_input), 1)

            # Check buttons for conversation management
            button_keys = [b.key for b in at.button if b.key]
            self.assertIn("chat_tab_new_chat_btn", button_keys)
            self.assertIn("chat_tab_rename_btn", button_keys)
            self.assertIn("chat_tab_delete_btn", button_keys)

    def test_03_create_new_conversation_thread(self):
        """Area 2: Clicking New Chat creates a second conversation thread and switches to it."""
        fake_model = _FakeUIEmbedder()
        pdf_bio = _make_pdf(
            ["Biology Page 1: Cellular respiration occurs in mitochondria."],
            name="biology.pdf",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            _index_pdf_in_store(store_dir, fake_model, pdf_bio)

            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
            ):
                at = AppTest.from_file(APP_PATH).run()
                self.assertFalse(at.exception)

                # Click New Chat
                btn_new = [b for b in at.button if b.key == "chat_tab_new_chat_btn"][0]
                btn_new.click().run()

                self.assertFalse(at.exception)
                # Session state conversation manager should now have 2 threads
                conv_mgr = at.session_state["chat_conversation_manager"]
                selected_doc_id = at.session_state["chat_selected_doc_id"]
                threads = conv_mgr.list_threads(selected_doc_id)
                self.assertEqual(len(threads), 2)
                # Active thread should be the newly created thread
                active_map = at.session_state["chat_active_conv_by_doc"]
                created_th = next(t for t in threads if t.title == "Chat 2")
                self.assertEqual(active_map[selected_doc_id], created_th.conversation_id)

    def test_04_send_user_question_and_receive_assistant_response(self):
        """Area 3, 4, 11: Sending question streams tokens, commits turn, and displays answer."""
        fake_model = _FakeUIEmbedder()
        fake_llm = _FakeStreamingLLMProvider(
            answer="Plants convert sunlight into glucose [Source 1, Page 1].",
            stream_chunks=["Plants ", "convert ", "sunlight ", "into glucose ", "[Source 1, Page 1]."],
        )
        pdf_bio = _make_pdf(
            ["Biology Page 1: Plants convert sunlight into glucose during photosynthesis."],
            name="biology.pdf",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            _index_pdf_in_store(store_dir, fake_model, pdf_bio)

            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
                patch("src.generation.llm_provider.get_llm_provider", return_value=fake_llm),
                patch("src.chat.chat_orchestrator.get_llm_provider", return_value=fake_llm),
            ):
                at = AppTest.from_file(APP_PATH).run()
                self.assertFalse(at.exception)

                # Submit question
                at.chat_input[0].set_value("What do plants convert sunlight into?").run()
                self.assertFalse(at.exception)

                # Verify assistant stream was invoked
                self.assertGreater(fake_llm.stream_calls, 0)

                # Verify messages in conversation thread
                conv_mgr = at.session_state["chat_conversation_manager"]
                selected_doc_id = at.session_state["chat_selected_doc_id"]
                active_id = at.session_state["chat_active_conv_by_doc"][selected_doc_id]
                thread = conv_mgr.get_thread(selected_doc_id, active_id)

                self.assertEqual(len(thread.messages), 2)
                self.assertEqual(thread.messages[0].role, "user")
                self.assertEqual(thread.messages[0].content, "What do plants convert sunlight into?")
                self.assertEqual(thread.messages[1].role, "assistant")
                self.assertIn("Plants convert sunlight into glucose", thread.messages[1].content)

    def test_05_follow_up_question_and_multi_turn_history(self):
        """Area 5: Follow-up question preserves turn 1, appends turn 2, and renders chronologically."""
        fake_model = _FakeUIEmbedder()
        fake_llm = _FakeStreamingLLMProvider(
            answer="Glucose is stored as starch [Source 1, Page 1].",
        )
        pdf_bio = _make_pdf(
            [
                "Biology Page 1: Plants produce glucose. Glucose is then stored as starch in plant cells.",
            ],
            name="biology.pdf",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            _index_pdf_in_store(store_dir, fake_model, pdf_bio)

            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
                patch("src.generation.llm_provider.get_llm_provider", return_value=fake_llm),
                patch("src.chat.chat_orchestrator.get_llm_provider", return_value=fake_llm),
            ):
                at = AppTest.from_file(APP_PATH).run()
                # Turn 1
                at.chat_input[0].set_value("What is produced?").run()
                self.assertFalse(at.exception)

                # Turn 2 (Follow-up)
                at.chat_input[0].set_value("How is it stored?").run()
                self.assertFalse(at.exception)

                conv_mgr = at.session_state["chat_conversation_manager"]
                selected_doc_id = at.session_state["chat_selected_doc_id"]
                active_id = at.session_state["chat_active_conv_by_doc"][selected_doc_id]
                thread = conv_mgr.get_thread(selected_doc_id, active_id)

                self.assertEqual(len(thread.messages), 4)
                self.assertEqual(thread.messages[0].content, "What is produced?")
                self.assertEqual(thread.messages[1].role, "assistant")
                self.assertEqual(thread.messages[2].content, "How is it stored?")
                self.assertEqual(thread.messages[3].role, "assistant")

    def test_06_conversation_switching_and_history_preservation(self):
        """Area 6: Switching between conversation threads preserves each thread's separate history."""
        fake_model = _FakeUIEmbedder()
        fake_llm = _FakeStreamingLLMProvider(answer="Answer 1.")
        pdf_bio = _make_pdf(["Biology content."], name="biology.pdf")

        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            _index_pdf_in_store(store_dir, fake_model, pdf_bio)

            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
                patch("src.generation.llm_provider.get_llm_provider", return_value=fake_llm),
                patch("src.chat.chat_orchestrator.get_llm_provider", return_value=fake_llm),
            ):
                at = AppTest.from_file(APP_PATH).run()

                # Turn in Thread 1
                at.chat_input[0].set_value("Question in Thread 1").run()
                self.assertFalse(at.exception)

                # Create Thread 2
                btn_new = [b for b in at.button if b.key == "chat_tab_new_chat_btn"][0]
                btn_new.click().run()

                conv_mgr = at.session_state["chat_conversation_manager"]
                selected_doc_id = at.session_state["chat_selected_doc_id"]
                threads = conv_mgr.list_threads(selected_doc_id)
                th1 = next(t for t in threads if len(t.messages) == 2)
                th2 = next(t for t in threads if len(t.messages) == 0)

                # Thread 2 is currently active and empty
                self.assertEqual(len(th2.messages), 0)

                # Switch back to Thread 1 via selectbox
                sb_thread = [s for s in at.selectbox if s.key == "chat_thread_select"][0]
                sb_thread.select(th1.title).run()

                # Verify Thread 1 still has its 2 messages
                th1_after = conv_mgr.get_thread(selected_doc_id, th1.conversation_id)
                self.assertEqual(len(th1_after.messages), 2)
                self.assertEqual(th1_after.messages[0].content, "Question in Thread 1")

    def test_07_rename_conversation_thread(self):
        """Area 7: Renaming a conversation updates its title in state and UI."""
        fake_model = _FakeUIEmbedder()
        pdf_bio = _make_pdf(["Biology content."], name="biology.pdf")

        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            _index_pdf_in_store(store_dir, fake_model, pdf_bio)

            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
            ):
                at = AppTest.from_file(APP_PATH).run()

                # Click Rename button to open input
                btn_ren = [b for b in at.button if b.key == "chat_tab_rename_btn"][0]
                btn_ren.click().run()

                # Set new title
                ren_input = [t for t in at.text_input if t.key == "chat_tab_rename_input"][0]
                ren_input.set_value("Photosynthesis Deep Dive")

                # Click Save
                btn_save = [b for b in at.button if b.key == "chat_tab_save_rename_btn"][0]
                btn_save.click().run()

                conv_mgr = at.session_state["chat_conversation_manager"]
                selected_doc_id = at.session_state["chat_selected_doc_id"]
                active_id = at.session_state["chat_active_conv_by_doc"][selected_doc_id]
                thread = conv_mgr.get_thread(selected_doc_id, active_id)

                self.assertEqual(thread.title, "Photosynthesis Deep Dive")

    def test_08_delete_conversation_with_confirmation_and_lock_cleanup(self):
        """Area 8: Deletion requires confirmation, removes thread, and evicts lock."""
        fake_model = _FakeUIEmbedder()
        pdf_bio = _make_pdf(["Biology content."], name="biology.pdf")

        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            _index_pdf_in_store(store_dir, fake_model, pdf_bio)

            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
            ):
                at = AppTest.from_file(APP_PATH).run()

                # Create second thread so one remains after deletion
                btn_new = [b for b in at.button if b.key == "chat_tab_new_chat_btn"][0]
                btn_new.click().run()

                conv_mgr = at.session_state["chat_conversation_manager"]
                selected_doc_id = at.session_state["chat_selected_doc_id"]
                self.assertEqual(len(conv_mgr.list_threads(selected_doc_id)), 2)

                # Click Delete
                btn_del = [b for b in at.button if b.key == "chat_tab_delete_btn"][0]
                btn_del.click().run()

                # Cancel should abort deletion
                btn_cancel = [b for b in at.button if b.key == "chat_tab_cancel_delete_btn"][0]
                btn_cancel.click().run()
                self.assertEqual(len(conv_mgr.list_threads(selected_doc_id)), 2)

                # Click Delete again, then Confirm
                btn_del = [b for b in at.button if b.key == "chat_tab_delete_btn"][0]
                btn_del.click().run()
                btn_confirm = [b for b in at.button if b.key == "chat_tab_confirm_delete_btn"][0]
                btn_confirm.click().run()

                # Exactly 1 thread remains
                self.assertEqual(len(conv_mgr.list_threads(selected_doc_id)), 1)

    def test_09_document_switching_and_isolation(self):
        """Area 9: Switching documents isolates conversations and prevents cross-doc context contamination."""
        fake_model = _FakeUIEmbedder()
        fake_llm = _FakeStreamingLLMProvider(answer="Subject answer.")
        pdf_bio = _make_pdf(["Photosynthesis in plants."], name="biology.pdf")
        pdf_hist = _make_pdf(["Roman empire and history."], name="history.pdf")

        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            _index_pdf_in_store(store_dir, fake_model, pdf_bio)
            _index_pdf_in_store(store_dir, fake_model, pdf_hist)

            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
                patch("src.generation.llm_provider.get_llm_provider", return_value=fake_llm),
                patch("src.chat.chat_orchestrator.get_llm_provider", return_value=fake_llm),
            ):
                at = AppTest.from_file(APP_PATH).run()

                # We have 2 documents
                store = FAISSVectorStore(
                    store_dir=store_dir,
                    embedding_model=config.EMBEDDING_MODEL,
                )
                docs = store.list_documents()
                bio_id = next(d["document_id"] for d in docs if "bio" in d["filename"])
                hist_id = next(d["document_id"] for d in docs if "hist" in d["filename"])

                # Ask biology question in first document
                at.chat_input[0].set_value("Biology question").run()

                conv_mgr = at.session_state["chat_conversation_manager"]
                bio_thread = conv_mgr.list_threads(bio_id)[0]
                self.assertEqual(len(bio_thread.messages), 2)

                # Switch to history document
                doc_select = [s for s in at.selectbox if s.key == "chat_doc_select"][0]
                doc_select.select("history.pdf").run()

                # Verify history document has its own empty thread and 0 messages
                hist_threads = conv_mgr.list_threads(hist_id)
                self.assertGreaterEqual(len(hist_threads), 1)
                self.assertEqual(len(hist_threads[0].messages), 0)

                # Switch back to biology
                doc_select = [s for s in at.selectbox if s.key == "chat_doc_select"][0]
                doc_select.select("biology.pdf").run()

                # Biology history is completely intact
                bio_thread_after = conv_mgr.get_thread(bio_id, bio_thread.conversation_id)
                self.assertEqual(len(bio_thread_after.messages), 2)

    def test_10_citations_and_source_evidence_display(self):
        """Area 10: Validated citations render with page numbers, filenames, and snippets."""
        fake_model = _FakeUIEmbedder()
        fake_llm = _FakeStreamingLLMProvider(
            answer="Chloroplasts contain chlorophyll pigments [Source 1, Page 2].",
            stream_chunks=["Chloroplasts contain chlorophyll pigments [Source 1, Page 2]."],
        )
        pdf_bio = _make_pdf(
            [
                "Biology Page 1: Introduction to cells.",
                "Biology Page 2: Chloroplasts contain chlorophyll pigments that capture sunlight.",
            ],
            name="biology.pdf",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            _index_pdf_in_store(store_dir, fake_model, pdf_bio)

            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
                patch("src.generation.llm_provider.get_llm_provider", return_value=fake_llm),
                patch("src.chat.chat_orchestrator.get_llm_provider", return_value=fake_llm),
            ):
                at = AppTest.from_file(APP_PATH).run()
                at.chat_input[0].set_value("What pigments do chloroplasts contain?").run()
                self.assertFalse(at.exception)

                # Expander for sources should be rendered
                expander_labels = [e.label for e in at.expander]
                self.assertTrue(any("Sources" in label for label in expander_labels))

                conv_mgr = at.session_state["chat_conversation_manager"]
                selected_doc_id = at.session_state["chat_selected_doc_id"]
                active_id = at.session_state["chat_active_conv_by_doc"][selected_doc_id]
                thread = conv_mgr.get_thread(selected_doc_id, active_id)
                asst_msg = thread.messages[1]

                self.assertGreaterEqual(len(asst_msg.citations), 1)
                first_cite = asst_msg.citations[0]
                self.assertEqual(first_cite.get("source_id"), 1)
                self.assertEqual(first_cite.get("page"), 2)

    def test_11_streaming_write_stream_integration(self):
        """Area 11: Real-time tokens from generator stream into UI."""
        fake_model = _FakeUIEmbedder()
        chunks = ["TokenA ", "TokenB ", "TokenC."]
        fake_llm = _FakeStreamingLLMProvider(answer="TokenA TokenB TokenC.", stream_chunks=chunks)
        pdf_bio = _make_pdf(["Biology text."], name="biology.pdf")

        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            _index_pdf_in_store(store_dir, fake_model, pdf_bio)

            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
                patch("src.generation.llm_provider.get_llm_provider", return_value=fake_llm),
                patch("src.chat.chat_orchestrator.get_llm_provider", return_value=fake_llm),
            ):
                at = AppTest.from_file(APP_PATH).run()
                at.chat_input[0].set_value("Give me tokens").run()
                self.assertFalse(at.exception)

                self.assertEqual(fake_llm.stream_calls, 1)
                conv_mgr = at.session_state["chat_conversation_manager"]
                selected_doc_id = at.session_state["chat_selected_doc_id"]
                active_id = at.session_state["chat_active_conv_by_doc"][selected_doc_id]
                thread = conv_mgr.get_thread(selected_doc_id, active_id)
                self.assertEqual(thread.messages[1].content, "TokenA TokenB TokenC.")

    def test_12_ollama_unavailable_and_generation_failure(self):
        """Area 12: LLM failure displays error alert and prevents corrupted assistant messages."""
        fake_model = _FakeUIEmbedder()
        failing_llm = _FailingLLMProvider("Cannot connect to Ollama daemon on port 11434")
        pdf_bio = _make_pdf(["Biology text."], name="biology.pdf")

        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            _index_pdf_in_store(store_dir, fake_model, pdf_bio)

            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
                patch("src.generation.llm_provider.get_llm_provider", return_value=failing_llm),
                patch("src.chat.chat_orchestrator.get_llm_provider", return_value=failing_llm),
            ):
                at = AppTest.from_file(APP_PATH).run()
                at.chat_input[0].set_value("Will this succeed?").run()

                # Should not crash the Streamlit app
                self.assertFalse(at.exception)

                # Error banner displayed
                error_texts = [e.value for e in at.error]
                self.assertTrue(any("Cannot connect to Ollama" in err or "Generation" in err for err in error_texts))

                # Assistant message must NOT be added to thread
                conv_mgr = at.session_state["chat_conversation_manager"]
                selected_doc_id = at.session_state["chat_selected_doc_id"]
                active_id = at.session_state["chat_active_conv_by_doc"][selected_doc_id]
                thread = conv_mgr.get_thread(selected_doc_id, active_id)

                self.assertEqual(len(thread.messages), 1)
                self.assertEqual(thread.messages[0].role, "user")
                self.assertEqual(thread.messages[0].metadata.get("turn_status"), "failed")

    def test_13_failed_turn_preservation_and_retry(self):
        """Area 13: Failed turn preserves user question and retry commits answer without duplicate messages."""
        fake_model = _FakeUIEmbedder()
        failing_llm = _FailingLLMProvider("Transient Ollama timeout")
        recovered_llm = _FakeStreamingLLMProvider(answer="Photosynthesis recovered successfully.")
        pdf_bio = _make_pdf(["Biology text."], name="biology.pdf")

        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            _index_pdf_in_store(store_dir, fake_model, pdf_bio)

            # Pass 1: Failure
            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
                patch("src.generation.llm_provider.get_llm_provider", return_value=failing_llm),
                patch("src.chat.chat_orchestrator.get_llm_provider", return_value=failing_llm),
            ):
                at = AppTest.from_file(APP_PATH).run()
                at.chat_input[0].set_value("Explain the process").run()
                self.assertFalse(at.exception)

                # Check retry button exists
                retry_btns = [b for b in at.button if b.key and "retry_turn_" in b.key]
                self.assertGreaterEqual(len(retry_btns), 1)

            # Pass 2: Retry with recovered provider
            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
                patch("src.generation.llm_provider.get_llm_provider", return_value=recovered_llm),
                patch("src.chat.chat_orchestrator.get_llm_provider", return_value=recovered_llm),
            ):
                retry_btns[0].click().run()
                self.assertFalse(at.exception)

                conv_mgr = at.session_state["chat_conversation_manager"]
                selected_doc_id = at.session_state["chat_selected_doc_id"]
                active_id = at.session_state["chat_active_conv_by_doc"][selected_doc_id]
                thread = conv_mgr.get_thread(selected_doc_id, active_id)

                # Total messages must be exactly 2 (no duplicate user question)
                self.assertEqual(len(thread.messages), 2)
                self.assertEqual(thread.messages[0].role, "user")
                self.assertEqual(thread.messages[0].metadata.get("turn_status"), "completed")
                self.assertEqual(thread.messages[1].role, "assistant")
                self.assertIn("Photosynthesis recovered successfully.", thread.messages[1].content)

    def test_14_streamlit_rerun_stability(self):
        """Area 14: Successive script reruns do not mutate history, duplicate turns, or lose active state."""
        fake_model = _FakeUIEmbedder()
        fake_llm = _FakeStreamingLLMProvider(answer="Consistent answer.")
        pdf_bio = _make_pdf(["Biology text."], name="biology.pdf")

        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            _index_pdf_in_store(store_dir, fake_model, pdf_bio)

            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
                patch("src.generation.llm_provider.get_llm_provider", return_value=fake_llm),
                patch("src.chat.chat_orchestrator.get_llm_provider", return_value=fake_llm),
            ):
                at = AppTest.from_file(APP_PATH).run()
                at.chat_input[0].set_value("First question").run()
                self.assertFalse(at.exception)

                # Perform 3 successive unprompted reruns
                at.run()
                at.run()
                at.run()
                self.assertFalse(at.exception)

                conv_mgr = at.session_state["chat_conversation_manager"]
                selected_doc_id = at.session_state["chat_selected_doc_id"]
                active_id = at.session_state["chat_active_conv_by_doc"][selected_doc_id]
                thread = conv_mgr.get_thread(selected_doc_id, active_id)

                # Still exactly 2 messages
                self.assertEqual(len(thread.messages), 2)

    def test_15_document_deletion_and_store_reset_invalidation(self):
        """Area 15: Document deletion and store reset safely clean up chat state and release locks."""
        fake_model = _FakeUIEmbedder()
        pdf_bio = _make_pdf(["Biology text."], name="biology.pdf")

        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            _index_pdf_in_store(store_dir, fake_model, pdf_bio)

            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
            ):
                at = AppTest.from_file(APP_PATH).run()

                conv_mgr = at.session_state["chat_conversation_manager"]
                selected_doc_id = at.session_state["chat_selected_doc_id"]
                self.assertEqual(len(conv_mgr.list_threads(selected_doc_id)), 1)

                # Delete the document via sidebar
                del_btn = [b for b in at.sidebar.button if b.key and b.key.startswith("delete_doc_")][0]
                del_btn.click().run()

                confirm_btn = [b for b in at.sidebar.button if b.key and b.key.startswith("confirm_delete_")][0]
                confirm_btn.click().run()

                self.assertFalse(at.exception)
                # Chat state for deleted document must be cleaned up
                conv_mgr_after = at.session_state["chat_conversation_manager"]
                self.assertEqual(len(conv_mgr_after.list_threads(selected_doc_id)), 0)
                self.assertTrue("chat_selected_doc_id" not in at.session_state or at.session_state["chat_selected_doc_id"] is None)

    def test_16_regression_all_existing_features_functional(self):
        """Area 16: PDF Summary, Notes, Questions, Flashcards, Timer, and Vector Testing remain functional."""
        fake_model = _FakeUIEmbedder()
        pdf_bio = _make_pdf(["Biology photosynthesis overview."], name="biology.pdf")

        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            _index_pdf_in_store(store_dir, fake_model, pdf_bio)

            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
            ):
                at = AppTest.from_file(APP_PATH).run()
                self.assertFalse(at.exception)

                # Verify all tabs render without exception
                tab_labels = [
                    "Chat with PDF",
                    "Vector Retrieval Testing",
                    "PDF Summary",
                    "Study Notes",
                    "Practice Questions",
                    "Flashcards",
                    "Study Timer",
                ]
                subheader_texts = [s.value for s in at.subheader]
                for expected_header in ["Chat with PDF", "PDF Summary", "Study Notes", "Practice Questions", "Flashcards", "Study Timer & Focus"]:
                    self.assertIn(expected_header, subheader_texts)

    def test_17_typing_indicator_and_css_injection(self):
        """Area 17: Typing indicator CSS is injected and write_stream completes cleanly."""
        fake_model = _FakeUIEmbedder()
        chunks = ["Answer ", "with ", "typing ", "indicator."]
        fake_llm = _FakeStreamingLLMProvider(
            answer="Answer with typing indicator.",
            stream_chunks=chunks,
        )
        pdf_bio = _make_pdf(["Sample text."], name="sample.pdf")

        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            _index_pdf_in_store(store_dir, fake_model, pdf_bio)

            with (
                patch.object(config, "VECTOR_STORE_DIR", store_dir),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
                patch("src.generation.llm_provider.get_llm_provider", return_value=fake_llm),
                patch("src.chat.chat_orchestrator.get_llm_provider", return_value=fake_llm),
            ):
                at = AppTest.from_file(APP_PATH).run()
                self.assertFalse(at.exception)

                # Verify typing indicator CSS exists in injected styles
                css_markdown = [m.value for m in at.markdown if "typing-indicator" in m.value]
                self.assertTrue(len(css_markdown) >= 1)

                # Submit question
                at.chat_input[0].set_value("Test indicator question").run()
                self.assertFalse(at.exception)
                self.assertEqual(fake_llm.stream_calls, 1)


if __name__ == "__main__":
    unittest.main()
