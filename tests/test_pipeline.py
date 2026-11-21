from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pymupdf

import src.config as config
from src.generation.answer_generator import generate_answer
from src.generation.llm_provider import LLMConfigurationError
from src.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from src.ingestion import chunker
from src.ingestion.pdf_loader import extract_pdf_text
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.retriever import retrieve_chunks
from src.retrieval.vector_store import FAISSVectorStore, index_pdf
from src.utils.file_utils import file_hash


class FakeUploadedFile:
    def __init__(self, data: bytes):
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def make_pdf_bytes(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class FakeEmbeddingModel:
    """Deterministic lightweight embedder for fast offline unit tests."""

    def get_embedding_dimension(self) -> int:
        return 4

    def encode(
        self,
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    ):
        vectors = []
        for text in texts:
            lower = text.lower()
            vectors.append([
                float("faiss" in lower or "vector" in lower),
                float("streamlit" in lower or "ui" in lower),
                float("groq" in lower or "llm" in lower),
                max(0.1, float(len(text)) / 1000.0),
            ])
        return np.array(vectors, dtype="float32")


class TestFileUtils(unittest.TestCase):
    def test_file_hash_deterministic(self):
        f1 = FakeUploadedFile(b"hello pdf")
        f2 = FakeUploadedFile(b"hello pdf")
        f3 = FakeUploadedFile(b"different pdf")
        self.assertEqual(file_hash(f1), file_hash(f2))
        self.assertNotEqual(file_hash(f1), file_hash(f3))
        self.assertEqual(len(file_hash(f1)), 64)


class TestPdfLoader(unittest.TestCase):
    def test_extract_pdf_text_multipage_and_skips_blank(self):
        pdf = FakeUploadedFile(make_pdf_bytes(["First page content", "", "Third page content"]))
        pages = extract_pdf_text(pdf)
        self.assertEqual(len(pages), 2)
        self.assertEqual(pages[0][0], 1)
        self.assertIn("First page content", pages[0][1])
        self.assertEqual(pages[1][0], 3)
        self.assertIn("Third page content", pages[1][1])

    def test_extract_pdf_text_empty_bytes(self):
        pdf = FakeUploadedFile(b"")
        self.assertEqual(extract_pdf_text(pdf), [])


class TestChunker(unittest.TestCase):
    def test_short_text_single_chunk(self):
        chunks = chunker.chunk_text([(1, "Short text.")], document_id="doc1", filename="notes.pdf")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["text"], "Short text.")
        self.assertEqual(chunks[0]["page"], 1)
        self.assertEqual(chunks[0]["page_range"], "1")
        self.assertEqual(chunks[0]["document_id"], "doc1")
        self.assertEqual(chunks[0]["filename"], "notes.pdf")
        self.assertEqual(chunks[0]["chunk_index"], 1)
        self.assertTrue(chunks[0]["chunk_id"])

    def test_text_near_chunk_size_no_duplicate_tail(self):
        text = " ".join(f"word{i}" for i in range(200))
        chunks = chunker.chunk_text([(1, text)])
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["text"], text)

    def test_overlapping_chunks(self):
        sentences = " ".join(f"Sentence number {i} explains a study concept clearly." for i in range(60))
        chunks = chunker.chunk_text([(2, sentences)])
        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(c["page"] == 2 for c in chunks))
        self.assertTrue(all(c["token_count"] <= 220 for c in chunks))

    def test_invalid_overlap_raises(self):
        with self.assertRaises(ValueError):
            chunker.chunk_text([(1, "Some text")], chunk_size=100, chunk_overlap=100)


class TestRetrievalPipeline(unittest.TestCase):
    def test_index_and_retrieve(self):
        fake_state = {}
        pdf = FakeUploadedFile(
            make_pdf_bytes([
                "FAISS is used for vector similarity search.",
                "Streamlit provides the web UI.",
                "Groq runs the LLM inference.",
            ])
        )
        fake_model = FakeEmbeddingModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch("src.retrieval.vector_store.st.session_state", fake_state),
                patch("src.retrieval.retriever.st.session_state", fake_state),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
            ):
                service = EmbeddingService(model_name="all-MiniLM-L6-v2")
                store = FAISSVectorStore(store_dir=tmpdir, embedding_service=service)

                count = index_pdf(pdf, store=store, embedding_service=service)
                self.assertEqual(count, 3)

                results = retrieve_chunks(
                    "How does vector search with FAISS work?",
                    store=store,
                    embedding_service=service,
                )
                self.assertGreater(len(results), 0)
                self.assertIn("FAISS", results[0]["text"])
                self.assertEqual(results[0]["metadata"]["page"], 1)
                self.assertIn("score", results[0])
                self.assertIn("distance", results[0])
                self.assertIn("filename", results[0]["metadata"])

    def test_index_empty_pdf_raises(self):
        fake_state = {}
        pdf = FakeUploadedFile(make_pdf_bytes([""]))
        fake_model = FakeEmbeddingModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch("src.retrieval.vector_store.st.session_state", fake_state),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
            ):
                service = EmbeddingService(model_name="all-MiniLM-L6-v2")
                store = FAISSVectorStore(store_dir=tmpdir, embedding_service=service)
                with self.assertRaises(ValueError):
                    index_pdf(pdf, store=store, embedding_service=service)
                self.assertEqual(store.total_documents, 0)
                self.assertEqual(store.total_vectors, 0)

    def test_retrieve_without_index_raises(self):
        fake_model = FakeEmbeddingModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch("src.retrieval.retriever.st.session_state", {}),
                patch("src.retrieval.embeddings.load_embedding_model", return_value=fake_model),
            ):
                service = EmbeddingService(model_name="all-MiniLM-L6-v2")
                store = FAISSVectorStore(store_dir=tmpdir, embedding_service=service)
                with self.assertRaises(ValueError):
                    retrieve_chunks("Any question?", store=store, embedding_service=service)


class TestGeneration(unittest.TestCase):
    def test_build_user_prompt(self):
        chunks = [{"text": "Sample fact.", "metadata": {"page": 2, "chunk_index": 1}}]
        prompt = build_user_prompt("What is the fact?", chunks)
        self.assertIn("Source 1 (page 2):", prompt)
        self.assertIn("Sample fact.", prompt)
        self.assertIn("Question: What is the fact?", prompt)

    def test_generate_answer_missing_api_key_raises(self):
        chunks = [{"text": "Sample", "metadata": {"page": 1, "chunk_index": 1}}]
        with (
            patch.object(config, "LLM_PROVIDER", "groq"),
            patch("src.generation.answer_generator.GROQ_API_KEY", None),
        ):
            with self.assertRaises((ValueError, LLMConfigurationError)):
                generate_answer("Q?", chunks)

    def test_generate_answer_empty_chunks_raises(self):
        with (
            patch.object(config, "LLM_PROVIDER", "groq"),
            patch("src.generation.answer_generator.GROQ_API_KEY", "test-key"),
        ):
            with self.assertRaises(ValueError):
                generate_answer("Q?", [])

    def test_generate_answer_calls_groq(self):
        chunks = [{"text": "Study Bot uses FAISS.", "metadata": {"page": 1, "chunk_index": 1}}]
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="It uses FAISS (Source 1)."))]

        with (
            patch.object(config, "LLM_PROVIDER", "groq"),
            patch("src.generation.answer_generator.GROQ_API_KEY", "test-key"),
            patch("src.generation.answer_generator.GROQ_MODEL", "openai/gpt-oss-120b"),
            patch("src.generation.answer_generator.Groq") as mock_groq_cls,
        ):
            mock_client = mock_groq_cls.return_value
            mock_client.chat.completions.create.return_value = mock_response

            answer = generate_answer("What vector store is used?", chunks)
            self.assertEqual(answer, "It uses FAISS (Source 1).")
            mock_client.chat.completions.create.assert_called_once()
            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            self.assertEqual(call_kwargs["model"], "openai/gpt-oss-120b")
            self.assertEqual(call_kwargs["messages"][0]["content"], SYSTEM_PROMPT)


class TestStreamlitApp(unittest.TestCase):
    def test_app_renders_initial_state(self):
        import tempfile
        from streamlit.testing.v1 import AppTest

        app_path = Path(__file__).resolve().parent.parent / "app.py"
        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            with patch.object(config, "VECTOR_STORE_DIR", store_dir):
                at = AppTest.from_file(str(app_path)).run(timeout=15)
                self.assertFalse(at.exception)
                self.assertEqual(at.title[0].value, "PDF Chat - RAG Question Answering System")
                self.assertTrue(
                    any("Upload a PDF to start asking questions." in info.value for info in at.info),
                    f"Expected info message not found in {[i.value for i in at.info]}",
                )


if __name__ == "__main__":
    unittest.main()
