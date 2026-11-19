import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pymupdf

from src.generation.prompts import build_user_prompt
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.retriever import retrieve_chunks
from src.retrieval.vector_store import (
    DuplicateDocumentError,
    FAISSVectorStore,
    VectorStoreCompatibilityError,
    VectorStoreCorruptionError,
    VectorStorePersistenceError,
)


class NamedFakeUpload:
    def __init__(self, data: bytes, name: str):
        self._data = data
        self.name = name

    def getvalue(self) -> bytes:
        return self._data


def make_pdf(pages: list[str], name: str) -> NamedFakeUpload:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(50, 50, 545, 780), text, fontsize=10)
    raw = doc.tobytes()
    doc.close()
    return NamedFakeUpload(raw, name=name)


class TopicEmbeddingModel:
    """Deterministic 8-dim topic embedder for offline multi-PDF tests."""

    TOPICS = (
        ("biology", "cell", "mitosis", "photosynthesis", "dna"),
        ("history", "empire", "revolution", "treaty", "war"),
        ("faiss", "vector", "embedding", "index", "similarity"),
        ("calculus", "derivative", "integral", "limit", "function"),
        ("physics", "newton", "gravity", "force", "velocity"),
        ("chemistry", "atom", "molecule", "acid", "reaction"),
        ("streamlit", "python", "web", "interface", "widget"),
    )

    def __init__(self, dim: int = 8):
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
            vec = np.zeros(self._dim, dtype=np.float32)
            for idx, keywords in enumerate(self.TOPICS):
                if idx < self._dim - 1:
                    vec[idx] = float(sum(1 for kw in keywords if kw in lower))
            vec[-1] = 0.05  # Prevent zero-norm vectors for unrelated text
            rows.append(vec)
        return np.array(rows, dtype=np.float32)


class TestStage3PersistentVectorStore(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.store_dir = Path(self._tmpdir.name) / "vector_store"
        self.fake_model = TopicEmbeddingModel(dim=8)
        self._patcher = patch(
            "src.retrieval.embeddings.load_embedding_model",
            return_value=self.fake_model,
        )
        self._patcher.start()
        self.service = EmbeddingService(model_name="all-MiniLM-L6-v2")

    def tearDown(self):
        self._patcher.stop()
        self._tmpdir.cleanup()

    def _open_store(self, model_name: str = "all-MiniLM-L6-v2", dim: int | None = 8) -> FAISSVectorStore:
        return FAISSVectorStore(
            store_dir=self.store_dir,
            embedding_model=model_name,
            embedding_dimension=dim,
            embedding_service=self.service if model_name == self.service.model_name else None,
        )

    def test_multi_pdf_persistence_restart_retrieval_and_deletion(self):
        pdf_bio = make_pdf(
            [
                "Biology page 1: Photosynthesis in plant cells converts sunlight into glucose.",
                "Biology page 2: DNA replication precedes mitosis during cell division.",
            ],
            name="biology_notes.pdf",
        )
        pdf_cs = make_pdf(
            [
                "CS page 1: FAISS stores dense vector embeddings for fast similarity search.",
                "CS page 2: Streamlit builds Python web interface widgets quickly.",
            ],
            name="cs_notes.pdf",
        )

        store = self._open_store()
        res1 = store.add_pdf(pdf_bio, embedding_service=self.service)
        res2 = store.add_pdf(pdf_cs, embedding_service=self.service)

        self.assertEqual(res1["status"], "indexed")
        self.assertEqual(res2["status"], "indexed")
        self.assertEqual(store.total_documents, 2)
        self.assertEqual(store.total_vectors, 4)
        self.assertEqual(res1["vector_ids"], [1, 2])
        self.assertEqual(res2["vector_ids"], [3, 4])
        self.assertEqual(store.next_vector_id, 5)

        # Simulate full application restart (brand-new store loaded from disk)
        restarted_store = self._open_store()
        self.assertEqual(restarted_store.total_documents, 2)
        self.assertEqual(restarted_store.total_vectors, 4)
        self.assertEqual(restarted_store.next_vector_id, 5)

        bio_hits = retrieve_chunks(
            "How does photosynthesis work in plant cells?",
            top_k=2,
            store=restarted_store,
            embedding_service=self.service,
        )
        self.assertEqual(bio_hits[0]["metadata"]["filename"], "biology_notes.pdf")
        self.assertEqual(bio_hits[0]["metadata"]["page"], 1)
        self.assertGreater(bio_hits[0]["score"], 0.8)

        cs_hits = retrieve_chunks(
            "How does FAISS vector similarity index work?",
            top_k=2,
            store=restarted_store,
            embedding_service=self.service,
        )
        self.assertEqual(cs_hits[0]["metadata"]["filename"], "cs_notes.pdf")
        self.assertEqual(cs_hits[0]["metadata"]["page"], 1)

        # Verify backwards compatibility with Groq prompt builder
        prompt = build_user_prompt("What is FAISS?", cs_hits)
        self.assertIn("Source 1 (page 1):", prompt)

        # Delete biology PDF and verify CS PDF stays intact
        del_res = restarted_store.delete_document(res1["document_id"])
        self.assertEqual(del_res["status"], "deleted")
        self.assertEqual(del_res["removed_vectors"], 2)
        self.assertEqual(restarted_store.total_documents, 1)
        self.assertEqual(restarted_store.total_vectors, 2)
        # Monotonic IDs never reset after deletion
        self.assertEqual(restarted_store.next_vector_id, 5)

        # Restart again after deletion
        after_delete_store = self._open_store()
        self.assertEqual(after_delete_store.total_documents, 1)
        self.assertEqual(after_delete_store.total_vectors, 2)
        self.assertEqual(after_delete_store.next_vector_id, 5)
        self.assertFalse(after_delete_store.is_document_indexed(res1["document_id"]))
        self.assertTrue(after_delete_store.is_document_indexed(res2["document_id"]))

        remaining_hits = retrieve_chunks(
            "FAISS vector similarity search",
            top_k=2,
            store=after_delete_store,
            embedding_service=self.service,
        )
        self.assertTrue(all(h["metadata"]["filename"] == "cs_notes.pdf" for h in remaining_hits))

        # Add a third PDF and verify vector IDs continue monotonically from 5 (never reuse 1 or 2)
        pdf_phys = make_pdf(["Physics: Newton laws explain force and gravity."], name="physics.pdf")
        res3 = after_delete_store.add_pdf(pdf_phys, embedding_service=self.service)
        self.assertEqual(res3["vector_ids"], [5])
        self.assertEqual(after_delete_store.next_vector_id, 6)

    def test_duplicate_pdf_skipped_without_re_embedding(self):
        pdf = make_pdf(["Calculus covers derivatives and integrals."], name="calc.pdf")
        store = self._open_store()

        first = store.add_pdf(pdf, embedding_service=self.service)
        calls_after_first = self.fake_model.encode_calls
        second = store.add_pdf(pdf, embedding_service=self.service)

        self.assertEqual(first["status"], "indexed")
        self.assertEqual(second["status"], "duplicate")
        self.assertTrue(second["already_indexed"])
        self.assertEqual(second["embeddings_generated"], 0)
        self.assertEqual(self.fake_model.encode_calls, calls_after_first)
        self.assertEqual(store.total_documents, 1)
        self.assertEqual(store.total_vectors, 1)

        with self.assertRaises(DuplicateDocumentError):
            store.add_pdf(pdf, embedding_service=self.service, skip_duplicates=False)

    def test_empty_pdf_and_repeated_text_pdf(self):
        store = self._open_store()
        empty_pdf = make_pdf(["", "   "], name="empty.pdf")

        with self.assertRaises(ValueError):
            store.add_pdf(empty_pdf, embedding_service=self.service)
        self.assertEqual(store.total_documents, 0)
        self.assertEqual(store.total_vectors, 0)
        self.assertEqual(store.active_generation, 0)

        repeated_pdf = make_pdf(
            [
                "Chemistry studies atoms and molecules in reactions.",
                "Chemistry studies atoms and molecules in reactions.",
            ],
            name="repeated.pdf",
        )
        res = store.add_pdf(repeated_pdf, embedding_service=self.service)
        self.assertEqual(res["chunks_generated"], 1)
        self.assertEqual(store.total_vectors, 1)

    def test_document_filtered_search_and_unrelated_query(self):
        store = self._open_store()
        r1 = store.add_pdf(
            make_pdf(["History discusses ancient empire treaties and revolution."], name="hist.pdf"),
            embedding_service=self.service,
        )
        store.add_pdf(
            make_pdf(["Biology discusses plant cells and DNA photosynthesis."], name="bio.pdf"),
            embedding_service=self.service,
        )

        filtered = retrieve_chunks(
            "plant cells photosynthesis",
            top_k=3,
            document_id=r1["document_id"],
            store=store,
            embedding_service=self.service,
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["metadata"]["filename"], "hist.pdf")

        unrelated = retrieve_chunks(
            "unrelated culinary recipe for sourdough bread",
            top_k=2,
            store=store,
            embedding_service=self.service,
        )
        self.assertEqual(len(unrelated), 2)
        # Score for unrelated text should be much lower than a topic match
        self.assertLess(unrelated[0]["score"], 0.5)

    def test_failed_indexing_and_failed_persistence_roll_back_cleanly(self):
        store = self._open_store()
        pdf1 = make_pdf(["Physics explains gravity and force."], name="phys.pdf")
        res1 = store.add_pdf(pdf1, embedding_service=self.service)
        gen_before = store.active_generation

        # Simulate persistence failure mid-commit on adding pdf2
        pdf2 = make_pdf(["Chemistry explains acids and molecules."], name="chem.pdf")
        with patch("src.retrieval.vector_store.faiss.write_index", side_effect=OSError("disk full")):
            with self.assertRaises(VectorStorePersistenceError):
                store.add_pdf(pdf2, embedding_service=self.service)

        # In-memory state must remain untouched
        self.assertEqual(store.total_documents, 1)
        self.assertEqual(store.total_vectors, 1)
        self.assertEqual(store.active_generation, gen_before)
        self.assertTrue(store.is_document_indexed(res1["document_id"]))

        # Reopening from disk must still cleanly load generation 1
        reopened = self._open_store()
        self.assertEqual(reopened.total_documents, 1)
        self.assertEqual(reopened.total_vectors, 1)

        # Simulate persistence failure during delete_document
        with patch("src.retrieval.vector_store.os.replace", side_effect=OSError("io error")):
            with self.assertRaises(VectorStorePersistenceError):
                reopened.delete_document(res1["document_id"])

        # Document must still exist both in memory and on disk
        self.assertEqual(reopened.total_documents, 1)
        self.assertEqual(reopened.total_vectors, 1)
        reopened2 = self._open_store()
        self.assertEqual(reopened2.total_documents, 1)

    def test_incompatible_embedding_model_or_dimension_rejected(self):
        store = self._open_store()
        store.add_pdf(make_pdf(["FAISS vector index notes."], name="v.pdf"), embedding_service=self.service)

        with self.assertRaises(VectorStoreCompatibilityError):
            self._open_store(model_name="BAAI/bge-small-en-v1.5", dim=8)

        with self.assertRaises(VectorStoreCompatibilityError):
            self._open_store(model_name="all-MiniLM-L6-v2", dim=768)

    def test_corruption_and_desync_detection(self):
        store = self._open_store()
        store.add_pdf(
            make_pdf(["Biology cells DNA.", "History empire treaty."], name="mix.pdf"),
            embedding_service=self.service,
        )
        manifest = json.loads((self.store_dir / "manifest.json").read_text(encoding="utf-8"))

        # 1. Corrupted metadata JSON
        meta_path = self.store_dir / manifest["metadata_file"]
        original_meta = meta_path.read_text(encoding="utf-8")
        meta_path.write_text("{broken json", encoding="utf-8")
        with self.assertRaises(VectorStoreCorruptionError):
            self._open_store()

        # 2. Metadata chunk count desync vs FAISS
        parsed = json.loads(original_meta)
        parsed["chunks"].pop("1")
        meta_path.write_text(json.dumps(parsed), encoding="utf-8")
        with self.assertRaises(VectorStoreCorruptionError):
            self._open_store()
        meta_path.write_text(original_meta, encoding="utf-8")

        # 3. Unregistered document reference in documents.json
        docs_path = self.store_dir / manifest["documents_file"]
        original_docs = docs_path.read_text(encoding="utf-8")
        docs_path.write_text(json.dumps({"schema_version": 1, "documents": {}}), encoding="utf-8")
        with self.assertRaises(VectorStoreCorruptionError):
            self._open_store()
        docs_path.write_text(original_docs, encoding="utf-8")

        # 4. Corrupted FAISS binary
        idx_path = self.store_dir / manifest["index_file"]
        original_idx = idx_path.read_bytes()
        idx_path.write_bytes(b"not a valid faiss index")
        with self.assertRaises(VectorStoreCorruptionError):
            self._open_store()
        idx_path.write_bytes(original_idx)

        # 5. Missing manifest while generation files exist
        manifest_path = self.store_dir / "manifest.json"
        original_manifest = manifest_path.read_text(encoding="utf-8")
        manifest_path.unlink()
        with self.assertRaises(VectorStoreCorruptionError):
            self._open_store()
        manifest_path.write_text(original_manifest, encoding="utf-8")

        # Sanity: restored store opens cleanly again
        self.assertEqual(self._open_store().total_vectors, 2)

    def test_large_pdf_indexing_and_retrieval(self):
        pages = [
            f"Page {i}: "
            + " ".join(
                f"Fact {i}.{j} covers {'FAISS vector similarity' if i == 15 else 'general study material'}."
                for j in range(30)
            )
            for i in range(1, 21)
        ]
        large_pdf = make_pdf(pages, name="large_textbook.pdf")
        store = self._open_store()

        res = store.add_pdf(large_pdf, embedding_service=self.service)
        self.assertGreater(res["chunks_generated"], 20)
        self.assertEqual(store.total_vectors, res["chunks_generated"])

        hits = retrieve_chunks(
            "FAISS vector similarity",
            top_k=3,
            store=store,
            embedding_service=self.service,
        )
        self.assertEqual(hits[0]["metadata"]["page"], 15)
        self.assertEqual(hits[0]["metadata"]["filename"], "large_textbook.pdf")

    def test_invalid_query_rejected(self):
        store = self._open_store()
        store.add_pdf(make_pdf(["Sample text."], name="s.pdf"), embedding_service=self.service)
        with self.assertRaises(ValueError):
            retrieve_chunks("   ", store=store, embedding_service=self.service)


if __name__ == "__main__":
    unittest.main()
