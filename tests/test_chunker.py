import time
import unittest

from src import config
from src.ingestion.chunker import chunk_text, estimate_tokens


class TestStage1Config(unittest.TestCase):
    def test_default_stage1_configuration(self):
        self.assertEqual(config.EMBEDDING_MODEL, "all-MiniLM-L6-v2")
        self.assertEqual(config.CHUNK_SIZE_TOKENS, 220)
        self.assertEqual(config.CHUNK_OVERLAP_TOKENS, 40)
        self.assertGreater(config.CHUNK_SIZE_TOKENS, config.CHUNK_OVERLAP_TOKENS)
        self.assertGreater(config.EMBEDDING_BATCH_SIZE, 0)
        self.assertGreater(config.TOP_K, 0)
        self.assertTrue(str(config.VECTOR_STORE_DIR).endswith("vector_store"))


class TestStage1Chunker(unittest.TestCase):
    def test_preserves_paragraphs_within_chunk(self):
        page = "First paragraph about mitosis.\n\nSecond paragraph about meiosis."
        chunks = chunk_text([(1, page)], document_id="abc123456789", filename="bio.pdf")
        self.assertEqual(len(chunks), 1)
        self.assertIn("\n\n", chunks[0]["text"])
        self.assertEqual(chunks[0]["chunk_id"], "abc123456789_p1_c1")
        self.assertEqual(chunks[0]["filename"], "bio.pdf")
        self.assertEqual(chunks[0]["page"], 1)
        self.assertEqual(chunks[0]["page_range"], "1")

    def test_sentence_boundaries_preserved_and_overlap_applied(self):
        sentences = [
            f"Concept {i}: Photosynthesis converts light energy into chemical energy stored in glucose."
            for i in range(1, 35)
        ]
        page_text = " ".join(sentences)
        chunks = chunk_text(
            [(3, page_text)],
            document_id="doc_bio_001",
            filename="biology.pdf",
            chunk_size=120,
            chunk_overlap=30,
        )
        self.assertGreaterEqual(len(chunks), 3)

        # Every chunk should end cleanly on a sentence period (no mid-sentence cuts)
        for ch in chunks:
            self.assertTrue(ch["text"].endswith("."), f"Cut mid-sentence: {ch['text'][-30:]}")
            self.assertLessEqual(ch["token_count"], 120)
            self.assertEqual(ch["page"], 3)

        # Consecutive chunks should share overlapping content
        first_tail = chunks[0]["text"].split(".")[-2].strip()
        self.assertIn(first_tail, chunks[1]["text"])

    def test_multi_page_topics_stay_attributed_to_exact_pages(self):
        pages = [
            (1, "Biology covers cells, DNA replication, and protein synthesis."),
            (2, "World History examines trade routes and the Industrial Revolution."),
            (5, "Calculus introduces limits, derivatives, and integrals."),
        ]
        chunks = chunk_text(pages, document_id="multi_topic_doc", filename="guide.pdf")
        self.assertEqual(len(chunks), 3)
        self.assertEqual([c["page"] for c in chunks], [1, 2, 5])
        self.assertEqual([c["page_range"] for c in chunks], ["1", "2", "5"])
        self.assertEqual([c["chunk_index"] for c in chunks], [1, 2, 3])

    def test_multiple_documents_produce_unique_chunk_ids(self):
        pages = [(1, "Shared introductory sentence for testing.")]
        chunks_a = chunk_text(pages, document_id="aaaa111122223333", filename="a.pdf")
        chunks_b = chunk_text(pages, document_id="bbbb444455556666", filename="b.pdf")
        self.assertNotEqual(chunks_a[0]["chunk_id"], chunks_b[0]["chunk_id"])
        self.assertEqual(chunks_a[0]["filename"], "a.pdf")
        self.assertEqual(chunks_b[0]["filename"], "b.pdf")

    def test_empty_and_whitespace_pages_produce_no_chunks(self):
        self.assertEqual(chunk_text([]), [])
        self.assertEqual(chunk_text([(1, ""), (2, "   \n\t  ")]), [])

    def test_repeated_identical_text_deduplicated(self):
        pages = [
            (1, "Repeated study definition about Newton's second law."),
            (2, "Repeated study definition about Newton's second law."),
            (3, "  REPEATED   study definition about Newton's second law.  "),
        ]
        chunks = chunk_text(pages, document_id="repeat_doc", filename="physics.pdf")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["page"], 1)

    def test_unbroken_long_string_splits_cleanly(self):
        giant_token = "".join(f"tok{i:04d}" for i in range(300))
        chunks = chunk_text([(1, giant_token)], chunk_size=100, chunk_overlap=20)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(c["text"] for c in chunks))

    def test_large_document_performance_and_token_bounds(self):
        pages = []
        for page_num in range(1, 41):
            paragraph = " ".join(
                f"Section {page_num}.{i}: algorithms optimize search and retrieval across large corpora."
                for i in range(45)
            )
            pages.append((page_num, f"Page {page_num} header.\n\n{paragraph}"))

        start = time.perf_counter()
        chunks = chunk_text(pages, document_id="large_doc_hash", filename="textbook.pdf")
        elapsed = time.perf_counter() - start

        self.assertGreater(len(chunks), 100)
        self.assertLess(elapsed, 2.0, f"Chunking large doc took too long: {elapsed:.2f}s")
        self.assertEqual(len({c["chunk_id"] for c in chunks}), len(chunks))
        self.assertTrue(all(0 < c["token_count"] <= config.CHUNK_SIZE_TOKENS for c in chunks))

    def test_invalid_arguments_raise_value_error(self):
        with self.assertRaises(ValueError):
            chunk_text([(1, "text")], chunk_size=0, chunk_overlap=0)
        with self.assertRaises(ValueError):
            chunk_text([(1, "text")], chunk_size=100, chunk_overlap=-1)
        with self.assertRaises(ValueError):
            chunk_text([(1, "text")], chunk_size=100, chunk_overlap=100)
        with self.assertRaises(ValueError):
            chunk_text([(1, "text")], chunk_size=50, chunk_overlap=60)

    def test_estimate_tokens_handles_empty(self):
        self.assertEqual(estimate_tokens(""), 0)
        self.assertEqual(estimate_tokens("   "), 0)
        self.assertGreater(estimate_tokens("Hello, world!"), 0)


if __name__ == "__main__":
    unittest.main()
