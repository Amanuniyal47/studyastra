"""Targeted unit tests for layout-aware PDF text extraction (Stage 11.3 Work C)."""

import io
import unittest
from unittest.mock import MagicMock
import pymupdf

from src.ingestion.pdf_loader import extract_pdf_text, extract_page_text_layout_aware


class TestPdfLoaderLayoutAware(unittest.TestCase):
    """Test PDF text extraction across single-column, two-column, and mixed layouts."""

    def _create_synthetic_two_column_pdf(self) -> bytes:
        """Create a PDF page with spanning title and two distinct columns."""
        doc = pymupdf.open()
        page = doc.new_page(width=600, height=800)

        # 1. Spanning title at top
        page.insert_text((50, 60), "Title of Academic Research Paper", fontsize=18)
        page.insert_text((50, 90), "Abstract: This is a full-width abstract spanning across the entire page.", fontsize=11)

        # 2. Column 1 (left half: x in 50..270)
        page.insert_text((50, 150), "Col 1 Para 1: Introduction to methods.", fontsize=10)
        page.insert_text((50, 200), "Col 1 Para 2: Details of experiments.", fontsize=10)
        page.insert_text((50, 250), "Col 1 Para 3: Concluding observations of col 1.", fontsize=10)

        # 3. Column 2 (right half: x in 330..550)
        page.insert_text((330, 150), "Col 2 Para 1: Related work overview.", fontsize=10)
        page.insert_text((330, 200), "Col 2 Para 2: Theoretical derivations.", fontsize=10)
        page.insert_text((330, 250), "Col 2 Para 3: Future research directions.", fontsize=10)

        pdf_bytes = doc.write()
        doc.close()
        return pdf_bytes

    def _create_synthetic_single_column_pdf(self) -> bytes:
        """Create a single-column PDF page."""
        doc = pymupdf.open()
        page = doc.new_page(width=600, height=800)
        page.insert_text((60, 80), "Single Column Heading", fontsize=16)
        page.insert_text((60, 130), "Paragraph one of single column text.", fontsize=11)
        page.insert_text((60, 180), "Paragraph two of single column text.", fontsize=11)
        page.insert_text((60, 230), "Paragraph three of single column text.", fontsize=11)
        pdf_bytes = doc.write()
        doc.close()
        return pdf_bytes

    def test_01_two_column_reading_order_preservation(self):
        """Verify that Column 1 paragraphs appear before Column 2 paragraphs."""
        pdf_bytes = self._create_synthetic_two_column_pdf()
        stream = io.BytesIO(pdf_bytes)

        pages = extract_pdf_text(stream, sort_layout=True)
        self.assertEqual(len(pages), 1)
        page_num, text = pages[0]
        self.assertEqual(page_num, 1)

        # Verify title and abstract are at top
        idx_title = text.find("Title of Academic Research Paper")
        idx_abstract = text.find("Abstract: This is a full-width abstract")
        self.assertTrue(idx_title < idx_abstract)

        # Verify all Column 1 paragraphs appear BEFORE Column 2 paragraphs
        idx_c1_p1 = text.find("Col 1 Para 1")
        idx_c1_p2 = text.find("Col 1 Para 2")
        idx_c1_p3 = text.find("Col 1 Para 3")

        idx_c2_p1 = text.find("Col 2 Para 1")
        idx_c2_p2 = text.find("Col 2 Para 2")
        idx_c2_p3 = text.find("Col 2 Para 3")

        self.assertTrue(idx_abstract < idx_c1_p1)
        self.assertTrue(idx_c1_p1 < idx_c1_p2 < idx_c1_p3)
        self.assertTrue(idx_c1_p3 < idx_c2_p1, "All Col 1 text must precede Col 2 text in reading order")
        self.assertTrue(idx_c2_p1 < idx_c2_p2 < idx_c2_p3)

    def test_02_single_column_preserves_reading_order(self):
        """Verify single-column pages are extracted correctly without layout disruption."""
        pdf_bytes = self._create_synthetic_single_column_pdf()
        stream = io.BytesIO(pdf_bytes)

        pages = extract_pdf_text(stream, sort_layout=True)
        self.assertEqual(len(pages), 1)
        _, text = pages[0]

        idx_h = text.find("Single Column Heading")
        idx_p1 = text.find("Paragraph one")
        idx_p2 = text.find("Paragraph two")
        idx_p3 = text.find("Paragraph three")

        self.assertTrue(idx_h < idx_p1 < idx_p2 < idx_p3)

    def test_03_progress_callback_emits_per_page(self):
        """Verify progress callback is called for each page during multi-page extraction."""
        doc = pymupdf.open()
        for i in range(3):
            page = doc.new_page()
            page.insert_text((50, 50), f"Page {i+1} content")
        pdf_bytes = doc.write()
        doc.close()

        callback_records = []
        def _cb(page, total, msg):
            callback_records.append((page, total, msg))

        pages = extract_pdf_text(io.BytesIO(pdf_bytes), progress_callback=_cb)
        self.assertEqual(len(pages), 3)
        self.assertEqual(len(callback_records), 3)
        self.assertEqual(callback_records[0][0], 1)
        self.assertEqual(callback_records[0][1], 3)
        self.assertEqual(callback_records[2][0], 3)
        self.assertEqual(callback_records[2][1], 3)

    def test_04_empty_document_handling(self):
        """Verify empty byte streams return empty list safely."""
        self.assertEqual(extract_pdf_text(b""), [])
        self.assertEqual(extract_pdf_text(io.BytesIO(b"")), [])

    def test_05_uneven_block_counts_two_column_no_interleaving(self):
        """Verify 2 blocks in Col 1 and 1 block in Col 2 do not interleave horizontally."""
        doc = pymupdf.open()
        page = doc.new_page(width=600, height=800)

        # Col 1 has two paragraphs (y in 100..200 and 210..300)
        page.insert_text((50, 120), "Left Col Para 1: Substantial discussion of neural networks.", fontsize=10)
        page.insert_text((50, 220), "Left Col Para 2: Substantial discussion of backpropagation.", fontsize=10)

        # Col 2 has one paragraph (y in 160..260, interleaves horizontally if sorted by y)
        page.insert_text((340, 160), "Right Col Para 1: Substantial discussion of transformer attention.", fontsize=10)

        pdf_bytes = doc.write()
        doc.close()

        pages = extract_pdf_text(io.BytesIO(pdf_bytes), sort_layout=True)
        _, text = pages[0]

        idx_l1 = text.find("Left Col Para 1")
        idx_l2 = text.find("Left Col Para 2")
        idx_r1 = text.find("Right Col Para 1")

        self.assertTrue(idx_l1 < idx_l2, "Left Col Para 1 must precede Left Col Para 2")
        self.assertTrue(idx_l2 < idx_r1, "All Left Col text must precede Right Col text (no interleaving)")

    def test_06_spanning_figure_in_middle_between_columns(self):
        """Verify spanning elements in the middle of a page divide columns into upper and lower bands."""
        doc = pymupdf.open()
        page = doc.new_page(width=600, height=900)

        # Top spanning title
        page.insert_text((50, 60), "Top Title: Distributed Systems", fontsize=18)

        # Upper band: Col 1 & Col 2
        page.insert_text((50, 120), "Upper Col 1: Consensus algorithms introduction.", fontsize=10)
        page.insert_text((340, 120), "Upper Col 2: Paxos and Raft overview.", fontsize=10)

        # Mid-page spanning figure caption (width > 0.55 * 600 = 330)
        page.insert_text((50, 350), "Figure 1: Full-width architecture diagram spanning across the entire system pipeline.", fontsize=11)

        # Lower band: Col 1 & Col 2
        page.insert_text((50, 450), "Lower Col 1: Performance metrics and benchmark analysis.", fontsize=10)
        page.insert_text((340, 450), "Lower Col 2: Fault tolerance and recovery results.", fontsize=10)

        pdf_bytes = doc.write()
        doc.close()

        pages = extract_pdf_text(io.BytesIO(pdf_bytes), sort_layout=True)
        _, text = pages[0]

        idx_title = text.find("Top Title")
        idx_u_c1 = text.find("Upper Col 1")
        idx_u_c2 = text.find("Upper Col 2")
        idx_fig = text.find("Figure 1")
        idx_l_c1 = text.find("Lower Col 1")
        idx_l_c2 = text.find("Lower Col 2")

        # Expected logical order: Top Title -> Upper Col 1 -> Upper Col 2 -> Figure 1 -> Lower Col 1 -> Lower Col 2
        self.assertTrue(idx_title < idx_u_c1 < idx_u_c2 < idx_fig < idx_l_c1 < idx_l_c2)

    def test_07_uneven_column_widths(self):
        """Verify reading order with uneven column widths (wide left column + narrow right sidebar)."""
        doc = pymupdf.open()
        page = doc.new_page(width=600, height=800)

        # Wide main content column on left (x in 50..380)
        page.insert_text((50, 100), "Main Article Para 1: Comprehensive historical analysis.", fontsize=10)
        page.insert_text((50, 180), "Main Article Para 2: Primary investigative findings.", fontsize=10)

        # Narrow sidebar column on right (x in 410..560)
        page.insert_text((410, 100), "Sidebar Para 1: Biographical summary.", fontsize=10)
        page.insert_text((410, 180), "Sidebar Para 2: Chronological timeline.", fontsize=10)

        pdf_bytes = doc.write()
        doc.close()

        pages = extract_pdf_text(io.BytesIO(pdf_bytes), sort_layout=True)
        _, text = pages[0]

        idx_m1 = text.find("Main Article Para 1")
        idx_m2 = text.find("Main Article Para 2")
        idx_s1 = text.find("Sidebar Para 1")
        idx_s2 = text.find("Sidebar Para 2")

        self.assertTrue(idx_m1 < idx_m2 < idx_s1 < idx_s2)


if __name__ == "__main__":
    unittest.main()
