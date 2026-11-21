"""Layout-aware PDF text extraction module using PyMuPDF (fitz).

Provides intelligent column detection and reading order preservation:
- Single-column pages: sorted strictly top-to-bottom.
- Multi-column pages: partitions columns separated by spanning headings/figures.
- Spanning elements: titles, abstracts, full-width tables/figures preserved in natural order.
- Safe fallback to geometric sort=True or raw text if layout is ambiguous.
- Incremental progress callbacks for large documents (100–500 pages).
"""

from collections.abc import Callable
import logging
from pathlib import Path
from typing import Any, List, Tuple
import pymupdf

import src.config as config

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]


def extract_page_text_layout_aware(
    page: pymupdf.Page,
    sort_layout: bool = True,
) -> str:
    """Extract text from a PyMuPDF Page preserving logical column reading order.

    Args:
        page: A PyMuPDF Page object.
        sort_layout: If True, detects multi-column layout and orders text logically.
                     If False, uses standard PyMuPDF text extraction.

    Returns:
        Cleaned, ordered page text string.
    """
    if not sort_layout:
        return page.get_text("text").strip()

    try:
        # blocks: list of (x0, y0, x1, y1, text, block_no, block_type)
        raw_blocks = page.get_text("blocks")
        text_blocks = [
            b for b in raw_blocks
            if b[6] == 0 and isinstance(b[4], str) and b[4].strip()
        ]

        if not text_blocks:
            return ""
        if len(text_blocks) == 1:
            return text_blocks[0][4].strip()

        page_width = float(page.rect.width)
        mid_x = page_width / 2.0

        # Classify blocks into spanning vs column candidates
        # Spanning: width exceeds 55% of page or starts in left third and ends in right third
        spanning_blocks: List[Any] = []
        column_blocks: List[Any] = []

        for b in text_blocks:
            x0, y0, x1, y1 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
            block_width = x1 - x0
            is_spanning = (
                block_width > 0.55 * page_width
                or (x0 < 0.35 * page_width and x1 > 0.65 * page_width)
            )
            if is_spanning:
                spanning_blocks.append(b)
            else:
                column_blocks.append(b)

        # Detect if the non-spanning blocks actually form 2 columns
        left_blocks = [b for b in column_blocks if (float(b[0]) + float(b[2])) / 2.0 < mid_x]
        right_blocks = [b for b in column_blocks if (float(b[0]) + float(b[2])) / 2.0 >= mid_x]

        left_chars = sum(len(b[4].strip()) for b in left_blocks)
        right_chars = sum(len(b[4].strip()) for b in right_blocks)

        vertical_overlap = 0.0
        if left_blocks and right_blocks:
            min_left_y = min(float(b[1]) for b in left_blocks)
            max_left_y = max(float(b[3]) for b in left_blocks)
            min_right_y = min(float(b[1]) for b in right_blocks)
            max_right_y = max(float(b[3]) for b in right_blocks)
            vertical_overlap = min(max_left_y, max_right_y) - max(min_left_y, min_right_y)

        is_multi_column = (
            (len(left_blocks) >= 2 and len(right_blocks) >= 2)
            or (
                len(left_blocks) >= 1
                and len(right_blocks) >= 1
                and left_chars >= 30
                and right_chars >= 30
                and vertical_overlap > 0.0
            )
        )

        if not is_multi_column:
            # Single-column page: sort all blocks purely by vertical position (y0)
            sorted_blocks = sorted(text_blocks, key=lambda b: (round(float(b[1]), 1), round(float(b[0]), 1)))
            return "\n\n".join(b[4].strip() for b in sorted_blocks if b[4].strip())

        # Multi-column page with potential spanning headers/footers/figures
        spanning_blocks.sort(key=lambda b: float(b[1]))

        ordered_blocks: List[Any] = []

        # Helper to process a vertical band of column blocks sorted column-by-column
        def _process_band(band_blocks: List[Any]):
            if not band_blocks:
                return
            b_left = sorted(
                [b for b in band_blocks if (float(b[0]) + float(b[2])) / 2.0 < mid_x],
                key=lambda b: float(b[1]),
            )
            b_right = sorted(
                [b for b in band_blocks if (float(b[0]) + float(b[2])) / 2.0 >= mid_x],
                key=lambda b: float(b[1]),
            )
            ordered_blocks.extend(b_left)
            ordered_blocks.extend(b_right)

        # Slice column blocks into bands partitioned by spanning elements without boundary gaps
        num_spans = len(spanning_blocks)
        bands: List[List[Any]] = [[] for _ in range(num_spans + 1)]

        for b in column_blocks:
            b_yc = (float(b[1]) + float(b[3])) / 2.0
            assigned = False
            for s_idx, span_b in enumerate(spanning_blocks):
                span_yc = (float(span_b[1]) + float(span_b[3])) / 2.0
                if b_yc < span_yc:
                    bands[s_idx].append(b)
                    assigned = True
                    break
            if not assigned:
                bands[-1].append(b)

        # Assemble: Band 0 -> Span 0 -> Band 1 -> Span 1 -> ... -> Band N
        for s_idx in range(num_spans):
            _process_band(bands[s_idx])
            ordered_blocks.append(spanning_blocks[s_idx])
        _process_band(bands[-1])

        # Safety invariant: ensure every single block is preserved
        if len(ordered_blocks) != len(text_blocks):
            missing = [b for b in text_blocks if b not in ordered_blocks]
            ordered_blocks.extend(missing)

        return "\n\n".join(b[4].strip() for b in ordered_blocks if b[4].strip())

    except Exception as exc:
        logger.warning("Layout-aware extraction encountered error (%s); falling back to sort=True.", exc)
        return page.get_text("text", sort=True).strip()


def extract_pdf_text(
    pdf_file: Any,
    sort_layout: bool | None = None,
    progress_callback: ProgressCallback | None = None,
) -> List[Tuple[int, str]]:
    """Extract page-numbered text from a PDF file with optional layout-aware column sorting.

    Args:
        pdf_file: A Streamlit UploadedFile, BytesIO, bytes, file path (str/Path), or file-like object.
        sort_layout: If True, uses layout-aware block sorting. If None, reads from config.
        progress_callback: Optional callback reporting (page_number, total_pages, message).

    Returns:
        List of (page_number, page_text) tuples for pages containing non-empty text.
    """
    text_by_page: List[Tuple[int, str]] = []

    if hasattr(pdf_file, "getvalue"):
        pdf_bytes = pdf_file.getvalue()
    elif isinstance(pdf_file, (bytes, bytearray)):
        pdf_bytes = bytes(pdf_file)
    elif hasattr(pdf_file, "read"):
        pdf_bytes = pdf_file.read()
    elif isinstance(pdf_file, (str, Path)):
        with open(pdf_file, "rb") as f:
            pdf_bytes = f.read()
    else:
        pdf_bytes = b""

    if not pdf_bytes:
        return text_by_page

    use_sort_layout = (
        sort_layout
        if sort_layout is not None
        else getattr(config, "PDF_LAYOUT_SORT_ENABLED", True)
    )

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
        total_pages = len(document)
        for page_number, page in enumerate(document, start=1):
            if progress_callback is not None:
                progress_callback(
                    page_number,
                    total_pages,
                    f"Extracting text from page {page_number}/{total_pages}...",
                )
            page_text = extract_page_text_layout_aware(page, sort_layout=use_sort_layout)
            if page_text:
                text_by_page.append((page_number, page_text))

    return text_by_page
