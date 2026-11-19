import pymupdf


def extract_pdf_text(pdf_file):
    text_by_page = []
    pdf_bytes = pdf_file.getvalue()

    if not pdf_bytes:
        return text_by_page

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
        for page_number, page in enumerate(document, start=1):
            page_text = page.get_text().strip()
            if page_text:
                text_by_page.append((page_number, page_text))

    return text_by_page
