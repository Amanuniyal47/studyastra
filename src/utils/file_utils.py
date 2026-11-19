import hashlib
def file_hash(pdf_file):
    return hashlib.sha256(pdf_file.getvalue()).hexdigest()