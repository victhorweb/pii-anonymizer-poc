from io import BytesIO

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = (".txt", ".pdf")
STRICT_TEXT_ENCODINGS = ("utf-8-sig", "cp1252")
LENIENT_TEXT_ENCODING = "latin-1"


class UnsupportedFileError(ValueError):
    pass


def extract_text(filename: str, content: bytes) -> str:
    lowered = (filename or "").lower()
    if lowered.endswith(".pdf"):
        return _extract_pdf(content)
    if lowered.endswith(".txt"):
        return _decode_text(content)
    raise UnsupportedFileError(f"Formato não suportado: {filename}. Use {', '.join(SUPPORTED_EXTENSIONS)}.")


def _extract_pdf(content: bytes) -> str:
    reader = PdfReader(BytesIO(content))
    return "\n".join((page.extract_text() or "") for page in reader.pages).strip()


def _decode_text(content: bytes) -> str:
    for encoding in STRICT_TEXT_ENCODINGS:
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode(LENIENT_TEXT_ENCODING)
