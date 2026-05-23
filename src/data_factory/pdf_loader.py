from pathlib import Path
from typing import List, Dict, Any, Optional

import fitz  # PyMuPDF
import pdfplumber
from tqdm import tqdm


class PDFLoader:
    """
    Loads a PDF page by page and extracts raw text.

    Primary extractor:
        - PyMuPDF

    Fallback extractor:
        - pdfplumber

    Output format:
        [
            {
                "page_number": 1,
                "text": "...",
                "source": "pymupdf"
            }
        ]
    """

    def __init__(self, pdf_path: str | Path):
        self.pdf_path = Path(pdf_path)

        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found at: {self.pdf_path}")

    def _extract_with_pymupdf(self) -> List[Dict[str, Any]]:
        """
        Extract text using PyMuPDF.
        """
        pages = []

        doc = fitz.open(self.pdf_path)

        for page_index in tqdm(range(len(doc)), desc="Extracting PDF with PyMuPDF"):
            page = doc[page_index]
            text = page.get_text("text")

            pages.append(
                {
                    "page_number": page_index + 1,
                    "text": text.strip() if text else "",
                    "source": "pymupdf",
                }
            )

        doc.close()
        return pages

    def _extract_page_with_pdfplumber(self, page_number: int) -> Optional[str]:
        """
        Extract single page text using pdfplumber.
        page_number is 1-based.
        """
        with pdfplumber.open(self.pdf_path) as pdf:
            page = pdf.pages[page_number - 1]
            text = page.extract_text()

        return text.strip() if text else ""

    def load_pages(self, min_text_length: int = 80) -> List[Dict[str, Any]]:
        """
        Load pages from PDF.

        If PyMuPDF extraction gives too little text, fallback to pdfplumber
        for that page.
        """
        pages = self._extract_with_pymupdf()

        for page in tqdm(pages, desc="Checking weak pages with pdfplumber fallback"):
            text = page["text"]

            if len(text) < min_text_length:
                fallback_text = self._extract_page_with_pdfplumber(page["page_number"])

                if fallback_text and len(fallback_text) > len(text):
                    page["text"] = fallback_text
                    page["source"] = "pdfplumber"

        return pages