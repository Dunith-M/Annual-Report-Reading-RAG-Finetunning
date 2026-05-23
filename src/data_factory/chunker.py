from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PDFChunker:
    """
    Splits cleaned PDF pages into 1500-character chunks.

    Each chunk includes:
        - chunk_id
        - page_start
        - page_end
        - section_title
        - text
        - char_count
    """

    def __init__(
        self,
        document_id: str = "uber_2024",
        chunk_size: int = 1500,
        chunk_overlap: int = 150,
    ):
        self.document_id = document_id
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )

    def _create_chunk_id(self, page_start: int, chunk_number: int) -> str:
        """
        Create chunk ID like:
        uber_2024_p10_c001
        """
        return f"{self.document_id}_p{page_start}_c{chunk_number:03d}"

    def chunk_pages(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk cleaned pages page-by-page.

        This keeps page metadata clean and simple.
        """
        all_chunks = []
        global_chunk_counter = 1

        for page in pages:
            page_number = page["page_number"]
            text = page.get("text", "")
            section_title = page.get("section_title")

            if not text or len(text.strip()) < 30:
                continue

            text_chunks = self.splitter.split_text(text)

            for chunk_text in text_chunks:
                chunk_text = chunk_text.strip()

                if not chunk_text:
                    continue

                chunk = {
                    "chunk_id": self._create_chunk_id(
                        page_start=page_number,
                        chunk_number=global_chunk_counter,
                    ),
                    "page_start": page_number,
                    "page_end": page_number,
                    "section_title": section_title,
                    "text": chunk_text,
                    "char_count": len(chunk_text),
                }

                all_chunks.append(chunk)
                global_chunk_counter += 1

        return all_chunks