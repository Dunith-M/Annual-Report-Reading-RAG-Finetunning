from typing import List, Dict, Any, Optional
from collections import Counter
import regex as re


class PDFCleaner:
    """
    Cleans raw PDF text page by page.

    Main cleaning goals:
        - Remove repeated headers and footers
        - Remove page numbers
        - Fix broken whitespace
        - Remove useless lines
        - Extract possible section title
    """

    def __init__(self, pages: List[Dict[str, Any]]):
        self.pages = pages
        self.repeated_lines = self._detect_repeated_lines()

    def _normalize_line(self, line: str) -> str:
        """
        Normalize a single line for repeated-line detection.
        """
        line = line.strip()
        line = re.sub(r"\s+", " ", line)
        return line

    def _detect_repeated_lines(self, min_repetition_ratio: float = 0.15) -> set:
        """
        Detect repeated lines likely to be headers or footers.

        If a line appears in more than 15% of pages, treat it as repeated noise.
        """
        all_lines = []

        for page in self.pages:
            lines = page.get("text", "").splitlines()
            cleaned_lines = [self._normalize_line(line) for line in lines]
            cleaned_lines = [line for line in cleaned_lines if line]
            all_lines.extend(set(cleaned_lines))

        line_counts = Counter(all_lines)
        total_pages = max(len(self.pages), 1)

        repeated = {
            line
            for line, count in line_counts.items()
            if count / total_pages >= min_repetition_ratio
        }

        return repeated

    def _is_page_number(self, line: str) -> bool:
        """
        Detect simple page numbers.
        """
        line = line.strip()

        patterns = [
            r"^\d+$",
            r"^Page\s+\d+$",
            r"^\d+\s*\|\s*Page$",
            r"^-\s*\d+\s*-$",
        ]

        return any(re.match(pattern, line, flags=re.IGNORECASE) for pattern in patterns)

    def _is_noise_line(self, line: str) -> bool:
        """
        Detect lines that are mostly useless.
        """
        line = line.strip()

        if not line:
            return True

        if self._is_page_number(line):
            return True

        if line in self.repeated_lines:
            return True

        # Very short symbolic lines
        if len(line) <= 2:
            return True

        # Lines made mostly of punctuation/symbols
        if re.match(r"^[\p{P}\p{S}\s]+$", line):
            return True

        return False

    def _fix_broken_whitespace(self, text: str) -> str:
        """
        Fix common PDF whitespace issues.
        """
        text = text.replace("\xa0", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Fix words split by hyphen at line break:
        # exam-
        # ple -> example
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

        # Convert excessive line breaks inside paragraphs.
        # Keep paragraph breaks.
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

        # Normalize final whitespace
        text = re.sub(r"\s{2,}", " ", text)
        text = text.strip()

        return text

    def _extract_section_title(self, raw_text: str) -> Optional[str]:
        """
        Heuristic section-title extraction.

        Looks for early lines that resemble section headings.
        """
        lines = raw_text.splitlines()

        candidate_lines = []

        for line in lines[:15]:
            line = self._normalize_line(line)

            if not line:
                continue

            if self._is_noise_line(line):
                continue

            # Good title candidates:
            # - Short enough
            # - Not too numeric-heavy
            # - Often title case or uppercase
            if 4 <= len(line) <= 100:
                digit_ratio = sum(char.isdigit() for char in line) / max(len(line), 1)

                if digit_ratio < 0.4:
                    candidate_lines.append(line)

        if not candidate_lines:
            return None

        # Prefer a line that looks like a heading
        for line in candidate_lines:
            if line.isupper():
                return line

            words = line.split()
            if len(words) <= 10 and any(word[:1].isupper() for word in words):
                return line

        return candidate_lines[0]

    def clean_page_text(self, text: str) -> str:
        """
        Clean one page's text.
        """
        lines = text.splitlines()

        clean_lines = []

        for line in lines:
            line = self._normalize_line(line)

            if self._is_noise_line(line):
                continue

            clean_lines.append(line)

        cleaned_text = "\n".join(clean_lines)
        cleaned_text = self._fix_broken_whitespace(cleaned_text)

        return cleaned_text

    def clean_pages(self) -> List[Dict[str, Any]]:
        """
        Clean all pages.

        Output:
            [
                {
                    "page_number": 1,
                    "section_title": "...",
                    "text": "...",
                    "source": "pymupdf"
                }
            ]
        """
        cleaned_pages = []

        for page in self.pages:
            raw_text = page.get("text", "")
            cleaned_text = self.clean_page_text(raw_text)
            section_title = self._extract_section_title(raw_text)

            cleaned_pages.append(
                {
                    "page_number": page["page_number"],
                    "section_title": section_title,
                    "text": cleaned_text,
                    "source": page.get("source", "unknown"),
                }
            )

        return cleaned_pages