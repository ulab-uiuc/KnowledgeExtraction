import re

class KnowledgeCleaner:
    @staticmethod
    def clean_bullet_points(text: str, min_length: int = 5, require_space: bool = True):
        """Extracts knowledge points from bulleted or numbered lists, removing thoughts.

        Args:
            text: Raw text containing bullet points
            min_length: Minimum character length for a valid point (default 5 for taxonomy, use 20+ for knowledge points)
            require_space: If True, filter out items without spaces (for knowledge points).
                          Set to False for taxonomy parsing where single terms like "LSTM" are valid.
        """
        if not text:
            return []
            
        # 0. Pre-process to remove DeepSeek-style thought blocks
        text = re.sub(r'<thought>.*?</thought>', '', text, flags=re.DOTALL)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = re.sub(r'<thought>.*', '', text, flags=re.DOTALL) if '<thought>' in text and '</thought>' not in text else text
        text = re.sub(r'<think>.*', '', text, flags=re.DOTALL) if '<think>' in text and '</think>' not in text else text
        
        lines = text.split('\n')
        lines = [line for line in lines if '<think>' not in line and '</think>' not in line]

        # 1. Parse line by line to extract bullet items
        # This approach preserves hyphenated words like "Long Short-Term Memory"
        bullet_pattern = re.compile(r'^\s*[\*\-\•]\s+(.+)$')  # - item, * item, • item
        numbered_pattern = re.compile(r'^\s*\d+[\.\)]\s+(.+)$')  # 1. item, 1) item

        points = []
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Try bullet pattern first
            match = bullet_pattern.match(line)
            if match:
                points.append(match.group(1))
                continue

            # Try numbered pattern
            match = numbered_pattern.match(line)
            if match:
                points.append(match.group(1))
        
        # 2. Cleanup and filter
        cleaned = []
        skip_prefixes = (
            "here are", "i will", "sure", "ok", "let me", "the following",
            "below are", "these are", "this is", "note that", "note:",
            "example", "for instance", "such as", "e.g.", "i.e.",
            "no additional", "none", "n/a"
        )
        for p in points:
            p = p.strip()
            # Skip if too short
            if len(p) < min_length:
                continue
            # Skip meta-statements and common non-knowledge prefixes
            if p.lower().startswith(skip_prefixes):
                continue
            # Skip if doesn't contain a space (likely just a term, not a statement)
            # This check is skipped for taxonomy parsing where single terms are valid
            if require_space and ' ' not in p:
                continue
            # Remove trailing punctuation
            p = re.sub(r'[.:;]$', '', p)
            cleaned.append(p)

        return list(set(cleaned))
