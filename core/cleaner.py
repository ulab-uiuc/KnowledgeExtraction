import re
from typing import List

class KnowledgeCleaner:
    @staticmethod
    def clean_bullet_points(text: str) -> List[str]:
        """
        Extract and clean bullet points from LLM output.
        Handles various formats: -, *, •, numbered lists, etc.
        """
        points = []
        
        # Pattern 1: Standard bullet points (-, *, •, etc.)
        bullet_pattern = r'^\s*[\-\*\u2022\u25E6\u25AA]\s+(.+)$'
        matches = re.findall(bullet_pattern, text, re.MULTILINE)
        points.extend(matches)
        
        # Pattern 2: Numbered lists (1., 2., etc.)
        numbered_pattern = r'^\s*\d+[\.\)]\s+(.+)$'
        matches = re.findall(numbered_pattern, text, re.MULTILINE)
        points.extend(matches)
        
        # Pattern 3: If no structured format found, split by lines
        if not points:
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                # Skip empty lines, headers, or very short lines
                if line and len(line) > 3 and not line.startswith('#'):
                    # Remove common prefixes
                    line = re.sub(r'^[\-\*\u2022\d\.\)\s]+', '', line)
                    if line:
                        points.append(line)
        
        # Clean each point
        cleaned = []
        for point in points:
            # Remove markdown formatting
            point = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', point)  # Remove **bold**
            point = re.sub(r'_{1,2}([^_]+)_{1,2}', r'\1', point)  # Remove __italic__
            point = re.sub(r'#{1,6}\s+', '', point)  # Remove markdown headers
            
            # Remove trailing punctuation artifacts
            point = re.sub(r'[*_#\s]+$', '', point)
            
            # Trim whitespace
            point = point.strip()
            
            # Filter out very short or likely header-only entries
            if point and len(point) > 5:
                cleaned.append(point)
        
        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for p in cleaned:
            # Normalize for comparison (lowercase, remove extra spaces)
            normalized = ' '.join(p.lower().split())
            if normalized not in seen:
                seen.add(normalized)
                unique.append(p)
        
        return unique

