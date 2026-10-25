"""
CEM Data Engineering: PII Scrubber.
Removes Personally Identifiable Information using Regex.
"""

import re
from typing import Tuple

class PIIScrubber:
    def __init__(self):
        # Industry standard regex patterns
        self.patterns = {
            "EMAIL": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "PHONE": re.compile(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'),
            "SSN": re.compile(r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b'),
            "AADHAAR": re.compile(r'\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b'),
            "IP": re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
        }

    def scrub(self, text: str) -> Tuple[str, dict]:
        """Replaces PII with tags and returns stats."""
        stats = {}
        scrubbed_text = text
        
        for pii_type, pattern in self.patterns.items():
            matches = pattern.findall(scrubbed_text)
            if matches:
                stats[pii_type] = len(matches)
                scrubbed_text = pattern.sub(f"[{pii_type}]", scrubbed_text)
                
        return scrubbed_text, stats

# --- Test ---
if __name__ == "__main__":
    scrubber = PIIScrubber()
    sample = "Mera naam Ravi hai. Mera phone 9876543210 hai aur email ravi@test.com hai."
    clean, stats = scrubber.scrub(sample)
    print(clean) 
    # Output: Mera naam Ravi hai. Mera phone [PHONE] hai aur email [EMAIL] hai.
    print(stats)