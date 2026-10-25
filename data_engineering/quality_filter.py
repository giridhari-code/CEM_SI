"""
CEM Data Engineering: Quality Filter.
Applies Gopher-style heuristics to remove junk text.
"""

import re
from typing import List, Tuple

class GopherQualityFilter:
    def __init__(self, min_words: int = 5, max_symbol_ratio: float = 0.2):
        self.min_words = min_words
        self.max_symbol_ratio = max_symbol_ratio
        # Blocklist for boilerplate
        self.blocklist = ["lorem ipsum", "click here", "privacy policy", "terms of service"]

    def _is_high_quality(self, text: str) -> Tuple[bool, str]:
        text_lower = text.lower()
        
        # 1. Blocklist check
        for phrase in self.blocklist:
            if phrase in text_lower:
                return False, "blocklist_phrase"
                
        # 2. Length check
        words = text.split()
        if len(words) < self.min_words:
            return False, "too_short"
            
        # 3. Symbol ratio check (gibberish detection)
        symbol_count = sum(1 for c in text if c in "!@#$%^&*()_+{}[]|\\:;\"'<>,.?/")
        if symbol_count / max(len(text), 1) > self.max_symbol_ratio:
            return False, "too_many_symbols"
            
        # 4. Language check (Basic English/Hindi check)
        # In production, use fastText for LangID
        if not re.search(r'[a-zA-Z\u0900-\u097F]', text):
            return False, "invalid_language"
            
        return True, "passed"

    def filter_documents(self, documents: List[str]) -> Tuple[List[str], dict]:
        """Returns clean documents and rejection stats."""
        clean_docs = []
        stats = {}
        
        for doc in documents:
            is_good, reason = self._is_high_quality(doc)
            if is_good:
                clean_docs.append(doc)
            else:
                stats[reason] = stats.get(reason, 0) + 1
                
        return clean_docs, stats

# --- Test ---
if __name__ == "__main__":
    docs = [
        "Ravi ko diabetes hai aur usko dawa chahiye.", # Good
        "Hi.", # Too short
        "Lorem ipsum dolor sit amet.", # Blocklist
        "!!!!!!!!!!!!!!!!!!!!!!" # Symbols
    ]
    filterer = GopherQualityFilter()
    clean, stats = filterer.filter_documents(docs)
    print(f"Clean: {len(clean)}, Stats: {stats}")