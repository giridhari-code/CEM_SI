"""
CEM Data Engineering: Deduplication.
Uses MinHash LSH to remove near-duplicate documents.
"""

from datasketch import MinHash, MinHashLSH
from typing import List

class NearDuplicator:
    def __init__(self, threshold: float = 0.8, num_perm: int = 128):
        # threshold=0.8 means 80% similarity will be flagged as duplicate
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.num_perm = num_perm

    def _get_minhash(self, text: str) -> MinHash:
        """Creates MinHash signature for a document."""
        m = MinHash(num_perm=self.num_perm)
        # Simple word-level shingling
        for word in text.split():
            m.update(word.encode('utf8'))
        return m

    def deduplicate(self, documents: List[str]) -> List[str]:
        """Filters out duplicates and keeps the first unique document."""
        unique_docs = []
        
        for idx, doc in enumerate(documents):
            m = self._get_minhash(doc)
            # Check if similar doc already exists
            result = self.lsh.query(m)
            
            if not result:
                # New unique document
                self.lsh.insert(str(idx), m)
                unique_docs.append(doc)
            # else: Duplicate found, skip it
            
        return unique_docs

# --- Test ---
if __name__ == "__main__":
    docs = [
        "Ravi ko diabetes hai aur usko metformin leni chahiye.",
        "Ravi ko diabetes hai aur usko metformin leni chahiye.", # Exact Dup
        "Ravi ko diabetes hai aur usko metformin zaroor leni chahiye.", # Near Dup
        "Aaj mausam accha hai aur hum park jayenge."
    ]
    deduper = NearDuplicator(threshold=0.7)
    unique = deduper.deduplicate(docs)
    print(f"Original: {len(docs)}, Unique: {len(unique)}")