"""
CEM Utils: Tokenizer.
A simple Character-Level Tokenizer (No BPE, No external vocab).
"""

from typing import List

class CharTokenizer:
    """
    Character-level tokenizer.
    Maps every character (and special tokens) to an integer ID.
    """
    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        # Special Tokens
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2 # Beginning of Sequence
        self.eos_token_id = 3 # End of Sequence

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Converts string to list of integer IDs.
        """
        # Convert char to ASCII/Unicode value, offset by 4 (reserved for special tokens)
        ids = [ord(c) % self.vocab_size for c in text]
        
        # Shift to avoid collision with special tokens
        ids = [i + 4 if i < self.vocab_size - 4 else self.unk_token_id for i in ids]
        
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
            
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Converts list of integer IDs back to string.
        """
        chars = []
        for i in ids:
            if skip_special_tokens and i in [self.pad_token_id, self.unk_token_id, self.bos_token_id, self.eos_token_id]:
                continue
            # Reverse the offset
            if i >= 4:
                chars.append(chr(i - 4))
            else:
                chars.append("?") # Unknown
        return "".join(chars)