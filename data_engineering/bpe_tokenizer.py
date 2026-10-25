"""
CEM Data Engineering: BPE Tokenizer.
Trains and uses a Byte-Pair Encoding tokenizer.
"""

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from typing import List

class BPEWrapper:
    def __init__(self, vocab_size: int = 10000):
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Whitespace()
        self.trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"]
        )

    def train(self, texts: List[str], save_path: str = "bpe_tokenizer.json"):
        """Trains BPE on raw texts."""
        self.tokenizer.train_from_iterator(texts, trainer=self.trainer)
        self.tokenizer.save(save_path)
        print(f"Tokenizer saved to {save_path}")

    def load(self, path: str = "bpe_tokenizer.json"):
        self.tokenizer = Tokenizer.from_file(path)

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)

# --- Test ---
if __name__ == "__main__":
    # 1. Train (usually done once on a large corpus)
    corpus = ["Ravi ko diabetes hai", "Sita ko bukhar hai", "Mausam accha hai"] * 100
    bpe = BPEWrapper(vocab_size=100)
    bpe.train(corpus, save_path="test_bpe.json")
    
    # 2. Test
    ids = bpe.encode("Ravi ko diabetes hai")
    print(f"Encoded IDs: {ids}")
    print(f"Decoded: {bpe.decode(ids)}")