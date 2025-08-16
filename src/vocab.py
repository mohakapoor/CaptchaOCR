from typing import List
from src.config import cfg

itos = ["<blank>"] + list(cfg.chars)

stoi = {c: i+1 for i,c in enumerate(cfg.chars)}

def encode_text(text: str) -> List[int]:
    return [stoi[c] for c in text]

def decode_indices(indices: List[int]) -> str:
    return "".join(itos[i] for i in indices if i != 0)

def ctc_greedy_decode(logits) -> List[str]:
    """
    Greedy CTC decode for a batch.
    logits: torch.Tensor of shape [T, B, V] (before softmax or log_softmax).
    Returns: list of B decoded strings.
    """
    import torch
    pred = logits.argmax(dim=-1)
    B = pred.shape[1]
    decoded = []
    for b in range(B):
        prev = -1
        chars = []
        for t in pred[:,b].tolist():
            if t!=0 and t!= prev:
                chars.append(itos[t])
            prev = t
        decoded.append("".join(chars))
    return decoded

def vocab_size() -> int:
    return len(itos)