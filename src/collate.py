from typing import List,Tuple
import torch
from src.config import cfg
from src.vocab import encode_text

def ctc_collate(batch: List[Tuple[torch.Tensor, str, str]]):
    """
    batch: list of (image_tensor [C,H,W_max], label_str, rel_path)
    returns:
      images: [B,C,H,W_max]
      targets_flat: [sum(len(label_i))]
      target_lengths: [B]
      input_lengths: [B]  (all equal if same W_max/stride)
      rel_paths: list[str]
    """

    images = torch.stack([item[0] for item in batch],dim =0)

    labels = [item[1] for item in batch]
    encoded = [torch.tensor(encode_text(t),dtype = torch.long) for t in labels]
    target_lengths = torch.tensor([len(t) for t in encoded],dtype = torch.long)
    if len(encoded) > 0:
        targets_flat = torch.cat(encoded,dim = 0)
    else:
        targets_flat = torch.empty(0,dtype = torch.long)

    
    B, C, H, W = images.shape
    
    input_len = W // cfg.total_stride
    input_lengths = torch.full((B,), input_len, dtype=torch.long)

    rel_paths = [item[2] for item in batch]
    return images, targets_flat, target_lengths, input_lengths, rel_paths    