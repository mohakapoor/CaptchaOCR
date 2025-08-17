import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.config import cfg
from src.collate import ctc_collate
from src.captcha_dataset import CaptchaDataset
from src.vocab import vocab_size, ctc_greedy_decode
from src.model_crnn import CRNN


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_ch = 1 if cfg.grayscale else 3

    print("Creating datasets...")
    train_ds = CaptchaDataset("train")
    val_ds = CaptchaDataset("val")
    
    print(f"Training dataset size: {len(train_ds)}")
    print(f"Validation dataset size: {len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, 
                          num_workers=cfg.num_workers, pin_memory=True, 
                          drop_last=True, collate_fn=ctc_collate)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, 
                        num_workers=cfg.num_workers, pin_memory=True, 
                        drop_last=True, collate_fn=ctc_collate)

    # # Test training data
    # print("\nTesting training data...")
    # for batch in train_dl:
    #     images, targets_flat, target_lengths, input_lengths, paths = batch
    #     print(f"Training batch shape: {images.shape}")
    #     print(f"Sample labels: {targets_flat[:10]}")
    #     break

    # # Test validation data
    # print("\nTesting validation data...")
    # try:
    #     for batch in val_dl:
    #         images, targets_flat, target_lengths, input_lengths, paths = batch
    #         print(f"Validation batch shape: {images.shape}")
    #         print(f"Sample labels: {targets_flat[:10]}")
    #         break
    # except Exception as e:
    #     print(f"Error in validation data: {e}")
    #     print("This suggests there are issues with some validation images")

    model = CRNN(vocab_size=vocab_size()).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.amp and device.type == "cuda")

    model.train()
    steps = 200
    it = iter(train_dl)
    for step in range(1,steps+1):
        try:
            images, targets_flat, target_lengths, input_lengths, paths = next(it)
        except StopIteration:
            it = iter(train_dl)
            images, targets_flat, target_lengths, input_lengths, paths = next(it)
        
        images = images.to(device)
        targets_flat = targets_flat.to(device)
        target_lengths = target_lengths.to(device)
        input_lengths = input_lengths.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=scaler.is_enabled()):
            logits = model(images)
            log_probs = logits.log_softmax(dim=-1)
            loss = criterion(log_probs,targets_flat,input_lengths,target_lengths)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


        if step % 20 == 0:
            print(f"step {step}/{steps} - loss {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        images, targets_flat, target_lengths, input_lengths, paths = next(iter(val_dl))
        images = images.to(device)
        logits = model(images)
        preds = ctc_greedy_decode(logits)

    print("Sanity check complete")


if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    main()