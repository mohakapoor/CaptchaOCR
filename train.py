import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.config import cfg
from src.collate import ctc_collate
from src.captcha_dataset import CaptchaDataset
from src.vocab import vocab_size, ctc_greedy_decode, decode_indices, itos
from src.plotting import TrainingMetrics
from src.model_crnn import CRNN
import difflib

def cer(pred: str, tgt: str) -> float:
    """Approximate Character Error Rate using difflib."""
    sm = difflib.SequenceMatcher(a=pred, b=tgt)
    return 1 - sm.ratio()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_ch = 1 if cfg.grayscale else 3

    print("Creating datasets...")
    train_ds = CaptchaDataset("train")
    val_ds = CaptchaDataset("val")
    
    # Debug: Check vocabulary
    print(f"Vocabulary size: {vocab_size()}")
    print(f"First 10 characters: {list(cfg.chars)[:10]}")
    print(f"First 10 itos: {itos[:10]}")
    
    print(f"Training dataset size: {len(train_ds)}")
    print(f"Validation dataset size: {len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, 
                          num_workers=cfg.num_workers, pin_memory=True, 
                          drop_last=True, collate_fn=ctc_collate)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, 
                        num_workers=cfg.num_workers, pin_memory=True, 
                        drop_last=True, collate_fn=ctc_collate)

    model = CRNN(vocab_size=vocab_size()).to(device)
    
    # Initialize final layer with small weights for stability
    with torch.no_grad():
        torch.nn.init.uniform_(model.fc.weight, -1e-3, 1e-3)
        torch.nn.init.zeros_(model.fc.bias)
    
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=False)  # Disable AMP for stability

    # Epoch-based training with scheduler
    epochs = 20  # Increased for OneCycleLR
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-4, steps_per_epoch=len(train_dl), epochs=epochs
    )
    print(f"\nStarting training for {epochs} epochs...")
    
    metrics = TrainingMetrics()
    
    # Early stopping setup
    best_val_loss = float('inf')
    patience = 5  # Stop if no improvement for 5 epochs
    patience_counter = 0
    early_stop = False

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        num_batches = 0
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("Training...")
        
        for batch_idx, batch in enumerate(train_dl):
            images, targets_flat, target_lengths, input_lengths, paths = batch
            
            # CTC sanity checks (first batch of each epoch)
            if batch_idx == 0:
                assert targets_flat.numel() == target_lengths.sum().item(), "Target lengths mismatch"
                assert torch.all(target_lengths <= input_lengths), "Target longer than input"
                print(f"    Batch 0 sanity: input_lens={input_lengths[:5].tolist()}, target_lens={target_lengths[:5].tolist()}")
                print(f"    Image stats: min={images.min():.3f}, max={images.max():.3f}, mean={images.mean():.3f}")
            
            
            images = images.to(device)
            targets_flat = targets_flat.to(device)
            target_lengths = target_lengths.to(device)
            input_lengths = input_lengths.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=False):
                logits = model(images)
                log_probs = logits.log_softmax(dim=-1)
                loss = criterion(log_probs, targets_flat, input_lengths, target_lengths)
            
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()  # OneCycleLR step per batch

            total_train_loss += loss.item()
            num_batches += 1
            
            # Progress update every 50 batches
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(train_dl)} - Loss: {loss.item():.4f}")
        
        avg_train_loss = total_train_loss / num_batches
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        num_val_batches = 0
        
        print("Validating...")
        with torch.no_grad():
            for batch in val_dl:
                images, targets_flat, target_lengths, input_lengths, paths = batch
                images = images.to(device)
                targets_flat = targets_flat.to(device)
                target_lengths = target_lengths.to(device)
                input_lengths = input_lengths.to(device)
                
                logits = model(images)
                log_probs = logits.log_softmax(dim=-1)
                loss = criterion(log_probs, targets_flat, input_lengths, target_lengths)
                
                total_val_loss += loss.item()
                num_val_batches += 1
        
        avg_val_loss = total_val_loss / num_val_batches
        
        print(f"Epoch {epoch+1}/{epochs} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        metrics.add_epoch(epoch+1, avg_train_loss, avg_val_loss)
        
        # Enhanced early stopping check
        val_train_ratio = avg_val_loss / (avg_train_loss + 1e-8)  # Avoid division by zero
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            print(f"  New best validation loss: {best_val_loss:.4f}")
            print(f"  Val/Train ratio: {val_train_ratio:.3f}")
            
            # Save best model checkpoint with metadata
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_train_ratio': val_train_ratio,
                'config': {
                    'vocab_size': vocab_size(),
                    'hidden_size': 320,
                    'total_stride': cfg.total_stride,
                    'H': cfg.H,
                    'W_max': cfg.W_max
                }
            }
            torch.save(checkpoint, "checkpoints/best_model.pth")
            print(f"  Best model saved to checkpoints/best_model.pth")
            
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epochs")
            print(f"  Val/Train ratio: {val_train_ratio:.3f}")
            
        # Enhanced early stopping: Check both absolute loss and ratio
        if patience_counter >= patience or val_train_ratio > 3.0:  # Stop if ratio > 3x
            if val_train_ratio > 3.0:
                print(f"  Early stopping triggered! Val/Train ratio too high: {val_train_ratio:.3f}")
            else:
                print(f" Early stopping triggered! No improvement for {patience} epochs")
            early_stop = True
            break

        # Test some predictions
        if epoch % 2 == 0:  # Every 2 epochs
            print("Sample predictions:")
            with torch.no_grad():
                test_batch = next(iter(val_dl))
                test_images = test_batch[0][:5].to(device)  # First 5 images
                print(f"    Input image shape: {test_images.shape}")
                print(f"    Input image min/max: {test_images.min():.4f}/{test_images.max():.4f}")
                test_logits = model(test_images)
                
                # Debug: Check logits shape and values
                print(f"    Logits shape: {test_logits.shape}")
                print(f"    Expected logits shape: [W//stride, B, V] = [{cfg.W_max}//{cfg.total_stride}, 5, 63] = [{cfg.W_max//cfg.total_stride}, 5, 63]")
                print(f"    Logits min/max: {test_logits.min():.4f}/{test_logits.max():.4f}")
                
                # Check raw predictions and blank probability (from softmax)
                raw_preds = test_logits.argmax(dim=-1)
                probs = test_logits.log_softmax(-1).exp()
                avg_blank_prob = probs[..., 0].mean().item()
                print(f"    Raw predictions shape: {raw_preds.shape}")
                print(f"    Raw predictions sample: {raw_preds[:10, 0].tolist()}")
                print(f"    Avg blank prob (softmax): {avg_blank_prob:.4f}")
                print(f"    Blank probability (argmax): {(raw_preds == 0).float().mean():.4f}")
                
                test_preds = ctc_greedy_decode(test_logits)
                
                # Decode the target integers back to text strings with proper offsets
                targets_flat, target_lengths = test_batch[1], test_batch[2]
                offsets = torch.zeros(len(target_lengths), dtype=torch.long)
                offsets[1:] = torch.cumsum(target_lengths[:-1], dim=0)
                test_targets = []
                for i in range(min(5, len(target_lengths))):
                    s = offsets[i].item()
                    e = s + target_lengths[i].item()
                    indices = targets_flat[s:e].tolist()
                    test_targets.append(decode_indices(indices))
                
                # Calculate CER for this batch
                batch_cer = sum(cer(p, t) for p, t in zip(test_preds, test_targets)) / len(test_targets)
                print(f"    Val CER (approx): {batch_cer:.3f}")
                
                for i, (pred, target) in enumerate(zip(test_preds, test_targets)):
                    print(f"    {i}: Predicted='{pred}', Target='{target}'")
                
                metrics.add_predictions(test_preds, test_targets)

    if early_stop:
        print(f"\nTraining stopped early at epoch {epoch+1} due to no improvement!")
    else:
        print(f"\nTraining completed for all {epochs} epochs!")
    
    # Save final model
    final_checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'final_val_loss': avg_val_loss,
        'final_train_loss': avg_train_loss,
        'config': {
            'vocab_size': vocab_size(),
            'hidden_size': 320,
            'total_stride': cfg.total_stride,
            'H': cfg.H,
            'W_max': cfg.W_max
        }
    }
    torch.save(final_checkpoint, "checkpoints/final_model.pth")
    print(f"Final model saved to checkpoints/final_model.pth")
    
    print("\nGenerating training metrics and plots...")
    os.makedirs("Metrics", exist_ok=True)
    metrics.plot_losses()
    metrics.plot_loss_comparison()
    metrics.save_metrics()
    
    # Final validation test
    model.eval()
    with torch.no_grad():
        images, targets_flat, target_lengths, input_lengths, paths = next(iter(val_dl))
        images = images.to(device)
        logits = model(images)
        preds = ctc_greedy_decode(logits)
        
        print("\nFinal validation predictions:")
        for i, pred in enumerate(preds[:10]):
            print(f"  {i}: {pred}")


if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    main()