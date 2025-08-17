import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os


class TrainingMetrics:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        self.sample_predictions = []
        self.sample_targets = []
        
    def add_epoch(self, epoch, train_loss, val_loss):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
    
    def add_predictions(self, predictions, targets):
        self.sample_predictions.extend(predictions)
        self.sample_targets.extend(targets)
    
    def plot_losses(self, save_path="Metrics/training_losses.png"):
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(self.epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Loss plot saved to: {save_path}")
    
    def plot_loss_comparison(self, save_path="Metrics/loss_comparison.png"):
        plt.figure(figsize=(12, 8))
        
        # Main loss plot
        plt.subplot(2, 2, 1)
        plt.plot(self.epochs, self.train_losses, 'b-', label='Training Loss')
        plt.plot(self.epochs, self.val_losses, 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training vs Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Loss difference plot
        plt.subplot(2, 2, 2)
        loss_diff = [t - v for t, v in zip(self.train_losses, self.val_losses)]
        plt.plot(self.epochs, loss_diff, 'g-', label='Train - Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Difference')
        plt.title('Overfitting Indicator')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Loss ratio plot
        plt.subplot(2, 2, 3)
        loss_ratio = [v/t if t > 0 else 0 for t, v in zip(self.train_losses, self.val_losses)]
        plt.plot(self.epochs, loss_ratio, 'm-', label='Val/Train Loss Ratio')
        plt.xlabel('Epoch')
        plt.ylabel('Ratio')
        plt.title('Validation/Training Loss Ratio')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Loss improvement plot
        plt.subplot(2, 2, 4)
        train_improvement = [self.train_losses[0] - t for t in self.train_losses]
        val_improvement = [self.val_losses[0] - v for v in self.val_losses]
        plt.plot(self.epochs, train_improvement, 'b-', label='Training Improvement')
        plt.plot(self.epochs, val_improvement, 'r-', label='Validation Improvement')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Improvement')
        plt.title('Loss Improvement from Start')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Loss comparison plot saved to: {save_path}")
    
    def save_metrics(self, save_path="Metrics/training_metrics.txt"):
        with open(save_path, 'w') as f:
            f.write("CAPTCHA OCR Training Metrics\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total epochs: {len(self.epochs)}\n\n")
            
            f.write("Loss Summary:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Final training loss: {self.train_losses[-1]:.4f}\n")
            f.write(f"Final validation loss: {self.val_losses[-1]:.4f}\n")
            f.write(f"Best training loss: {min(self.train_losses):.4f}\n")
            f.write(f"Best validation loss: {min(self.val_losses):.4f}\n")
            f.write(f"Training loss improvement: {self.train_losses[0] - self.train_losses[-1]:.4f}\n")
            f.write(f"Validation loss improvement: {self.val_losses[0] - self.val_losses[-1]:.4f}\n\n")
            
            f.write("Sample Predictions:\n")
            f.write("-" * 20 + "\n")
            for i, (pred, target) in enumerate(zip(self.sample_predictions[:10], self.sample_targets[:10])):
                f.write(f"Sample {i+1}: Predicted='{pred}', Target='{target}'\n")
    
    def plot_results(self, image_paths, predictions, targets, save_path="Metrics/inference_results.png"):
        """
        Plot CAPTCHA images with their predictions and targets.
        
        Args:
            image_paths: List of paths to CAPTCHA images
            predictions: List of predicted texts
            targets: List of target texts
            save_path: Path to save the plot
        """
        import cv2
        
        n_images = len(image_paths)
        if n_images == 0:
            print("No images to plot!")
            return
        
        # Force 2x2 grid for 4 images
        rows, cols = 2, 2
        fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
        
        # Flatten axes for easier indexing
        axes = axes.flatten()
        
        for i, (img_path, pred, target) in enumerate(zip(image_paths, predictions, targets)):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Load and display image
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    ax.imshow(img)
                    
                    # Determine if prediction is correct
                    is_correct = pred == target
                    color = 'green' if is_correct else 'red'
                    status = 'CORRECT' if is_correct else 'WRONG'
                    
                    # Set title with prediction and target
                    title = f"Pred: {pred}\nTarget: {target}\n{status}"
                    ax.set_title(title, fontsize=10, color=color, fontweight='bold')
                    
                else:
                    ax.text(0.5, 0.5, f"Failed to load\n{os.path.basename(img_path)}", 
                           ha='center', va='center', transform=ax.transAxes, fontsize=12)
                    
            except Exception as e:
                ax.text(0.5, 0.5, f"Error loading image\n{str(e)[:30]}...", 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10, color='red')
            
            # Remove axes
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(n_images, len(axes)):
            axes[i].axis('off')
        
        # Add overall title
        fig.suptitle('CAPTCHA OCR Inference Results', fontsize=16, fontweight='bold', y=0.98)
        
        # Calculate accuracy
        correct = sum(1 for p, t in zip(predictions, targets) if p == t)
        accuracy = (correct / len(targets)) * 100
        
        # Add accuracy info
        fig.text(0.5, 0.02, f'Accuracy: {correct}/{len(targets)} ({accuracy:.1f}%)', 
                ha='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.15)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Results plot saved to: {save_path}")