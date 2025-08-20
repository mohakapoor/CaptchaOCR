import torch
import cv2
import os
import random
import numpy as np
from src.config import cfg
from src.model_crnn import CRNN
from src.vocab import ctc_greedy_decode, vocab_size
from src.plotting import TrainingMetrics
from src.generateCaptcha import generate_captcha

def load_model(checkpoint_path="checkpoints/best_model.pth"):
    """Load the trained model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Detect available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint to the detected device
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"   Best validation loss: {checkpoint['best_val_loss']:.4f}")
    print(f"   Loading to device: {device}")
    
    # Create model and load weights
    model = CRNN(vocab_size=vocab_size(), hidden=320, dropout=0.05)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model

def preprocess_image(image_path, target_size=(cfg.W_max, cfg.H)):
    """Preprocess image for inference (same as training)."""
    # Load image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Read and preprocess
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE if cfg.grayscale else cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Resize to target dimensions
    img = cv2.resize(img, target_size)
    
    # Convert to tensor and normalize
    img_tensor = torch.from_numpy(img).float() / 255.0
    
    # Add batch and channel dimensions
    if cfg.grayscale:
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    else:
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    
    return img_tensor

def predict_captcha(model, image_tensor, device):
    """Run inference on a single image."""
    with torch.no_grad():
        # Move to device
        image_tensor = image_tensor.to(device)
        
        # Forward pass
        logits = model(image_tensor)
        
        # Decode prediction
        prediction = ctc_greedy_decode(logits)
        
        return prediction[0] if prediction else ""

def generate_test_captcha(text, filename, width=256, height=60):
    """Generate a test CAPTCHA image using enhanced generation."""
    # Use the enhanced CAPTCHA generation from generateCaptcha.py
    img = generate_captcha(text, width=width, height=height)
    
    # Ensure results directory exists
    os.makedirs(cfg.RESULT_DIR, exist_ok=True)
    
    filepath = os.path.join(cfg.RESULT_DIR, filename)
    img.save(filepath)
    print(f"Generated enhanced test CAPTCHA: {filename}")
    return filepath

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(cfg.RESULT_DIR, exist_ok=True)
    
    try:
        # Load trained model
        print("Loading trained model...")
        model = load_model()
        model = model.to(device)
        print("Model loaded successfully!")
        
        # Generate test CAPTCHAs
        print("\nGenerating enhanced test CAPTCHAs...")
        test_cases = []
        
        for i in range(4):
            # Generate random text
            text = ''.join(random.choices(cfg.chars, k=random.randint(cfg.CAPTCHA_LEN_LOWER_LIMIT, cfg.CAPTCHA_LEN_UPPER_LIMIT)))
            filename = f"{text}_{i}.png"
            
            # Generate enhanced image
            image_path = generate_test_captcha(text, filename, width=cfg.W_max, height=cfg.H)
            test_cases.append((text, image_path, ""))  # Add empty prediction slot
        
        # Run inference
        print("\nRunning inference...")
        print("-" * 60)
        print(f"{'Target':<15} {'Prediction':<15} {'Correct':<10} {'Image':<20}")
        print("-" * 60)
        
        correct_count = 0
        for i, (target_text, image_path, _) in enumerate(test_cases):
            try:
                # Preprocess image
                image_tensor = preprocess_image(image_path)
                
                # Run prediction
                prediction = predict_captcha(model, image_tensor, device)
                
                # Store prediction in test_cases
                test_cases[i] = (target_text, image_path, prediction)
                
                # Check if correct
                is_correct = prediction == target_text
                if is_correct:
                    correct_count += 1
                
                # Display result
                status = "CORRECT" if is_correct else "WRONG"
                print(f"{target_text:<15} {prediction:<15} {status:<10} {os.path.basename(image_path):<20}")
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        
        # Summary
        print("-" * 60)
        accuracy = (correct_count / len(test_cases)) * 100
        print(f"Overall Accuracy: {correct_count}/{len(test_cases)} ({accuracy:.1f}%)")
        
        # Calculate individual character accuracy
        total_chars = 0
        correct_chars = 0
        for target_text, _, prediction in test_cases:
            total_chars += len(target_text)
            # Count correct characters (position by position)
            min_len = min(len(target_text), len(prediction))
            for i in range(min_len):
                if target_text[i] == prediction[i]:
                    correct_chars += 1
        
        char_accuracy = (correct_chars / total_chars) * 100 if total_chars > 0 else 0
        print(f"Character Accuracy: {correct_chars}/{total_chars} ({char_accuracy:.1f}%)")
        
        if accuracy >= 80:
            print("Excellent performance!")
        elif accuracy >= 60:
            print("Good performance!")
        else:
            print("Room for improvement...")
        
        # Create and save results plot
        print("\nGenerating results visualization...")
        try:
            metrics = TrainingMetrics()
            image_paths = [case[1] for case in test_cases]
            predictions = [case[2] for case in test_cases]
            targets = [case[0] for case in test_cases]
            
            # Create results directory if it doesn't exist
            os.makedirs("Metrics", exist_ok=True)
            
            # Plot results
            metrics.plot_results(image_paths, predictions, targets)
            print("Results plot generated successfully!")
            
        except Exception as e:
            print(f"Warning: Could not generate plot: {e}")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have a trained model in checkpoints/best_model.pth")

if __name__ == "__main__":
    main()


