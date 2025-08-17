# CAPTCHA OCR Project

A PyTorch-based CAPTCHA recognition system using synthetic data generation and CTC-based sequence modeling.

## 🎯 Project Overview

This project implements an end-to-end CAPTCHA OCR system that can recognize text in CAPTCHA images. It uses:
- **Synthetic CAPTCHA generation** for training data
- **CRNN (CNN + RNN) architecture** for sequence recognition
- **CTC (Connectionist Temporal Classification)** loss for training
- **PyTorch** with CUDA support for GPU acceleration

## 🏗️ Current Status

### ✅ Completed Components
- **Dataset Generation**: Synthetic CAPTCHA creation with train/val/test splits (8k train, 1k val)
- **Configuration**: Centralized config with image dimensions and training parameters
- **Vocabulary System**: Character encoding/decoding with CTC blank token support (63 classes)
- **CTC Collate Function**: Proper batching for variable-length sequences
- **CTC Decoding**: Greedy decode for inference
- **PyTorch Dataset Class**: Image loading and preprocessing with proper cv2 resizing
- **CRNN Model**: CNN encoder + BiLSTM + LayerNorm + linear output (working!)
- **Training Loop**: Complete epoch-based training pipeline with validation
- **Metrics & Plotting**: Training/validation loss tracking with beautiful visualizations
- **Debugging Tools**: Comprehensive logging of logits, predictions, and model health

### ✅ What's Working
- **Training Pipeline**: Stable training loop with excellent loss convergence
- **Model Architecture**: CRNN produces correct output shapes (64×batch×63) with H=60, W=256
- **Data Loading**: Proper image preprocessing and CTC batching
- **Full CAPTCHA Recognition**: Model now recognizes complete CAPTCHA sequences
- **Inference Pipeline**: Complete inference script with visualization and accuracy metrics
- **Early Stopping**: Enhanced early stopping prevents overfitting automatically
- **High Accuracy**: 75-100% overall accuracy, 25/26+ character accuracy (96%+)

### 🎯 Training Status
- **Current**: Epoch 8, excellent convergence achieved
- **Best Model**: Validation loss 0.1782, early stopping working perfectly
- **Performance**: 75-100% accuracy on fresh CAPTCHAs (varies by run)

### 📊 Training Results
![Training Losses](Metrics/training_losses.png)
![Loss Comparison](Metrics/loss_comparison.png)

**Key Insights:**
- **Rapid convergence**: Loss dropped from 21→0.1 in first 7 epochs
- **No overfitting**: Enhanced early stopping prevents overfitting
- **Stable training**: Val/Train ratio stays healthy throughout training

### 🔍 Inference Results
![Inference Results](Metrics/inference_results_readme.png)

**Model Performance:**
- **Visual predictions**: Shows actual CAPTCHA images with predicted text
- **High accuracy**: 75-100% overall accuracy on fresh CAPTCHAs
- **Character-level precision**: 96%+ character accuracy (25/26+ correct)

## 📁 Project Structure

```
CaptchaDetect/
├── Dataset/                 # Full dataset (100k images) - for Colab training
├── Dataset_test/           # Test dataset (1k images) - for local development
│   └── captchas/
│       ├── train/          # 80% of data
│       ├── val/            # 10% of data
│       └── test/           # 10% of data
├── src/
│   ├── config.py           # Configuration and hyperparameters
│   ├── vocab.py            # Character vocabulary and CTC encoding/decoding
│   ├── data.py             # Dataset generation script
│   ├── collate.py          # CTC batching function
│   ├── captcha_dataset.py  # PyTorch Dataset class
│   ├── model_crnn.py       # CRNN model architecture
│   └── plotting.py         # Training metrics and visualization
├── train.py                # Main training script (✅ WORKING!)
├── Metrics/                # Training plots and logs (auto-generated)
├── .gitignore              # Ignores dataset contents, keeps structure
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Install PyTorch with CUDA support (adjust version as needed)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install other dependencies
pip install captcha pandas pillow
```

### 2. Generate Training Dataset
```bash
cd src
python data.py
```
This creates 10,000 synthetic CAPTCHAs in `Dataset_test/captchas/` with proper train/val/test splits.

### 3. Start Training
```bash
python train.py
```
This starts the full training pipeline with automatic metrics generation.

### 4. Monitor Progress
Training will show:
- Real-time loss and prediction samples
- Automatic plot generation in `Metrics/` folder
- Comprehensive training logs and summaries

## 🎮 Usage

### Training
```bash
python train.py
```
- **Automatic early stopping** prevents overfitting
- **Real-time metrics** and sample predictions
- **Checkpoint saving** for best model

### Inference
```bash
python inference.py
```
- **Loads best trained model** automatically
- **Generates test CAPTCHAs** for evaluation
- **Shows both overall and character accuracy**
- **Creates visualization plots** in Metrics folder

### Local Development (GTX 1650)
- Use `Dataset_test` (1k images)
- Batch size: 32
- Good for rapid iteration and testing

### Colab Training (Tesla T4)
- Use `Dataset` (100k images)
- Batch size: 128
- Expected training time: 2-4 hours for 40 epochs

## 🔬 Technical Details

### Model Architecture (CRNN)

The model uses a **CNN + RNN + CTC** architecture specifically designed for sequence recognition:

#### **CNN Encoder (SmallCNN)**
```
Input: [B, 1, 60, 256] → Output: [64, B, 128]
```
- **Conv1 Block**: 3×3 conv → BatchNorm → ReLU → MaxPool(2×2)
  - Channels: 1 → 64
  - Spatial: 60×256 → 30×128
- **Conv2 Block**: 3×3 conv → BatchNorm → ReLU → MaxPool(1×2)  
  - Channels: 64 → 128
  - Spatial: 30×128 → 30×64
- **Residual Block**: 3×3 conv → BatchNorm → ReLU → 3×3 conv → BatchNorm + Skip Connection
  - Maintains 128 channels and 30×64 spatial dimensions
- **Height Pooling**: AdaptiveAvgPool2d(1, None) → squeeze(2)
  - Spatial: 30×64 → 1×64 → 64 timesteps
  - Final: [64, B, 128] where T=64, B=batch_size, C=128

#### **RNN Decoder (BiLSTM)**
```
Input: [64, B, 128] → Output: [64, B, 640]
```
- **Architecture**: 2-layer bidirectional LSTM
- **Hidden Size**: 320 per direction (total 640)
- **Dropout**: 0.05 between layers
- **Output**: [T, B, 2×hidden] = [64, B, 640]

#### **Output Layer**
```
Input: [64, B, 640] → Output: [64, B, 63]
```
- **LayerNorm**: Stabilizes 640-dimensional features
- **Linear**: Maps to vocabulary size (62 chars + 1 blank token)
- **Final Shape**: [T=64, B=batch_size, V=63]

#### **Key Design Features**
- **Total Stride**: 4 (256 → 64 timesteps)
- **Height Compression**: 60 → 1 (via pooling)
- **Residual Connections**: Prevents gradient vanishing
- **Bidirectional LSTM**: Captures context from both directions
- **LayerNorm**: Training stability before final classification

### Training Optimizations
- **AdamW Optimizer**: lr=3e-4, weight_decay=1e-4
- **Gradient Clipping**: max_norm=1.0 prevents exploding gradients
- **Weight Initialization**: Small uniform weights (-1e-3, 1e-3) for stability
- **Numeric Stability**: AMP disabled during initial training for stability

### CTC Training
- **Input**: Images resized to 60×256 (height×width)
- **Output**: Character sequences (a-z, A-Z, 0-9)
- **Loss**: CTCLoss with blank=0, zero_infinity=True
- **Decoding**: Greedy CTC decode with duplicate removal

### Data Pipeline
- **Images**: Grayscale, normalized to [0,1], proper cv2 resizing
- **Labels**: CSV with filename and text label
- **Batching**: Variable-length sequences with custom CTC collate function
- **Debugging**: Real-time monitoring of logits, blank probability, predictions

## 📊 Performance Expectations

### GTX 1650 (4GB VRAM)
- Training time: 3-8 hours for 100k×40 epochs
- Batch size: 32
- Memory efficient with H=60, W=256

### Tesla T4 (16GB VRAM)
- Training time: 2-4 hours for 100k×40 epochs
- Batch size: 128
- Mixed precision (AMP) enabled

## 🛠️ Development Workflow

1. **Implement Dataset class** - Load and preprocess images
2. **Build CRNN model** - CNN + BiLSTM architecture
3. **Create training loop** - With validation and checkpoints
4. **Add metrics** - CER and accuracy tracking
5. **Test on small dataset** - Verify everything works
6. **Scale to full dataset** - Train on Colab

## 🤝 Contributing

This is a learning project! Feel free to:
- Ask questions about implementation details
- Experiment with different architectures
- Improve the data generation or training pipeline

## 📚 Resources

- [CTC Paper](https://www.cs.toronto.edu/~graves/icml_2006.pdf)
- [CRNN Architecture](https://arxiv.org/abs/1507.05717)
- [PyTorch CTC Tutorial](https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html)

## 📝 License

This project is for educational purposes. Feel free to use and modify as needed.

---

**Happy coding! 🚀**
