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
- **Dataset Generation**: Synthetic CAPTCHA creation with train/val/test splits
- **Configuration**: Centralized config with image dimensions and training parameters
- **Vocabulary System**: Character encoding/decoding with CTC blank token support
- **CTC Collate Function**: Proper batching for variable-length sequences
- **CTC Decoding**: Greedy decode for inference

### 🔧 In Progress / Next Steps
- **PyTorch Dataset Class**: Image loading and preprocessing
- **CRNN Model**: CNN encoder + BiLSTM + linear output
- **Training Loop**: Complete training pipeline with validation
- **Metrics**: CER (Character Error Rate) and exact match accuracy
- **Inference Pipeline**: Model loading and prediction

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
│   ├── vocab.py            # Character vocabulary and CTC encoding
│   ├── data.py             # Dataset generation script
│   ├── collate.py          # CTC batching function
│   └── [model files]       # Coming soon...
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

### 2. Generate Test Dataset
```bash
cd src
python data.py
```
This creates 1,000 synthetic CAPTCHAs in `Dataset_test/captchas/` with proper train/val/test splits.

### 3. Configuration
Edit `src/config.py` to adjust:
- Image dimensions (H=48, W_max=224)
- Batch sizes (32 for local GTX 1650, 128 for Colab T4)
- Training parameters

## 🎮 Usage

### Local Development (GTX 1650)
- Use `Dataset_test` (1k images)
- Batch size: 32-48
- Good for rapid iteration and testing

### Colab Training (Tesla T4)
- Use `Dataset` (100k images)
- Batch size: 128
- Expected training time: 2-4 hours for 40 epochs

## 🔬 Technical Details

### Model Architecture
- **CNN Encoder**: Reduces image to sequence representation
- **BiLSTM**: Processes sequential features
- **Linear Output**: Maps to vocabulary size (including blank token)

### CTC Training
- **Input**: Images resized to 48×224
- **Output**: Character sequences (a-z, A-Z, 0-9)
- **Loss**: CTCLoss with blank=0
- **Decoding**: Greedy CTC decode

### Data Format
- **Images**: Grayscale, normalized tensors
- **Labels**: CSV with filename and text label
- **Batching**: Variable-length sequences handled by custom collate

## 📊 Performance Expectations

### GTX 1650 (4GB VRAM)
- Training time: 3-8 hours for 100k×40 epochs
- Batch size: 32-48
- Memory efficient with H=48

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
