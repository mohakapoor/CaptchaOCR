import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    
    # GPU detection and info
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs: {gpu_count}")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # Convert to GB
        print(f"GPU {i}: {gpu_name}")
        print(f"GPU {i} Memory: {gpu_memory:.1f} GB")
    
    # Current GPU
    current_gpu = torch.cuda.current_device()
    print(f"Current GPU: {current_gpu}")
    
    # Test GPU tensor operations
    print("\nTesting GPU operations...")
    try:
        # Create a test tensor on GPU
        test_tensor = torch.randn(1000, 1000).cuda()
        print(f"✓ Successfully created tensor on GPU: {test_tensor.shape}")
        print(f"✓ Tensor device: {test_tensor.device}")
        
        # Test basic operations
        result = torch.mm(test_tensor, test_tensor.T)
        print(f"✓ Matrix multiplication successful: {result.shape}")
        
        # Memory usage
        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        cached = torch.cuda.memory_reserved() / 1024**2     # MB
        print(f"✓ GPU Memory allocated: {allocated:.1f} MB")
        print(f"✓ GPU Memory cached: {cached:.1f} MB")
        
        # Clean up
        del test_tensor, result
        torch.cuda.empty_cache()
        print("✓ GPU memory cleaned up successfully")
        
    except Exception as e:
        print(f"✗ GPU test failed: {e}")
else:
    print("CUDA not available - PyTorch will use CPU only")