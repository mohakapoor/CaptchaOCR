"""
Simple CAPTCHA Generation Utility
Generates individual CAPTCHA images using enhanced rendering
"""

import random
import string
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import cv2
import io

# Configuration - match your training setup
IMG_WIDTH = 256
IMG_HEIGHT = 60
GRAYSCALE = True
CHARS = string.ascii_letters + string.digits
CAPTCHA_LEN_LOWER_LIMIT = 5
CAPTCHA_LEN_UPPER_LIMIT = 7

def rand_color(lo=0, hi=255):
    """Generate random RGB color."""
    return tuple(random.randint(lo, hi) for _ in range(3))

def gradient_bg(w, h):
    """Create gradient background."""
    top = rand_color(200, 255)
    bot = rand_color(200, 255)
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        t = y / max(1, h - 1)
        arr[y, :, :] = (np.array(top) * (1 - t) + np.array(bot) * t).astype(np.uint8)
    return Image.fromarray(arr)

def add_interference(img, line_range=(0, 3), dot_range=(10, 80)):
    """Add interference patterns (lines and dots)."""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for _ in range(random.randint(*line_range)):
        x1, y1 = random.randint(0, w-1), random.randint(0, h-1)
        x2, y2 = random.randint(0, w-1), random.randint(0, h-1)
        draw.line((x1, y1, x2, y2), fill=rand_color(50, 180), width=random.randint(1, 2))
    for _ in range(random.randint(*dot_range)):
        x, y = random.randint(0, w-1), random.randint(0, h-1)
        r = random.choice([0, 1])
        draw.ellipse((x-r, y-r, x+r, y+r), fill=rand_color(0, 200))
    return img

def perspective_warp(img, max_ratio=0.03):
    """Apply perspective warping."""
    if max_ratio <= 0:
        return img
    w, h = img.size
    dx = int(w * max_ratio)
    dy = int(h * max_ratio * 0.7)
    src = np.float32([[0,0],[w,0],[w,h],[0,h]])
    dst = np.float32([[random.randint(0,dx), random.randint(0,dy)],
                      [w-random.randint(0,dx), random.randint(0,dy)],
                      [w-random.randint(0,dx), h-random.randint(0,dy)],
                      [random.randint(0,dx), h-random.randint(0,dy)]])
    M = cv2.getPerspectiveTransform(src, dst)
    arr = np.array(img.convert("RGB"))[:, :, ::-1]  # to BGR
    out = cv2.warpPerspective(arr, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(out[:, :, ::-1])  # back to RGB

def jpeg_recompress(img, qmin=70, qmax=95):
    """Recompress image to simulate JPEG artifacts."""
    q = random.randint(qmin, qmax)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def add_noise_and_blur(img, noise_sigma=(0.0, 6.0), blur_sigma=(0.0, 0.8), motion_prob=0.1):
    """Add noise and blur effects."""
    # Gaussian noise
    s = random.uniform(*noise_sigma)
    if s > 0.05:
        arr = np.array(img).astype(np.float32)
        arr += np.random.normal(0, s, arr.shape).astype(np.float32)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    
    # Blur
    if random.random() < motion_prob:
        # Simple directional blur
        ksize = random.choice([3,5])
        kernel = Image.new("L", (ksize, ksize), 0)
        draw = ImageDraw.Draw(kernel)
        draw.line((0, ksize//2, ksize-1, ksize//2), fill=255, width=1)
        kernel = kernel.rotate(random.uniform(0, 180), resample=Image.BILINEAR)
        kernel = np.array(kernel, dtype=np.float32)
        kernel /= max(1, kernel.sum())
        arr = np.array(img)
        arr = cv2.filter2D(arr, -1, kernel)
        img = Image.fromarray(arr)
    else:
        sigma = random.uniform(*blur_sigma)
        if sigma > 0.05:
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    
    return img

def generate_captcha(text=None, width=IMG_WIDTH, height=IMG_HEIGHT, save_path=None):
    """
    Generate a single enhanced CAPTCHA image.
    
    Args:
        text (str, optional): Text to render. If None, generates random text.
        width (int): Image width
        height (int): Image height
        save_path (str, optional): Path to save the image. If None, returns PIL Image.
    
    Returns:
        PIL Image if save_path is None, otherwise saves and returns the path
    """
    # Generate random text if none provided
    if text is None:
        text = ''.join(random.choices(CHARS, k=random.randint(CAPTCHA_LEN_LOWER_LIMIT, CAPTCHA_LEN_UPPER_LIMIT)))
    
    # Randomize basic style
    bg_choice = random.choice(["solid", "gradient"])
    fg_color = rand_color(0, 80)
    
    if bg_choice == "solid":
        bg_color = rand_color(210, 255)
        bg = Image.new("RGB", (width, height), color=bg_color)
    else:
        bg = gradient_bg(width, height)

    # Adjust font sizes for larger dimensions
    font_sizes = [int(height * 0.7), int(height * 0.75), int(height * 0.8), int(height * 0.85)]
    font_size = random.choice(font_sizes)
    
    # Use ImageCaptcha for base text rendering
    from captcha.image import ImageCaptcha
    image = ImageCaptcha(width=width, height=height, fonts=None, font_sizes=[font_size])
    
    # Draw base image
    base = Image.frombytes('RGB', (width, height), image.generate_image(text).tobytes())

    # Apply enhancements
    angle = random.uniform(-6, 6)
    base = base.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=bg.getpixel((0,0)))

    # Perspective warp (very light)
    if random.random() < 0.6:
        base = perspective_warp(base, max_ratio=0.025)

    # Add interference
    base = add_interference(base, line_range=(0, 3), dot_range=(10, 60))

    # Noise + blur + JPEG recompression
    base = add_noise_and_blur(base, noise_sigma=(0.0, 5.0), blur_sigma=(0.0, 0.7), motion_prob=0.12)
    base = jpeg_recompress(base, qmin=72, qmax=92)

    # Optional low contrast
    if random.random() < 0.2:
        base = base.point(lambda p: int(p*0.95 + 6))

    # Convert to grayscale if specified
    if GRAYSCALE:
        base = base.convert('L')
    
    # Save or return
    if save_path:
        base.save(save_path)
        return save_path
    else:
        return base


if __name__ == "__main__":
    # Example usage
    print("Generating sample CAPTCHAs...")
    
    # Generate with specific text
    img1 = generate_captcha("HELLO", save_path="sample_HELLO.png")
    print(f"Generated: sample_HELLO.png")
    
    print("Done! Check the generated images.")
