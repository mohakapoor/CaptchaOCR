import os
import random
import gradio as gr
from PIL import Image
import torch

# Import your inference module
import inference as inf
from src.generateCaptcha import generate_captcha
from src.config import cfg  # sizes, charset, dirs

# Device and one-time model load
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = inf.load_model("checkpoints/best_model.pth").to(DEVICE).eval()

# Ensure results dir exists
os.makedirs(cfg.RESULT_DIR, exist_ok=True)

def random_text():
    L = random.randint(cfg.CAPTCHA_LEN_LOWER_LIMIT, cfg.CAPTCHA_LEN_UPPER_LIMIT)
    return "".join(random.choices(cfg.chars, k=L))

def calculate_accuracy(prediction, target):
    """Calculate character-by-character accuracy."""
    if not prediction or not target:
        return "0%"
    
    correct_chars = 0
    min_len = min(len(prediction), len(target))
    
    for i in range(min_len):
        if prediction[i] == target[i]:
            correct_chars += 1
    
    if min_len == 0:
        return "0%"
    
    accuracy = (correct_chars / min_len) * 100
    return f"{accuracy:.1f}%"

def ui_generate():
    text = random_text()
    filename = f"{text}_{random.randint(1000,9999)}.png"
    # Use generateCaptcha.py directly
    img = generate_captcha(text, width=cfg.W_max, height=cfg.H)
    
    # Save to results directory
    filepath = os.path.join(cfg.RESULT_DIR, filename)
    img.save(filepath)

    # Enable and turn Solve green now that an image exists
    solve_btn_state = gr.update(interactive=True, variant="primary")
    return img, text, filepath, solve_btn_state

def ui_solve(path_hint: str, ground_truth: str):
    if path_hint and os.path.exists(path_hint):
        tensor = inf.preprocess_image(path_hint, (cfg.W_max, cfg.H))
        pred = inf.predict_captcha(MODEL, tensor, DEVICE)
        
        # Calculate accuracy
        accuracy = calculate_accuracy(pred, ground_truth)
        
        return accuracy, pred
    return "0%", "No image generated yet. Click Generate CAPTCHA first."

with gr.Blocks(title="CAPTCHA OCR (checkpoint)") as demo:
    gr.Markdown("## CAPTCHA OCR ")

    with gr.Row():
        # Left column: Generate button + Solve button stacked vertically
        with gr.Column(scale=1):
            gen_btn = gr.Button("Generate CAPTCHA", variant="primary")
            solve_btn = gr.Button("Solve", interactive=False, variant="secondary")
        
        # Right column: Ground Truth
        gt_out = gr.Textbox(label="Ground Truth", interactive=False, text_align="center")

    with gr.Row():
        # Fixed height container for the CAPTCHA image
        with gr.Box(style={"height": "200px"}):
            img_out = gr.Image(label="Generated CAPTCHA", type="pil")
        path_box = gr.Textbox(label="Internal Path", interactive=False, visible=False)

    # Prediction row split into two columns
    with gr.Row():
        accuracy_out = gr.Textbox(label="Character Accuracy", interactive=False, text_align="center")
        pred_out = gr.Textbox(label="Prediction", interactive=False, text_align="center")

    # Generate: outputs image, ground truth, path, and enables Solve (green)
    gen_btn.click(
        fn=ui_generate,
        outputs=[img_out, gt_out, path_box, solve_btn],
    )

    # Solve: only uses the internal path (no upload option anymore)
    solve_btn.click(
        fn=ui_solve,
        inputs=[path_box, gt_out],
        outputs=[accuracy_out, pred_out],
    )

if __name__ == "__main__":
    demo.launch()