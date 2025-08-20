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

def ui_generate():
    text = random_text()
    filename = f"{text}_{random.randint(1000,9999)}.png"
    # Use generateCaptcha.py directly
    img = generate_captcha(text, width=cfg.W_max, height=cfg.H)
    
    # Save to results directory
    filepath = os.path.join(cfg.RESULT_DIR, filename)
    img.save(filepath)
    
    return img, text, filepath

def ui_solve(img: Image.Image, path_hint: str):
    # Prefer uploaded image
    if img is not None:
        tmp_path = os.path.join(cfg.RESULT_DIR, f"upload_{random.randint(1000,9999)}.png")
        img.save(tmp_path)
        tensor = inf.preprocess_image(tmp_path, (cfg.W_max, cfg.H))
        pred = inf.predict_captcha(MODEL, tensor, DEVICE)
        return pred
    # Otherwise, solve the last generated image
    if path_hint and os.path.exists(path_hint):
        tensor = inf.preprocess_image(path_hint, (cfg.W_max, cfg.H))
        pred = inf.predict_captcha(MODEL, tensor, DEVICE)
        return pred
    return "No image provided. Generate or upload first."

with gr.Blocks(title="CAPTCHA OCR (checkpoint)") as demo:
    gr.Markdown("## CAPTCHA OCR demo")

    with gr.Row():
        gen_btn = gr.Button("Generate CAPTCHA", variant="primary")
        gt_out = gr.Textbox(label="Ground Truth", interactive=False)

    with gr.Row():
        img_out = gr.Image(label="Generated CAPTCHA", type="pil")
        path_box = gr.Textbox(label="Internal Path", interactive=False, visible=False)

    gen_btn.click(fn=ui_generate, outputs=[img_out, gt_out, path_box])

    gr.Markdown("### Solve")
    with gr.Row():
        img_in = gr.Image(label="Upload CAPTCHA (optional)", type="pil")
        solve_btn = gr.Button("Solve")
        pred_out = gr.Textbox(label="Prediction", interactive=False)

    solve_btn.click(fn=ui_solve, inputs=[img_in, path_box], outputs=[pred_out])

if __name__ == "__main__":
    demo.launch()