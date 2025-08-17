import glob
import cv2
import pandas as pd
import torch
import os
from src.config import cfg 
from dataclasses import dataclass

@dataclass
class CaptchaDataset(torch.utils.data.Dataset):
    def __init__(self,folder:str):
        self.data_root = cfg.data_root        
        df = pd.read_csv(f"{self.data_root}/{folder}/labels.csv")
        self.data = []
        for _,row in df.iterrows():
            filename = row['filename']
            label = row['label']
            img_path = f"{self.data_root}/{folder}/{row['filename']}"
            
            # Check if file actually exists
            if os.path.exists(img_path):
                self.data.append((img_path,label,folder))
            else:
                print(f"Warning: Image file not found: {img_path}")
        
        print(f"Loaded {len(self.data)} valid images from {folder}")
        self.img_dim = (cfg.W_max, cfg.H)  # cv2.resize expects (width, height)

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        img_path, label_string,folder = self.data[idx]
        
        # Load image with error checking
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE if cfg.grayscale else cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        img = cv2.resize(img, self.img_dim)
        img_tensor = torch.from_numpy(img).float()/255.0  # Normalize to [0,1]
        img_tensor = img_tensor.unsqueeze(0)  # Add channel dimension
        return img_tensor, label_string, img_path
            



        