from captcha.image import ImageCaptcha
import random
import string
import os
import csv
import pandas as pd

# config
DATASET_DIR = "Dataset_test/captchas"
LABELS = "Dataset_test/labels.csv"
NUM_IMAGES = 1000
CHARS = string.ascii_letters + string.digits
CAPTCHA_LEN_LOWER_LIMIT = 5
CAPTCHA_LEN_UPPER_LIMIT = 7
directories = [["train",0.8],["test",0.1],["val",0.1]]

os.makedirs(DATASET_DIR, exist_ok=True)
image = ImageCaptcha(width=160, height=60)


with open(LABELS,mode="w",newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename","label"])
    OUTPUT_DIR = os.path.join(DATASET_DIR,directories[0][0])
    os.makedirs(OUTPUT_DIR,exist_ok=True)
    for i in range(NUM_IMAGES):
        if i%(NUM_IMAGES/100) ==0:
            print(f"{i} images made")
        if i>(0.8*NUM_IMAGES-1) and i<(0.9*NUM_IMAGES):
            OUTPUT_DIR = os.path.join(DATASET_DIR,directories[1][0])
            os.makedirs(OUTPUT_DIR,exist_ok=True)
        elif i>(0.9*NUM_IMAGES-1):

            OUTPUT_DIR = os.path.join(DATASET_DIR,directories[2][0])
            os.makedirs(OUTPUT_DIR,exist_ok=True)
        text = ''.join(random.choices(CHARS, k=random.randint(CAPTCHA_LEN_LOWER_LIMIT,CAPTCHA_LEN_UPPER_LIMIT)))
        filename = f"{text}_{i}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        image.write(text, filepath)
        writer.writerow([filename,text])

print("Data Generated!")


df = pd.read_csv(LABELS)

n = len(df)
train_end = int(n * directories[0][1])
val_end = train_end + int(n * directories[2][1])

# Split datasets
df_train = df.iloc[:train_end]
df_val = df.iloc[train_end:val_end]
df_test = df.iloc[val_end:]

# Save
df_train.to_csv(os.path.join(DATASET_DIR,"train/labels.csv"), index=False)
df_val.to_csv(os.path.join(DATASET_DIR,"val/labels.csv"), index=False)
df_test.to_csv(os.path.join(DATASET_DIR,"test/labels.csv"), index=False)
    
print("Labels Generated")


