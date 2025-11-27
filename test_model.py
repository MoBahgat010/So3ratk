import cv2
from PIL import Image
import os

from ultralytics import YOLO

model_path = "./projects/survive_depi/runs/detect/train4/weights/best.pt"

model = YOLO(model_path)

img_path = "./projects/survive_depi/molokheya.webp"

im1 = cv2.imread(img_path)
results = model.predict(source=im1, save=True, save_txt=True)

# from list of PIL/ndarray
results = model.predict(source=[im1])