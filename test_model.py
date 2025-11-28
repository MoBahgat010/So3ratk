import cv2
from PIL import Image
import os

from ultralytics import YOLO

model = YOLO("./projects/survive_depi/runs/detect/train4/weights/best.pt")

# from ndarray
im1 = cv2.imread("./projects/survive_depi/molokheya.webp")
results = model.predict(source=im1, save=True, save_txt=True)  # save predictions as labels

# from list of PIL/ndarray
results = model.predict(source=[im1])