import cv2
from PIL import Image
import os

from ultralytics import YOLO

model = YOLO("./projects/survive_depi/runs/detect/train4/weights/best.pt")

im1 = cv2.imread("./projects/survive_depi/molokheya.webp")
results = model.predict(source=im1, save=True, save_txt=True)

results = model.predict(source=[im1])