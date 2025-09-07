import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class EmotionDataset(Dataset):
    def __init__(self, img_dir, labels, transform=None, face_cascade_path=None):
        self.img_dir = img_dir
        self.labels = labels  
        self.transform = transform
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path) if face_cascade_path else None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name, label = self.labels[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        if self.face_cascade is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                image = image[y:y+h, x:x+w]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        return image, label
