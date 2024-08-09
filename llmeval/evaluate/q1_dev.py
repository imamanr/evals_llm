import base64
from PIL import Image
import os, io
import json
from .base import dataset

class q1_dev_logs(dataset):
    def __init__(self, data_dir) -> None:
        self.data_dir = data_dir
        with open(os.path.join(data_dir, 'text.json'), 'r') as f:
            self.text_json = json.load(f)
        self.system_prompt = ['none']
        self.labels = []
    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir + "images/", f'100000{index}00.jpg')
        with open(img_path, 'rb') as f:
            img_bytes = f.read()
        img = Image.open(io.BytesIO(img_bytes))
        text = self.text_json['responses'][index-1]['response']
        return img, text
    
    def __len__(self):
        return len(self.text_json)
