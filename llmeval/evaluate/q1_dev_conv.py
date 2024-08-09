import base64
from PIL import Image
import os, io
import json
from .base import dataset
from pathlib import Path

class q1_dev_conv(dataset):
    def __init__(self, data_dir) -> None:

        with open(os.path.join(data_dir, 'text_conv.json'), 'r') as f:
            self.text_json = json.load(f)
        # if (self.prompt_file):
        #     self.system_prompt = json.load(self.prompt_file)
        # else:
            self.system_prompt = ["none"]

        self.labels = ["here are labels"]
        self.image_present = False
        self.labels_gt = False

    def __getitem__(self, index):
        text = self.text_json['responses'][index]['response']
        label = self.text_json['responses'][index]['GT']
        return text, [], label
    
    def __len__(self):
        return len(self.text_json)
