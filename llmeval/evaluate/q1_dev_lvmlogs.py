import base64
from PIL import Image
import os, io
import json
import re
from llmeval.smp.vlm import is_valid_base64_image
from pathlib import Path
from .access_s3 import download_file_from_s3
from .base import dataset

class q1_dev_lvmlogs(dataset):

    def __init__(self, data_dir) -> None:
        self.data_dir = os.path.join(data_dir, 'lvm-logs-server.lvm-log.json')
        if Path(self.data_dir).is_file():
            with open(self.data_dir, 'r') as f:
                self.text_json = json.load(f)
        else:
            bucket_name = 'dummy-imama'
            filename = 'lvm-logs-server.lvm-log.json' 
            local_file_name = 'lvm-logs-server.lvm-log.json'
            download_file_from_s3(bucket_name, filename, local_file_name)
            with open(local_file_name, 'r') as f:
                self.text_json = json.load(f)

        self.system_prompt = ['none']
        self.labels = []
        self.image_present = True
        self.labels_gt = False


    def __getitem__(self, index):

        data = self.text_json[index]
        image = data['completion']['request']['imagesBase64OrUrl'][0]
        if is_valid_base64_image(image_string=image):
            image = image
        else:
            image = []
        text = data['completion']['request']['query']
        
        return text, image, []
    
    def __len__(self):
        return len(self.text_json)
