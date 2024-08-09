import base64
from PIL import Image
import os, io
import json
import csv
from pathlib import Path
import re
from .base import dataset

class intention_logs(dataset):
    def __init__(self, data_dir) -> None:
        self.data_dir = data_dir
        self.image_present = False
        self.labels_gt = True

        file_names = os.listdir(self.data_dir)
        self._ids = list(set([re.match(r"(\d+)-.*", file_name).group(1) for file_name in file_names]))
        prompt_file = Path(self.data_dir) / f"{self._ids[0]}-prompt.json"
        with open(prompt_file, "r") as f:
            prompt_string = f.read()
            full_prompt = json.loads(prompt_string)[1:]
        self.system_prompt = full_prompt[0]['content']

    def __getitem__(self, index):
        prompt_file = Path(self.data_dir) / f"{self._ids[index]}-prompt.json"
        label_file = Path(self.data_dir) / f"{self._ids[index]}-response.txt"
        full_prompt =[]
        labels = []
        with open(prompt_file, "r") as f:
                prompt_string = f.read()
                full_prompt = json.loads(prompt_string)[1:]
        with open(label_file, "r") as f:
                labels = f.read()
        return full_prompt, [], labels
    
    def __len__(self):
        return len(self._ids)//2

