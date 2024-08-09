import boto3
import time
import subprocess
import json
from ..models.hf import HfStreamLLM
from .base import BaseMetric
from ..models.fireworks import Fireworks
from ..models.bedrock import Claude3Wrapper, Claude3Wrapper_text
from ..smp.vlm import encode_image_file_to_base64
import json

class latency(BaseMetric):
    def __init__(self, vendor, model_id) -> None:
        # Replace with your input payload
        self.vendor = vendor
        self.model_id = model_id
        self.payload = '{"input": "what is this"}'
        if (self.vendor == 'hf' and self.model_id=='HfStreamLLM'):
            self.st_llm = HfStreamLLM()
        if (self.vendor == 'fireworks'):
            self.fireworks = Fireworks()
        if self.vendor == 'bedrock' and self.model_id == 'anthropic.claude-3-opus-20240229-v1:0':
            self.bedrock_ops = Claude3Wrapper_text()
        if self.vendor == 'bedrock' and self.model_id == 'anthropic.claude-3-haiku-20240307-v1:0':
            self.bedrock_haiku = Claude3Wrapper()

    def compute(self):
        self.endpoint_name = self.model_id
        self.vendor = self.vendor
        start_time, end_time = 0, 0

        if self.vendor == 'bedrock' and self.model_id=='anthropic.claude-3-opus-20240229-v1:0':
            start_time = time.time()
            # Invoke the Bedrock model endpoint
            input = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Arrange for a delivery from the Italian restaurant"}],
                    }
                ]

            response = self.bedrock_ops.invoke_model(prompt=input, system_prompt='', base64_image_data='')
            # Record the end time
            end_time = time.time()
            # Calculate the latency
        elif self.vendor == 'bedrock' and self.model_id=='anthropic.claude-3-haiku-20240307-v1:0':
            image = encode_image_file_to_base64('assets/apple.jpg')
            start_time = time.time()
            # Invoke the Bedrock model endpoint
            response = self.bedrock_haiku.invoke_model(prompt='what do you see', base64_image_data=image, system_prompt='')
            # Record the end time
            end_time = time.time()
        elif self.vendor == 'hf':
            if (self.endpoint_name == 'HfStreamLLM'):
                start_time = time.time()
                self.st_llm.invoke_model("what is this")
                end_time = time.time()
        elif self.vendor == 'fireworks':
            image = encode_image_file_to_base64('assets/apple.jpg')
            start_time = time.time()
            self.fireworks.invoke_model(prompt="what is this", 
                                        base64_image_data=image)
            end_time = time.time()

        latency = end_time - start_time
        return latency

    def tpot():
        here=0

    def total_elapsed_time():
        here=0
        
if __name__ == "__main__":
    here=0
