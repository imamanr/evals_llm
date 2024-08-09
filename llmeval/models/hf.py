import logging
import os
from threading import Thread
from queue import Queue
import urllib.request
from .base import BaseAPI

from leptonai.photon import Photon, handler,StreamingResponse
#from leptonai.utils import photon_run_local_server_simple
from leptonai.photon import create

from fastapi import FastAPI
import torch

import ray
from ray import serve

from transformers import pipeline
logger = logging.getLogger('LOAD_ENV')
from ..smp.vlm import decode_base64_to_image
app = FastAPI()


#@ray.remote
class HfStreamLLM_ray:

    requirement_Dependency = [
        "transformers",
        "Pillow",
        "torch",
        "validators",
    ]

    #def __init__(self):
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    _photon_model = "hf:TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    def __init__(self):
        self.model_pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device="cpu")
        
    #@ray.remote
    def response_text_local(self, prompt):
        messages = [
            {
                "role": "system",
                "content": "",
            },
            {"role": "user", "content": prompt},
        ]
        prompt = self.model_pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        resp = self.model_pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        return resp[0]["generated_text"]

    #@ray.remote
    def response_text(self, prompt):
        self.response_text_local(prompt)

    @ray.remote
    def invoke_model(prompt: str, system_prompt="", base64_image_data="") -> str:
    
        HfsRayClass = ray.remote(HfStreamLLM_ray)
        actor = HfsRayClass.remote()

        ray_result_text = ray.get(actor.response_text.remote(prompt=prompt))

        return ray_result_text

class HfStreamLLM_remote():
    
    def __init__(self) -> None:
        self.hf_llm = HfStreamLLM_ray.remote()

class HfStreamLLM(Photon):

    requirement_Dependency = [
        "transformers",
        "Pillow",
        "torch",
        "validators",
    ]
    _photon_model = "hf:TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    def init(self):
        self.model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        name = "my-photon12"
        
        #self.ph = create(name, model)
        self.model_pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device_map="cpu")
        
    @handler("response_text_local")
    def response_text_local(self, prompt):
        messages = [
            {
                "role": "system",
                "content": "conversation",
            },
            {"role": "user", "content": prompt},
        ]
        prompt = self.model_pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        resp = self.model_pipe(prompt, max_new_tokens=1024, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        return resp[0]["generated_text"]

    @handler("response_text")
    def response_text(self, prompt):
        self.response_text_local(prompt)


    def invoke_model(prompt: str, system_prompt="", base64_image_data="") -> str:
    
        hf_llm = HfStreamLLM()
        # HfsRayClass = ray.remote(HfStreamLLM)
        # actor = HfsRayClass.remote()

        local_result_text = hf_llm.response_text_local(prompt=prompt)
        #ray_result_text = ray.get(actor.response_text.remote(prompt=prompt))

        return local_result_text


    

# from fastapi import FastAPI
# from ray import serve

# from leptonai.photon import create as photon_create

# app = FastAPI()

# @serve.deployment
# @serve.ingress(app)
# class ClipRayAdapter:
#     def __init__(self):
#         self._instance = Clip()
#         self._instance.init()
#         app.mount("/", self._instance._create_app(load_mount=True))


#     @app.get("/lepton-version")
#     def version(self) -> str:
#         """
#         A helped endpoint to inspect
#         """

#         import leptonai
#         return leptonai.__version__

# clip_app = serve.run(ClipRayAdapter.bind())
# import requests
# print(requests.get("http://localhost:8000/lepton-version").text)


# serve_result = requests.post("http://localhost:8000/embed_image", json={"url": url}).json()
# print(image_local_results[:5])
# print(serve_result[:5])



# from fastapi import FastAPI
# from leptonai.api.phton import create as photon_create

# from ray import serve

# app = FastAPI()

# @serve_deployment
# @serve_ingress(app)
# class GenericRayAdapter:
#     def __inti__(self, model_str):
#         self.photon = photon_create(name="lepton-ray-adapt", model=model_str)
#         self.photon.init()
#         self.myapp = self.photon._create_app(load_mount=True)
#         app.mount("/lepton", self.myapp)

#     @app.get("/lepton-version")
#     def version(self) -> str:
#         import leptonai
#         return leptonai.__version__

# gpt2_app = serve.run(GenericRayAdapter.bind("hf.gpt2"))

# import requests

# version = requests.get("http://localhost:8000/lepton-version").text
# result = requests.post("http://localhost:8000/lepton/run", json={"inputs": "Once upon a time"}).json

# print(f"lepton {version}")
# print(f"leptonai resut {result}")




# import ray 
# from leptonai.photon import Photon

# class Echo(Photon):
#     @Photon.handler

#     def run(self, input: str) -> str:
#         return input
# actor = ray.remote(Echo).remote()


# ret = actor.run.remote 