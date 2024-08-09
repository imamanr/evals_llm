from PIL import Image
import requests
import torch
from transformers import CLIPProcessor, CLIPModel
# We can also initialize a CLIPConfig from a CLIPTextConfig and a CLIPVisionConfig
from transformers import CLIPTextConfig, CLIPVisionConfig
from transformers import CLIPConfig, CLIPModel, CLIPImageProcessor, CLIPTokenizer, AutoProcessor
from .base import embMetrics
from ..smp.vlm import decode_base64_to_image, encode_image_to_base64

import io
import os
import urllib
from typing import List
import open_clip

from PIL import Image
import torch
import validators

from leptonai.photon import Photon, handler, HTTPException
#from leptonai.photon.types import lepton_unpickle, is_pickled, LeptonPickled
import ray 
#ray.init()

DEFAULT_MODEL_NAME = "ViT-B-32-quickgelu"
DEFUALT_PRETRAINED = "laion400m_e32"

class Clip_photon(Photon):
    """
    This photon is used to embed text and image in embeddings
    """

    requirement_Dependency = [
        "open_clip_torch",
        "Pillow",
        "torch",
        "validators",
    ]

    def init(self):
        self.DEVICE = "cpu"
        if torch.cuda.is_available():
            self.DEVICE = "cuda"
        MODEL_NAME = (
            os.environ["MODEL_NAME"]
            if "MODEL_NAME" in os.environ
            else DEFAULT_MODEL_NAME
        )
        PRETRIANED = (
            os.environ["PRETRAINED"]
            if "PRETRAINED" in os.environ
            else DEFUALT_PRETRAINED
        )
        (
            self.CLIP_MODEL,
            _,
            self.CLIP_IMG_PROCESSESS,
        ) = open_clip.create_model_and_transforms   (
            model_name=MODEL_NAME, pretrained=PRETRIANED, device=self.DEVICE
        )
        self.TOKENIZER = open_clip.get_tokenizer(MODEL_NAME)

    @handler("embed")
    def embed(self, query:str) -> List[float]:
        if validators.url(query):
            return self.embed_image(query)

    @handler("embed_text")
    def embed_text(self, query: str) -> List[float]:
        query = self.TOKENIZER([query])
        with torch.no_grad():
            text_features = self.CLIP_MODEL.encode_text(query.to(self.DEVICE))
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return list(text_features.cpu().numpy()[0].astype(float))
    
    def embed_text_local(self, query: str) -> List[float]:
        query = self.TOKENIZER([query])
        with torch.no_grad():
            text_features = self.CLIP_MODEL.encode_text(query.to(self.DEVICE))
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return list(text_features.cpu().numpy()[0].astype(float))


    def embed_image_local(self, image: Image):
        image = self.CLIP_IMG_PREPROCESS(image).unsqueeze(0).to(self.DEVICE)
        with torch.no_grad():
            image_features = self.CLIP_MODEL.encode_image(image)
            image_features /= image_features.norm(dim=1, keepdim=True)
        return list(image_features.cpu().numpy()[0].astype(float))
        
    @handler("embed_image")
    def embed_image(self, url:str) -> List[float]:
        # open the image url
        try:
            raw_image = Image.open(io.BytesIO(urllib.request.urlopen(url).read()))
        except Exception as e:
            raise HTTPException(
                status_code =400,
                detail=(f'Cannot open image at url {url}. Detailed error message: {str(e)}'
                ),
            )
        return self.embed_image_local(raw_image)
        

    def invoke_embedding(self, prompt, base64_image_data, system_prompt=""):
            clip = Clip_photon()
            ClipRayClass = ray.remote(Clip_photon)
            actor = ClipRayClass.remote()

            local_result_text = clip.embed_text_local(query=prompt)
            ray_result_text = ray.get(actor.embed_text.remote(query=prompt))

            # print("comparing the results")
            # print(local_result_text[:5])
            # print(ray_result_text[:5])

            # image_local_result = clip.embed_image(decode_base64_to_image(base64_image_data))
            # ray_result_image = ray.get(actor.embed_image.remote(decode_base64_to_image(base64_image_data)))

            return ray_result_text#, ray_result_image
    
# c = Clip_photon()
# c.invoke_embedding(prompt="here",base64_image_data="")

class CLIP(embMetrics):
    def __init__(self) -> None:

        from transformers import (
            CLIPTextConfig,
            CLIPVisionConfig,
            CLIPTextModelWithProjection,
            CLIPVisionModelWithProjection
        )
        CLIP_CHECKPOINTS = "openai/clip-vit-base-patch32"  
        PROJECTION_DIM = 512  # Replace with your desired projection dimension
        padding_max_length = 100  # Replace with your desired maximum position embeddings

        textConfig = CLIPTextConfig.from_pretrained(CLIP_CHECKPOINTS)
        textConfig.projection_dim = PROJECTION_DIM
        textConfig.max_position_embeddings = padding_max_length

        visionConfig = CLIPVisionConfig.from_pretrained(CLIP_CHECKPOINTS)
        visionConfig.projection_dim = PROJECTION_DIM

        # Using CLIP text model with a projection head on top
        clipTextModel = CLIPTextModelWithProjection.from_pretrained(
            pretrained_model_name_or_path=CLIP_CHECKPOINTS,
            config=textConfig,
            ignore_mismatched_sizes=True
        )

        # Using CLIP vision ViT model with a projection head on top 
        clipVisionModel = CLIPVisionModelWithProjection.from_pretrained(
            pretrained_model_name_or_path=CLIP_CHECKPOINTS,
            config=visionConfig,
            ignore_mismatched_sizes=True
        )

        tokenizer = CLIPTokenizer(vocab_file = 'assets/vocab.json',
                    merges_file='assets/merges.txt',
                    errors = 'replace',
                    unk_token = '<|endoftext|>',
                    bos_token = '<|startoftext|>',
                    eos_token = '<|endoftext|>',
                    pad_token = '<|endoftext|>')
        #preocessor = CLIPProcessor(image_processor = clipVisionModel, tokenizer = tokenizer)

        self.processor_vision = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.processor_text = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        self.model_image = clipVisionModel
        self.model_text = clipTextModel

    def compute(self, text, image):
        # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        # image = Image.open(requests.get(url, stream=True).raw)

        inputs = self.processor_text(text=text[:100], return_tensors="pt", padding=True)
        outputs_text = self.model_text(**inputs)

        inputs = self.processor_vision(images=image, return_tensors="pt")
        outputs_images = self.model_image(**inputs)
        simil_score = torch.matmul(outputs_images.image_embeds, outputs_text.text_embeds.T)
        # logits_per_image = outputs_text.logits_per_image  # this is the image-text similarity score
        # probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        return float(simil_score)