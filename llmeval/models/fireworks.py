import fireworks.client
import base64
from .base import BaseAPI

class Fireworks(BaseAPI):

  def __init__(self):
    self.client = fireworks.client
    self.client.api_key = "UE4CA533sGV36TxlYxVGfMfWTCwtq4vzfyGK9xopl7CEIwG1"
    self._photon_model = "accounts/fireworks/models/firellava-13b"

  # Helper function to encode the image
  def encode_image(image_path):
    with open(image_path, "rb") as image_file:
      return base64.b64encode(image_file.read()).decode('utf-8')

  def invoke_model(self, prompt, base64_image_data, system_prompt=''):
      # The path to your image
      # image_path = "your_image.jpg"
      response =[]
      try:
        response = self.client.ChatCompletion.create(
            model = "accounts/fireworks/models/firellava-13b",
            messages = [{
                "role": "user",
                "content": [{
                  "type": "text",
                  "text": prompt,
                }, {
                  "type": "image_url",
                  "image_url": {"url": f"data:image/jpeg;base64,{base64_image_data}"},
                }]
                }
                ])
      except:
          return []


      return response.choices[0].message.content