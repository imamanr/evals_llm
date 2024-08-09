import boto3
from botocore.exceptions import ClientError

import base64
import json

import logging
logger = logging.getLogger(__name__)

from .base import BaseAPI

class Claude3Wrapper_text(BaseAPI):
    """Encapsulates Claude 3 model invocations using the Amazon Bedrock Runtime client."""

    def __init__(self, client=None):
        """
        :param client: A low-level client representing Amazon Bedrock Runtime.
                       Describes the API operations for running inference using Bedrock models.
                       Default: None
        """
        self.client = client
        self._photon_model = "anthropic.claude-3-opus-20240229-v1:0"
        self.model_id = "anthropic.claude-3-opus-20240229-v1:0"


    def invoke_model(self, prompt, system_prompt, base64_image_data=''):
        """
        Invokes Anthropic Claude 3 Sonnet to run an inference using the input
        provided in the request body.

        :param prompt: The prompt that you want Claude 3 to complete.
        :return: Inference response from the model.
        """

        # Initialize the Amazon Bedrock runtime client
        client = self.client or boto3.client(
            service_name="bedrock-runtime", region_name="us-west-2")

        # Invoke Claude 3 with the text prompt

        body={
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 256,
                "messages": prompt,
                # "messages": [
                #     {
                #         "role": "user",#prompt['role'],
                #         "content": [{"type": "text", "text": prompt}], #prompt['content']}],
                #     }
                # ],
                "system": system_prompt
                    }
        
        try:
            response = client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body))

            # Process and print the response
            result = json.loads(response.get("body").read())
            input_tokens = result["usage"]["input_tokens"]
            output_tokens = result["usage"]["output_tokens"]
            output_list = result.get("content", [])

            # print("Invocation details:")
            # print(f"- The input length is {input_tokens} tokens.")
            # print(f"- The output length is {output_tokens} tokens.")
            # print(f"- The model returned {len(output_list)} response(s):")
            # for output in output_list:
            #     print(output["text"])

            return result['content'][0]['text']

        except ClientError as err:
            logger.error(
                "Couldn't invoke Claude 3 Sonnet. Here's why: %s: %s",
                err.response["Error"]["Code"],
                err.response["Error"]["Message"],
            )
            raise

class Claude3Wrapper(BaseAPI):
    """Encapsulates Claude 3 model invocations using the Amazon Bedrock Runtime client."""

    def __init__(self, client=None):
        """
        :param client: A low-level client representing Amazon Bedrock Runtime.
                       Describes the API operations for running inference using Bedrock models.
                       Default: None
        """
        self.client = client
        self.model_id = "anthropic.claude-3-haiku-20240307-v1:0"
        self._photon_model = "anthropic.claude-3-haiku-20240307-v1:0"

    def invoke_model(self, prompt, system_prompt, base64_image_data):
        """
        Invokes Anthropic Claude 3 Sonnet to run a multimodal inference using the input
        provided in the request body.

        :param prompt:            The prompt that you want Claude 3 to use.
        :param base64_image_data: The base64-encoded image that you want to add to the request.
        :return: Inference response from the model.
        """

        # Initialize the Amazon Bedrock runtime client
        session = boto3.session.Session()#boto3.Session(region_name = 'us-west-2')
        client = session.client(
            service_name="bedrock-runtime", region_name="us-west-2"
        )
        # Invoke the model with the prompt and the encoded image
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64_image_data,
                            },
                        },
                    ],
                }
            ],
        }

        try:
            response = client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body),
            )

            # Process and print the response
            result = json.loads(response.get("body").read())
            # input_tokens = result["usage"]["input_tokens"]
            # output_tokens = result["usage"]["output_tokens"]
            # output_list = result.get("content", [])

            # print("Invocation details:")
            # print(f"- The input length is {input_tokens} tokens.")
            # print(f"- The output length is {output_tokens} tokens.")
            # print(f"- The model returned {len(output_list)} response(s):")
            
            # for output in output_list:
            #     print(output["text"])
            return result['content'][0]['text']
        
        except ClientError as err:
            logger.error(
                "Couldn't invoke Claude 3 Sonnet. Here's why: %s: %s",
                err.response["Error"]["Code"],
                err.response["Error"]["Message"],
            )
            return []
