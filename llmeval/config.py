from llmeval.models.bedrock import Claude3Wrapper
from functools import partial

PandaGPT_ROOT = None
MiniGPT4_ROOT = None
TransCore_ROOT = None
Yi_ROOT = None
OmniLMM_ROOT = None
Mini_Gemini_ROOT = None
VXVERSE_ROOT = None
LLAVA_V1_7B_MODEL_PTH = 'Please set your local path to LLaVA-7B-v1.1 here, the model weight is obtained by merging LLaVA delta weight based on vicuna-7b-v1.1 in https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md with vicuna-7b-v1.1. '


api_models = {
    # GPT-4V series
    # 'GPT4V': partial(GPT4V, model='gpt-4-1106-vision-preview', temperature=0, img_size=512, img_detail='low', retry=10),
    # 'GPT4V_HIGH': partial(GPT4V, model='gpt-4-1106-vision-preview', temperature=0, img_size=-1, img_detail='high', retry=10),
    # 'GPT4V_20240409': partial(GPT4V, model='gpt-4-turbo-2024-04-09', temperature=0, img_size=512, img_detail='low', retry=10),
    # 'GPT4V_20240409_HIGH': partial(GPT4V, model='gpt-4-turbo-2024-04-09', temperature=0, img_size=-1, img_detail='high', retry=10),
    # 'GPT4o': partial(GPT4V, model='gpt-4o-2024-05-13', temperature=0, img_size=512, img_detail='low', retry=10),
    # 'GPT4o_HIGH': partial(GPT4V, model='gpt-4o-2024-05-13', temperature=0, img_size=-1, img_detail='high', retry=10),
    # # Gemini-V
    # 'GeminiProVision': partial(GeminiProVision, temperature=0, retry=10),
    # # Qwen-VL Series
    # 'QwenVLPlus': partial(QwenVLAPI, model='qwen-vl-plus', temperature=0, retry=10),
    # 'QwenVLMax': partial(QwenVLAPI, model='qwen-vl-max', temperature=0, retry=10),
    # # Reka Series
    # 'RekaEdge': partial(Reka, model='reka-edge-20240208'), 
    # 'RekaFlash': partial(Reka, model='reka-flash-20240226'), 
    # 'RekaCore': partial(Reka, model='reka-core-20240415'), 
    # # Step1V Series
    # 'Step1V': partial(GPT4V, model='step-1v-8k', api_base="https://api.stepfun.com/v1/chat/completions", temperature=0, retry=10),
    # Internal Only
    # 'GPT4V_INT': partial(GPT4V_Internal, model='gpt-4-vision-preview', temperature=0, img_size=512, img_detail='low', retry=10),
    #'Step1V_INT': partial(Step1V_INT, temperature=0, retry=10),
    # 'Claude3V_Opus': partial(Claude3V, model='claude-3-opus-20240229', temperature=0, retry=10),
    # 'Claude3V_Sonnet': partial(Claude3V, model='claude-3-sonnet-20240229', temperature=0, retry=10),
    # 'Claude3V_Haiku': partial(Claude3V, model='claude-3-haiku-20240307', temperature=0, retry=10),
    'Claude3V_Haiku_v1': partial(Claude3Wrapper, model='claude-3-haiku-20240307-v1:0', temperature=0, retry=10),

    # GLM4V
    #'GLM4V': partial(GLMVisionAPI, model='glm4v-biz-eval', temperature=0, retry=10),
}


# llava_series = {
#     'llava_v1.5_7b': partial(LLaVA, model_pth='liuhaotian/llava-v1.5-7b'),
#     'llava_v1.5_13b': partial(LLaVA, model_pth='liuhaotian/llava-v1.5-13b'),
#     'llava_v1_7b': partial(LLaVA, model_pth=LLAVA_V1_7B_MODEL_PTH),
#     'sharegpt4v_7b': partial(LLaVA, model_pth='Lin-Chen/ShareGPT4V-7B'),
#     'sharegpt4v_13b': partial(LLaVA, model_pth='Lin-Chen/ShareGPT4V-13B'),
#     'llava_next_vicuna_7b': partial(LLaVA_Next, model_pth='llava-hf/llava-v1.6-vicuna-7b-hf'),
#     'llava_next_vicuna_13b': partial(LLaVA_Next, model_pth='llava-hf/llava-v1.6-vicuna-13b-hf'),
#     'llava_next_mistral_7b': partial(LLaVA_Next, model_pth='llava-hf/llava-v1.6-mistral-7b-hf'),
#     'llava_next_yi_34b': partial(LLaVA_Next, model_pth='llava-hf/llava-v1.6-34b-hf'),
# }

supported_VLM = {}

model_groups = [
    api_models
]

for grp in model_groups:
    supported_VLM.update(grp)

if __name__ == '__main__':
    import sys
    ver = sys.argv[1]
