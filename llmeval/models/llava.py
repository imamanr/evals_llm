import torch
from PIL import Image
import sys
import os.path as osp
from .base import BaseModel
from ..smp import 

import os
import os.path as osp
import string
import sys
import warnings

import pandas as pd
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          CLIPImageProcessor, CLIPVisionModel,
                          GenerationConfig, StoppingCriteriaList)

from .base import BaseModel
from ..smp import cn_string, get_cache_path
#from ..utils import DATASET_TYPE

class LLaVA(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self,
                 model_pth='liuhaotian/llava_v1.5_7b',
                 **kwargs):
        try:
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path
        except:
            warnings.warn('Please install llava before using LLaVA')
            sys.exit(-1)

        warnings.warn('Please install the latest version of llava from github before you evaluate the LLaVA model. ')
        assert osp.exists(model_pth) or splitlen(model_pth) == 2

        if model_pth == 'Lin-Chen/ShareGPT4V-7B':
            model_name = 'llava-v1.5-7b'
        elif model_pth == 'Lin-Chen/ShareGPT4V-13B':
            model_name = 'llava-v1.5-13b'
        else:
            model_name = get_model_name_from_path(model_pth)

        try:
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                model_path=model_pth,
                model_base=None,
                model_name=model_name,
                device='cpu',
                device_map='cpu'
            )
        except:
            if 'ShareGPT4V' in model_pth:
                import llmeval.models.llava as llava
                warnings.warn(
                    'Please manually remove the encoder type check in '
                    f'{llava.__path__[0]}/model/multimodal_encoder/builder.py '
                    'Line 8 to use the ShareGPT4V model. ')
            else:
                warnings.warn('Unknown error when loading LLaVA model.')
            exit(-1)

        self.model = self.model.cuda()
        self.conv_mode = 'llava_v1'

        kwargs_default = dict(do_sample=False, temperature=0, max_new_tokens=512, top_p=None, num_beams=1, use_cache=True) # noqa E501
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == 'multi-choice':
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        question = line['question']
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        if hint is not None:
            question = hint + '\n' + question

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f'\n{key}. {item}'
        prompt = question

        if len(options):
            prompt += (
                '\n请直接回答选项字母。' if cn_string(prompt) else
                "\nAnswer with the option's letter from the given choices directly."
            )
        else:
            prompt += '\n请直接回答问题。' if cn_string(prompt) else '\nAnswer the question directly.'

        message = [dict(type='image', value=s) for s in tgt_path]
        message.append(dict(type='text', value=prompt))
        return message

    def generate_inner(self, message, dataset=None):
        from llava.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria
        from llava.constants import (
            IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN)
        from llava.conversation import conv_templates, SeparatorStyle

        # Support interleave text and image
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], 'PLACEHOLDER')
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        content, images = '', []
        for msg in message:
            if msg['type'] == 'text':
                content += msg['value']
            elif msg['type'] == 'image':
                if self.model.config.mm_use_im_start_end:
                    content += DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n'
                else:
                    content += DEFAULT_IMAGE_TOKEN + '\n'
                images.append(msg['value'])

        images = [Image.open(s).convert('RGB') for s in images]
        args = abstractproperty()
        args.image_aspect_ratio = 'pad'
        image_tensor = process_images(images, self.image_processor, args).to('cuda', dtype=torch.float16)
        prompt = prompt.replace('PLACEHOLDER', content)

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids, images=image_tensor, stopping_criteria=[stopping_criteria], **self.kwargs)

        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return output


class LLaVA_Next(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_pth='llava-hf/llava-v1.6-vicuna-7b-hf', **kwargs):
        import transformers
        assert version_cmp(transformers.__version__, '4.39.0', 'ge')
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        self.model_pth = model_pth
        if '34b' in model_pth.lower():
            self.processor = LlavaNextProcessor.from_pretrained(self.model_pth, use_fast=False)
        else:
            self.processor = LlavaNextProcessor.from_pretrained(self.model_pth)
        flash_attn_flag = False
        try:
            import flash_attn
            flash_attn_flag = True
        except ImportError:
            pass

        if flash_attn_flag:
            model = LlavaNextForConditionalGeneration.from_pretrained(
                self.model_pth, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_flash_attention_2=True)
        else:
            model = LlavaNextForConditionalGeneration.from_pretrained(
                self.model_pth, torch_dtype=torch.float16, low_cpu_mem_usage=True)

        model = model.eval()
        self.model = model.cuda()
        kwargs_default = dict(do_sample=False, temperature=0, max_new_tokens=512, top_p=None, num_beams=1)
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def apply_prompt_template(self, prompt):
        model_pth = self.model_pth.lower()
        if 'mistral' in model_pth:
            template = '[INST] PLACEHOLDER [/INST]'
        elif 'vicuna' in model_pth:
            template = (
                'A chat between a curious human and an artificial intelligence assistant. '
                "The assistant gives helpful, detailed, and polite answers to the human's questions. "
                'USER: PLACEHOLDER ASSISTANT:'
            )
        elif '34b' in model_pth:
            template = (
                '<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\nPLACEHOLDER<|im_end|>'
                '<|im_start|>assistant\n'
            )
        else:
            raise NotImplementedError(f'Prompt template for {model_pth} not implemented.')

        prompt = template.replace('PLACEHOLDER', f'<image>\n{prompt}')
        return prompt

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == 'multi-choice':
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        question = line['question']
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        if hint is not None:
            question = hint + '\n' + question

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f'\n{key}. {item}'
        prompt = question

        if len(options):
            prompt += (
                '\n请直接回答选项字母。' if cn_string(prompt) else
                "\nAnswer with the option's letter from the given choices directly."
            )
        else:
            prompt += '\n请直接回答问题。' if cn_string(prompt) else '\nAnswer the question directly.'
        message = [dict(type='image', value=s) for s in tgt_path]
        message.append(dict(type='text', value=prompt))
        return message

    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message)
        prompt = prompt.replace('<image>', '[ImageHere]')
        if prompt.find('[ImageHere]') != prompt.rfind('[ImageHere]'):
            prompt += '\nThere exists multiple images in the conversation, but only the first one is displayed.'

        image = Image.open(image_path).convert('RGB')
        prompt = self.apply_prompt_template(prompt)

        inputs = self.processor(prompt, image, return_tensors='pt').to('cuda')
        output = self.model.generate(**inputs, **self.kwargs)
        answer = self.processor.decode(output[0], skip_special_token=True)
        if '<s>' in answer:
            answer = answer.replace('<s>', '').strip()
        if '[/INST]' in answer:
            answer = answer.split('[/INST]')[1].strip()
        elif 'ASSISTANT:' in answer:
            answer = answer.split('ASSISTANT:')[1].strip()
        elif 'assistant\n' in answer:
            answer = answer.split('assistant\n')[1].strip()

        if '</s>' in answer:
            answer = answer.split('</s>')[0].strip()
        if '<|im_end|>' in answer:
            answer = answer.split('<|im_end|>')[0].strip()

        return answer


class LLaVA_XTuner(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = False

    def __init__(self,
                 llava_path,
                 llm_path=None,
                 visual_encoder_path='openai/clip-vit-large-patch14-336',
                 visual_select_layer=-2,
                 prompt_template=None,
                 stop_words=[],
                 torch_dtype=torch.float16):
        try:
            from peft import PeftModel
            from xtuner.utils import PROMPT_TEMPLATE, StopWordStoppingCriteria
        except Exception:
            warnings.warn(
                'Please install xtuner with `pip install -U xtuner` before '
                'using LLaVA_XTuner')
            sys.exit(-1)

        if not osp.isdir(llava_path):
            cache_path = get_cache_path(llava_path)
            if cache_path is not None:
                llava_path = cache_path
            else:
                llava_path = snapshot_download(repo_id=llava_path)
        assert osp.exists(llava_path) and osp.isdir(llava_path)

        # build visual_encoder
        if 'llm' in os.listdir(llava_path):
            assert llm_path is None, (
                "Please don't specify the `llm_path` since passed "
                '`llava_path` contains a LLM!')
            llm_path = osp.join(llava_path, 'llm')
        else:
            assert llm_path is not None, 'Please specify the `llm_path`!'

        llm = AutoModelForCausalLM.from_pretrained(llm_path,
                                                   trust_remote_code=True,
                                                   torch_dtype=torch_dtype,
                                                   device_map='cpu')
        tokenizer = AutoTokenizer.from_pretrained(llm_path,
                                                  trust_remote_code=True,
                                                  encode_special_tokens=True)
        print(f'Load LLM from {llm_path}')

        # build visual_encoder
        if 'visual_encoder' in os.listdir(llava_path):
            assert visual_encoder_path is None, (
                "Please don't specify the `visual_encoder_path` since passed "
                '`llava_path` contains a visual encoder!')
            visual_encoder_path = osp.join(llava_path, 'visual_encoder')
        else:
            assert visual_encoder_path is not None, (
                'Please specify the `visual_encoder_path`!')
        visual_encoder = CLIPVisionModel.from_pretrained(
            visual_encoder_path, torch_dtype=torch_dtype, device_map='cpu')
        image_processor = CLIPImageProcessor.from_pretrained(
            visual_encoder_path)
        print(f'Load visual_encoder from {visual_encoder_path}')

        # load adapter
        if 'llm_adapter' in os.listdir(llava_path):
            adapter_path = osp.join(llava_path, 'llm_adapter')
            llm = PeftModel.from_pretrained(llm,
                                            adapter_path,
                                            trust_remote_code=True,
                                            device_map='cpu')
            print(f'Load LLM adapter from {llava_path}')
        if 'visual_encoder_adapter' in os.listdir(llava_path):
            adapter_path = osp.join(llava_path, 'visual_encoder_adapter')
            visual_encoder = PeftModel.from_pretrained(visual_encoder,
                                                       adapter_path,
                                                       trust_remote_code=True,
                                                       device_map='cpu')
            print(f'Load visual_encoder adapter from {llava_path}')

        # build projector
        projector_path = osp.join(llava_path, 'projector')
        projector = AutoModel.from_pretrained(projector_path,
                                              trust_remote_code=True,
                                              torch_dtype=torch_dtype,
                                              device_map='cpu')
        print(f'Load projector from {llava_path}')

        llm.eval()
        visual_encoder.eval()
        projector.eval()

        self.llm = llm.cuda()
        self.tokenizer = tokenizer
        self.visual_encoder = visual_encoder.cuda()
        self.image_processor = image_processor
        self.projector = projector.cuda()
        self.visual_select_layer = visual_select_layer
        if prompt_template is not None:
            # modified prompt template
            if prompt_template == 'llama3_chat':
                self.prompt_template = dict(
                    SYSTEM=('<|start_header_id|>system<|end_header_id|>\n\n'
                            '{system}<|eot_id|>'),
                    INSTRUCTION=(
                        '<|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|>'
                        '<|start_header_id|>assistant<|end_header_id|>\n\n'),
                    SUFFIX='<|eot_id|>',
                    SUFFIX_AS_EOS=True,
                    STOP_WORDS=['<|eot_id|>'])
            else:
                self.prompt_template = PROMPT_TEMPLATE[prompt_template]
            stop_words += self.prompt_template.get('STOP_WORDS', [])
        else:
            self.prompt_template = None

        self.stop_criteria = StoppingCriteriaList()
        for word in stop_words:
            self.stop_criteria.append(
                StopWordStoppingCriteria(self.tokenizer, word))

    def build_gen_config(self, dataset):
        gen_kwargs = dict(max_new_tokens=512,
                          do_sample=True,
                          temperature=1,
                          num_beams=5,
                          eos_token_id=self.tokenizer.eos_token_id,
                          pad_token_id=self.tokenizer.pad_token_id
                          if self.tokenizer.pad_token_id is not None else
                          self.tokenizer.eos_token_id)
        # For single word generation
        if (dataset is not None
                and DATASET_TYPE(dataset) in ['multi-choice', 'Y/N']):
            gen_kwargs.update(
                dict(max_new_tokens=5, do_sample=False, num_beams=1))
        return GenerationConfig(**gen_kwargs)

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == 'multi-choice':
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        question = line['question']
        hint = line['hint'] if ('hint' in line
                                and not pd.isna(line['hint'])) else None
        if hint is not None:
            question = hint + '\n' + question

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f'\n{key}. {item}'

        if not cn_string(question):
            prompt = question + '\n' + ("Answer with the option's letter "
                                        'from the given choices directly.')
        else:
            prompt = question + '\n' + '请直接回答选项字母。'

        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])
        return message

    def generate_inner(self, message, dataset=None):
        from xtuner.dataset.utils import expand2square
        from xtuner.model.utils import prepare_inputs_labels_for_multimodal
        from xtuner.utils import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        prompt, image_path = self.message_to_promptimg(message)
        prompt = prompt.replace('<image>', '')
        image = Image.open(image_path).convert('RGB')
        image = expand2square(
            image,
            tuple(int(x * 255) for x in self.image_processor.image_mean))
        image = self.image_processor.preprocess(
            image, return_tensors='pt')['pixel_values'][0]
        image = image.cuda().unsqueeze(0)
        visual_outputs = self.visual_encoder(image, output_hidden_states=True)
        pixel_values = self.projector(
            visual_outputs.hidden_states[self.visual_select_layer][:, 1:])

        inputs = DEFAULT_IMAGE_TOKEN + '\n' + prompt

        if self.prompt_template:
            inputs = self.prompt_template['INSTRUCTION'].format(input=inputs)

        chunk_encode = []
        for idx, chunk in enumerate(inputs.split(DEFAULT_IMAGE_TOKEN)):
            if idx == 0:
                cur_encode = self.tokenizer(chunk)
            else:
                cur_encode = self.tokenizer(chunk, add_special_tokens=False)
            chunk_encode.append(cur_encode)
        assert len(chunk_encode) == 2
        ids = []
        for idx, cur_chunk_encode in enumerate(chunk_encode):
            ids.extend(cur_chunk_encode['input_ids'])
            if idx != len(chunk_encode) - 1:
                ids.append(IMAGE_TOKEN_INDEX)
        ids = torch.tensor(ids).cuda().unsqueeze(0)
        mm_inputs = prepare_inputs_labels_for_multimodal(
            llm=self.llm, input_ids=ids, pixel_values=pixel_values)

        gen_config = self.build_gen_config(dataset)
        generate_output = self.llm.generate(
            **mm_inputs,
            generation_config=gen_config,
            streamer=None,
            bos_token_id=self.tokenizer.bos_token_id,
            stopping_criteria=self.stop_criteria)
        predict = self.tokenizer.decode(generate_output[0],
                                        skip_special_tokens=True).strip()
        return predict
