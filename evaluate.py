from llmeval.smp.misc import argparse
from llmeval.smp.vlm import encode_image_to_base64, decode_base64_to_image
import ray
ray.init()
ray.autoscaler.sdk.request_resources(bundles=[{"GPU": 1}] * 2)


vendors = {"bedrock": "bedrock",
           "huggingface": "hf",
           "fireworks": "fireworks"}

models = { "haiku": "Claude3Wrapper", 
           "haiku_text": "Claude3Wrapper_text",
           "hf_llm": "HfStreamLLM",
           "clip_hf": "Clip",
           "fireworks": "Fireworks"}

datasets = { "sample_r1": ["local","q1_dev_logs"],
             "sample_conv": ["local", "q1_dev_conv"],
             "lvm_logs": ["local", "q1_dev_lvmlogs"],
             "intent_classfication": ["local", "intention_logs"],
             "intentc_may16_csv": ["s3","intention_may16_csv"]}

metrics = { "F1_score": "F1_score",
            "Clip": "CLIP",
            "sentenceT": "sentenceTransformer",
            "latency": "latency",}

def parse_args():
    parser = argparse.ArgumentParser(description="A simple argument parser")

    # Add arguments
    parser.add_argument('-vd', '--vendor', type=str, help='model api endpoint examples are bedrock, huggingface', required=True)
    parser.add_argument('-m', '--model', type=str, help='Model to use examples are haiku, haiku_text', required=True)
    parser.add_argument('-d', '--dataset', type=str, help='Dataset to use examples are lvm_logs, intent_classfication', required=True)
    parser.add_argument('-ddir', '--data_path', type=str, help='Dataset to use examples are lvm_logs, intent_classfication', required=True)
    parser.add_argument('-mt1', '--metric1', type=str, help='Metrics to compute examples are F1_score, Clip, latency', required=True)
    parser.add_argument('-mt2', '--metric2', type=str, help='Metrics to compute examples are F1_score, Clip, latency', required=True)
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose mode')


    # Parse the arguments
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    avg_clip = 0
    resp = []
    labels = []
    model = __import__("llmeval.models." + vendors[args.vendor], fromlist=models[args.model])
    model_obj = getattr(model, models[args.model])

    dataset = __import__("llmeval.evaluate." + datasets[args.dataset][1] , fromlist=datasets[args.dataset][1])
    dataset_obj = getattr(dataset, datasets[args.dataset][1])(data_dir=args.data_path)
    
    metric1 = __import__("llmeval.metrics." + metrics[args.metric1], fromlist=metrics[args.metric1])
    metric1_obj = getattr(metric1, metrics[args.metric1])()

    metric2 = __import__("llmeval.metrics." + metrics[args.metric2], fromlist=metrics[args.metric2])
    metric2_obj = getattr(metric2, metrics[args.metric2])(vendor=vendors[args.vendor], model_id=model_obj._photon_model)

    num = len(dataset_obj)
    import ray

    for ids in range(num):
        text, image, label = dataset_obj[ids]
        b64 = image
        if image and type(image) is not str:
            b64 = encode_image_to_base64(image)
            text = "what is this"
        
        if vendors[args.vendor] == 'hf':
            response = model_obj.invoke_model(prompt=text, system_prompt = dataset_obj.system_prompt[0], base64_image_data=b64)
        else:
            response = model_obj().invoke_model(prompt=text, system_prompt = dataset_obj.system_prompt[0], base64_image_data=b64)

        resp.append(response)
        if image and response:
            avg_clip += metric1_obj.compute(image = decode_base64_to_image(image), text=response)
        if label:
            labels.append(label)
        if ids > 5:
            break

    if dataset_obj.image_present:
        val = avg_clip / sum(i != '' for i in resp)
        print(f'the {metrics[args.metric1]} score for {model_obj._photon_model} is {val}')
    
    elif dataset_obj.labels_gt:
        val = metric1_obj.compute(y_true=labels, y_pred=resp)
        print(f'the {metrics[args.metric1]} score for {model_obj._photon_model} is {val}')

    else:
        val = metric1_obj.compute_batch(text_1=labels, text_2=resp)
        print(f'the {metrics[args.metric1]} score for {model_obj._photon_model} is {val}')
    
    val2 = metric2_obj.compute()
    print(f'the {metrics[args.metric2]} score for {model_obj._photon_model} is {val2}')


if __name__ == '__main__':
    main()