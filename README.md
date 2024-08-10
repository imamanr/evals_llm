Model evaluation tools, with a focus on Large Language Models and Large Vision Models.

# How-to run Hello World

## Clone this repo

`$ git clone git@github.com:imamanr/evals_llm.git`

## Setup your python venv

`$ python3 -m venv venv`
`$ source venv/bin/activate`
`(venv)$ pip install -U pip`
`(venv)$ pip install -r requirements.txt`

## Authenticate with AWS

*TODO -- use `aws configure sso` instead*

1. Use AWS credentials

## Run Hello World

`$ python evaluate.py --help`

# Example evaluations

Evaluate data from s3 bucket on Haiku Claude via Bedrock API:

`$ python evaluate.py -m haiku -vd bedrock -d lvm_logs -ddir ~/Documents/Rabbit/Datasets/ -mt1 Clip -mt2 latency`

Evaluate local toy dataset on LLaVa via Fireworks API:

`$ python evaluate.py -m fireworks -vd fireworks -d lvm_logs -ddir ~/Documents/Rabbit/Datasets/ -mt1 Clip -mt2 latency`

Evaluate sample converstation data on locally run TinyLlama

`$ python evaluate.py -m hf_llm -vd huggingface -d sample_conv -ddir ./assets/data_r1/ -mt1 sentenceT -mt2 latency`

# Add a new model

First, add a new module under `llmeval/models/` within the respective vendor script (e.g. `bedrock.py`).  Next, append the model name to the `vendor` and `model` dicts in `evaluate.py`. Lastly, update `latency.py` with the corresponding `model` stump.

# Add a new dataset

First, add a new module under `llmeval/evalaute/`. Next, add the module name and class to the `dataset` dict in `evaluate.py`.

# Add a new metric

First, add a new module under `llmeval/metric/`. Next, add to the `metrics` dictionary in `evaluate.py` 

# References:
- LeptonAI: [Leptonai SDK](https://github.com/leptonai/leptonai)
- Boto: [Boto SDK](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html)
- Bedrock: [Bedrock API](https://aws.amazon.com/bedrock/?gclid=CjwKCAjwvIWzBhAlEiwAHHWgvX7x17KDXMh0dkJQxdjyBQ3zJl9py7L_RmhgMloOZoEIYgffpzHP_RoC9BYQAvD_BwE&trk=36201f68-a9b0-45cc-849b-8ab260660e1c&sc_channel=ps&ef_id=CjwKCAjwvIWzBhAlEiwAHHWgvX7x17KDXMh0dkJQxdjyBQ3zJl9py7L_RmhgMloOZoEIYgffpzHP_RoC9BYQAvD_BwE:G:s&s_kwcid=AL!4422!3!692006004850!e!!g!!bedrock!21048268689!159639953975)
- HuggingFace: [Hugginface models](https://huggingface.co/docs/transformers/en/main_classes/model)