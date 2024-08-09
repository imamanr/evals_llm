import os
import openai

client = openai.OpenAI(
    base_url="https://llama3-70b.lepton.run/api/v1/",
    api_key=os.environ.get('LEPTON_API_TOKEN')
)

completion = client.chat.completions.create(
    model="llama3-70b",
    messages=[
        {"role": "user", "content": "say hello"},
    ],
    max_tokens=128,
    stream=True,
)

for chunk in completion:
    if not chunk.choices:
        continue
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="")