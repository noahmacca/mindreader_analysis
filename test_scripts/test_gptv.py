# %%
from openai import OpenAI

client = OpenAI(
    organization="org-8UJvvxXiq5JJ1KU6H24u7BWd",
)

# %%
vision_model = "gpt-4-vision-preview"

chat_completion = client.chat.completions.create(
    model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hello world"}]
)

chat_completion.choices[0].message.content

# %%
image_files = [
    "https://mindreader-web.s3.amazonaws.com/image/464.jpg",
    "https://mindreader-web.s3.amazonaws.com/test_files/Screenshot+2024-03-08+at+12.44.03%E2%80%AFPM.png",
    "https://mindreader-web.s3.amazonaws.com/image/126.jpg",
    "https://mindreader-web.s3.amazonaws.com/test_files/Screenshot+2024-03-08+at+12.43.56%E2%80%AFPM.png",
    "https://mindreader-web.s3.amazonaws.com/image/150.jpg",
    "https://mindreader-web.s3.amazonaws.com/test_files/Screenshot+2024-03-08+at+12.43.48%E2%80%AFPM.png",
    "https://mindreader-web.s3.amazonaws.com/test_files/Screenshot+2024-03-08+at+12.43.42%E2%80%AFPM.png",
]

img_messages = [{"type": "image_url", "image_url": {"url": u}} for u in image_files]

prompt = """You are an expert at image analysis. Below are image files that a specific neuron in a neural network is maximally activating for. It's likely the that there is a shared, specific concept, or small number of concepts, across all of the provided images. you will see both normal images, and variants of those images with overlaid heatmaps, which highlight particular areas of the image that are highly activating to the neuron in yellow.
                    
Please provide the simplest, smallest, and most specific set of concepts that are shared across ALL of these images. Explain your reasoning step-by-step.

Sample Output:
Shared concepts:
- Trees

Reasoning:
- All of the images contain forests in some form.
- The highlighted yellow regions are highly overlapping with trees.

Please begin.
"""

response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                },
            ]
            + img_messages,
        }
    ],
    max_tokens=300,
)

print(response.choices[0].message.content)

# %%
