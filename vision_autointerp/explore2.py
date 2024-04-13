# %%
import base64
from openai import OpenAI

client = OpenAI()


import os
import base64


def load_images_for_neuron(neuron_id):
    image_dir = "../new_outputs_3_28/data/explore_outputs/"
    image_files = [
        f for f in os.listdir(image_dir) if f"neuronId={neuron_id}_imgId" in f
    ]

    base64_images = []
    for image_file in image_files:
        with open(os.path.join(image_dir, image_file), "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")
            base64_images.append(base64_image)

    return base64_images


NEURON_ID = "7_FC1_889"
img_data_list = load_images_for_neuron(NEURON_ID)
print("loaded {} images for neuronId={}".format(len(img_data_list), NEURON_ID))
img_payload = [
    {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{img_data}",
            "detail": "auto",
        },
    }
    for img_data in img_data_list
]


# file = "./activation_grids/neuronId=7_FC1_326_nImages=16.png"
# with open(file, "rb") as image_file:
#     base64_image_str = base64.b64encode(image_file.read()).decode("utf-8")

prompt_sonia = "This is a grid of 10 heatmaps of a TinyCLIP neuron's activations. These are the maximally activating images, \
    with yellowish being a more activating patch, and blue as less activating. Could you interpret for me what the neuron is selecting for? \
      Keep your answer and concise as possible, \
      notice small details and patterns across the images. Look at the yellow patches in particular. The labels can be misleading so don't give them too much weight. \
        Respond in a concise, descriptive phrase exactly what it's selecting for, and \
            try not to use words neuron, activation, or image, or focus/selective/etc, because that will be redundant when we visualize this information. \
            Just respond with the features themselves, paying special attention to the yellow patches. Keep your words simple but be pretty specific and granular. For example, instead of saying 'face,' note the \
              specific parts of a face, like ears, neck, etc."

prompt = """
You are an expert at image analysis and factually reporting the concepts within images.

Input details
- The provided file contains a grid of images
- Each image is overlaid with a heatmap indicating the importance of regions within the image. This represents regions of high activation for a vision machine learning model.
- The regions that are bright yellow are EXTREMELY IMPORTANT and the regions that are darker purple DO NOT MATTER AT ALL. Only focus on the yellow regions of the images, and heavily penalize any objects that are highlighted in purple.

Desired output
- Important concepts: Anything highlighted in yellow across any or all of the images. ONLY INCLUDE PARTS OF THE IMAGE THAT ARE YELLOW AND EXCLUDE ANY PARTS THAT ARE PURPLE IN THIS ANALYSIS.
- Top concept: Of the important concepts, pick the single top concept
- Unimportant concepts: Anything highlighted in purple across any of the images
- Number of images loaded
- Keep your answer and concise as possible, with no preamble.
"""

prompt_info = """
Please answer the following questions about the image:
1. What are the image ids in the image and their respective activation?
2. What's in the image in the bottom row, first column?
3. Which images include leaves?
4. If there are leaves, do they appear to be highlighed in yellow? For which images?
"""

response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt_sonia}] + img_payload[:9],
        }
    ],
    temperature=0,
    # max_tokens=1000,
)

print(response.choices[0].message.content)
