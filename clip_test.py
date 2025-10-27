import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# The only change is adding use_safetensors=True
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load your image
# Replace "your_image.jpg" with the path to your image file
try:
    image = Image.open("/home/hliu852/VLM for RL project/VLM-for-RL/test video/mug_history/04A.jpg")
except FileNotFoundError:
    print("Error: Image file not found. Please replace 'your_image.jpg' with the actual path to your image.")
    exit()

# Your text prompt
text_prompt = "The robot arm is holding a mug"
background_description = "there are some people in a lab, with a table and a robot arm."
print("text_prompt is : ", text_prompt)
print("background_description is : ", background_description)
# Preprocess the image and text
inputs = processor(text=[text_prompt], images=image, return_tensors="pt", padding=True)

# Get the image and text embeddings
with torch.no_grad():
    outputs = model(**inputs)
    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds

# Normalize the embeddings
image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

# Calculate the cosine similarity
cosine_similarity = torch.matmul(text_embeds, image_embeds.t()).item()

print(f"Cosine similarity between the image and the text prompt: {cosine_similarity}")