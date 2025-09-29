
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Path to your downloaded model
model_path = "/home/hliu852/Qwen2.5-VL-7B-Instruct"


# Load model and processor
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"   # uses your GPU if available
)
processor = AutoProcessor.from_pretrained(model_path)

# Simple example: image + text
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image briefly."},
        ],
    }
]

# Preprocess
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to("cuda" if torch.cuda.is_available() else "cpu")

# Generate output
with torch.inference_mode():
    generated_ids = model.generate(**inputs, max_new_tokens=64)

# Remove the input tokens from the output
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

# Decode
output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)

print("Model output:", output_text[0])
