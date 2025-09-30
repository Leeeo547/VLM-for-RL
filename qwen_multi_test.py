import torch
import json
from pathlib import Path
from datetime import datetime
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ============================================================
# Configuration
# ============================================================
MODEL_PATH = "/home/hliu852/Qwen2.5-VL-7B-Instruct"
PROMPTS_FILE = "prompts.json"  # Input file with your prompts
OUTPUT_FILE = "mug5.json"  # Output file to store conversations

# ============================================================
# Load Model and Processor
# ============================================================
print("Loading model and processor...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    dtype="auto",
    device_map="auto",
    local_files_only=True,
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    trust_remote_code=True
)
print("Model loaded successfully!\n")

# ============================================================
# Helper Functions
# ============================================================

def load_prompts(file_path):
    """
    Load prompts from a JSON file
    
    Expected format:
    {
        "prompts": [
            {
                "id": "prompt_1",
                "images": ["path/to/image1.jpg", "https://url/to/image2.jpg"],
                "text": "Your question or instruction here",
                "max_new_tokens": 256
            },
            ...
        ]
    }
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['prompts']


def save_conversation(conversation, output_file):
    """Save the full conversation to a JSON file"""
    # Load existing conversations if file exists
    if Path(output_file).exists():
        with open(output_file, 'r') as f:
            data = json.load(f)
    else:
        data = {"conversations": []}
    
    # Add new conversation with timestamp
    conversation_entry = {
        "timestamp": datetime.now().isoformat(),
        "conversation": conversation
    }
    data["conversations"].append(conversation_entry)
    
    # Save back to file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Conversation saved to {output_file}")


def query_model(messages, max_new_tokens=256):
    """
    Query the model with a conversation history
    
    Args:
        messages: List of message dictionaries in Qwen format
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        Generated text response
    """
    # Prepare inputs
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device)
    
    # Generate
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    # Trim and decode
    generated_ids_trimmed = [
        out_ids[len(in_ids):] 
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    
    return output_text[0]


def process_prompts_sequence(prompts_list):
    """
    Process a sequence of prompts, maintaining conversation context
    
    Args:
        prompts_list: List of prompt dictionaries
    
    Returns:
        Full conversation history with all responses
    """
    conversation = []
    
    for i, prompt in enumerate(prompts_list):
        print(f"\n{'='*60}")
        print(f"Processing Prompt {i+1}/{len(prompts_list)}")
        print(f"ID: {prompt.get('id', 'N/A')}")
        print(f"Text: {prompt['text'][:100]}..." if len(prompt['text']) > 100 else f"Text: {prompt['text']}")
        print('='*60)
        
        # Build content for this turn
        content = []
        
        # Add images if provided
        if 'images' in prompt and prompt['images']:
            for img_path in prompt['images']:
                content.append({"type": "image", "image": img_path})
                print(f"  - Image: {img_path}")
        
        # Add text
        content.append({"type": "text", "text": prompt['text']})
        
        # Add user message to conversation
        user_message = {
            "role": "user",
            "content": content
        }
        conversation.append(user_message)
        
        # Get model response
        max_tokens = prompt.get('max_new_tokens', 256)
        response = query_model(conversation, max_new_tokens=max_tokens)
        
        print(f"\nResponse:\n{response}\n")
        
        # Add assistant response to conversation
        assistant_message = {
            "role": "assistant",
            "content": [{"type": "text", "text": response}]
        }
        conversation.append(assistant_message)
    
    return conversation


def process_independent_prompts(prompts_list):
    """
    Process prompts independently (no conversation context between prompts)
    
    Args:
        prompts_list: List of prompt dictionaries
    
    Returns:
        List of individual conversations
    """
    all_conversations = []
    
    for i, prompt in enumerate(prompts_list):
        print(f"\n{'='*60}")
        print(f"Processing Independent Prompt {i+1}/{len(prompts_list)}")
        print(f"ID: {prompt.get('id', 'N/A')}")
        print(f"Text: {prompt['text'][:100]}..." if len(prompt['text']) > 100 else f"Text: {prompt['text']}")
        print('='*60)
        
        # Build content for this turn
        content = []
        
        # Add images if provided
        if 'images' in prompt and prompt['images']:
            for img_path in prompt['images']:
                content.append({"type": "image", "image": img_path})
                print(f"  - Image: {img_path}")
        
        # Add text
        content.append({"type": "text", "text": prompt['text']})
        
        # Create a fresh conversation for this prompt
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        # Get model response
        max_tokens = prompt.get('max_new_tokens', 256)
        response = query_model(messages, max_new_tokens=max_tokens)
        
        print(f"\nResponse:\n{response}\n")
        
        # Save the complete conversation
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": response}]
        })
        
        all_conversations.append({
            "prompt_id": prompt.get('id', f'prompt_{i+1}'),
            "conversation": messages
        })
    
    return all_conversations


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    # Check if prompts file exists
    if not Path(PROMPTS_FILE).exists():
        print(f"Error: {PROMPTS_FILE} not found!")
        print("Creating a sample prompts file...")
        
        # Create sample prompts file
        sample_prompts = {
            "prompts": [
                {
                    "id": "beach_description",
                    "images": ["https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"],
                    "text": "Describe what you see in this image in detail.",
                    "max_new_tokens": 256
                },
                {
                    "id": "mood_analysis",
                    "images": [],
                    "text": "What emotions or mood does this scene convey?",
                    "max_new_tokens": 128
                },
                {
                    "id": "creative_writing",
                    "images": [],
                    "text": "Write a short haiku inspired by this scene.",
                    "max_new_tokens": 100
                }
            ]
        }
        
        with open(PROMPTS_FILE, 'w') as f:
            json.dump(sample_prompts, f, indent=2)
        
        print(f"Sample {PROMPTS_FILE} created. Please edit it with your prompts and run again.")
        exit()
    
    # Load prompts
    print(f"Loading prompts from {PROMPTS_FILE}...")
    prompts = load_prompts(PROMPTS_FILE)
    print(f"Loaded {len(prompts)} prompts\n")
    
    # Choose processing mode
    print("Choose processing mode:")
    print("1. Sequential (maintains conversation context)")
    print("2. Independent (each prompt is separate)")
    mode = input("Enter mode (1 or 2, default=1): ").strip() or "1"
    
    if mode == "1":
        # Process prompts in sequence (conversation context maintained)
        conversation = process_prompts_sequence(prompts)
        save_conversation(conversation, OUTPUT_FILE)
    else:
        # Process prompts independently
        conversations = process_independent_prompts(prompts)
        
        # Save all independent conversations
        if Path(OUTPUT_FILE).exists():
            with open(OUTPUT_FILE, 'r') as f:
                data = json.load(f)
        else:
            data = {"conversations": []}
        
        data["conversations"].append({
            "timestamp": datetime.now().isoformat(),
            "mode": "independent",
            "conversations": conversations
        })
        
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nAll conversations saved to {OUTPUT_FILE}")
    
    print("\n" + "="*60)
    print("Processing complete!")
    print("="*60)