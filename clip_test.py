import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# ============================================================================
# CONFIGURATION
# ============================================================================

# Image path
IMAGE_PATH = "/home/hliu852/VLM for RL project/VLM-for-RL/test video/mug_history/04A.jpg"

# Text prompts
GOAL_DESCRIPTION = "No robotic grippers grasping a mug"
BASELINE_DESCRIPTION = "robotic grippers"

# Alpha values to test
ALPHA_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# ============================================================================
# LOAD MODEL AND IMAGE
# ============================================================================

print("Loading CLIP model...")

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", use_safetensors=True)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

print(f"Loading image: {IMAGE_PATH}")
try:
    image = Image.open(IMAGE_PATH)
    print(f"✓ Image loaded successfully ({image.size})\n")
except FileNotFoundError:
    print(f"✗ Error: Image not found at {IMAGE_PATH}")
    exit()

# ============================================================================
# COMPUTE EMBEDDINGS
# ============================================================================

# Process inputs
text_inputs = processor(text=[GOAL_DESCRIPTION, BASELINE_DESCRIPTION], 
                       return_tensors="pt", padding=True)
image_inputs = processor(images=image, return_tensors="pt")

# Get embeddings
with torch.no_grad():
    text_outputs = model.text_model(**text_inputs)
    text_embeds = model.text_projection(text_outputs.pooler_output)
    
    image_outputs = model.vision_model(**image_inputs)
    image_embeds = model.visual_projection(image_outputs.pooler_output)

# Normalize embeddings
goal_embed = text_embeds[0] / text_embeds[0].norm(p=2, dim=-1, keepdim=True)
baseline_embed = text_embeds[1] / text_embeds[1].norm(p=2, dim=-1, keepdim=True)
image_embed = image_embeds[0] / image_embeds[0].norm(p=2, dim=-1, keepdim=True)

# ============================================================================
# VLM-RM REWARD FUNCTION (EXACT IMPLEMENTATION)
# ============================================================================

def compute_vlmrm_reward(image_embed, goal_embed, baseline_embed, alpha):
    """
    Compute VLM-RM Goal-Baseline Regularized reward.
    
    Formula: y = 1 - 0.5 * ||(x - target) @ P||²
    where P = alpha * projection_matrix + (1 - alpha) * identity
    
    Args:
        image_embed: Normalized image embedding
        goal_embed: Normalized goal text embedding
        baseline_embed: Normalized baseline text embedding
        alpha: Regularization strength [0, 1]
    
    Returns:
        reward: Scalar reward value
    """
    # Compute direction from baseline to goal
    direction = goal_embed - baseline_embed
    
    # Compute projection matrix onto the direction
    proj_matrix = torch.outer(direction, direction) / (direction.norm() ** 2)
    
    # Identity matrix
    identity = torch.eye(direction.shape[0], device=direction.device)
    
    # Weighted projection matrix
    projection = alpha * proj_matrix + (1 - alpha) * identity
    
    # Compute reward: 1 - 0.5 * ||(x - goal) @ P||²
    diff = image_embed - goal_embed
    projected_diff = torch.matmul(diff, projection)
    reward = 1 - 0.5 * (projected_diff.norm() ** 2).item()
    
    return reward

# ============================================================================
# COMPUTE BASIC SIMILARITIES
# ============================================================================

sim_goal = torch.matmul(image_embed, goal_embed.t()).item()
sim_baseline = torch.matmul(image_embed, baseline_embed.t()).item()

# ============================================================================
# COMPUTE DIRECTION ALIGNMENT
# ============================================================================

direction = goal_embed - baseline_embed
direction_normalized = direction / direction.norm()
direction_alignment = torch.matmul(image_embed, direction_normalized.t()).item()

# ============================================================================
# DISPLAY RESULTS
# ============================================================================

print("=" * 80)
print("CLIP REWARD ANALYSIS")
print("=" * 80)

print("\n📝 PROMPTS:")
print(f"   Goal:     '{GOAL_DESCRIPTION}'")
print(f"   Baseline: '{BASELINE_DESCRIPTION}'")

print("\n📊 STANDARD CLIP SIMILARITIES (No Regularization):")
print(f"   Image ↔ Goal:     {sim_goal:.4f}")
print(f"   Image ↔ Baseline: {sim_baseline:.4f}")
print(f"   Difference:       {sim_goal - sim_baseline:.4f} ", end="")

print("\n🎯 DIRECTION ALIGNMENT:")
print(f"   Alignment with (goal - baseline): {direction_alignment:.4f}")
if direction_alignment > 0.1:
    print("   → Strong positive alignment (moving toward goal) ✓")
elif direction_alignment > 0:
    print("   → Weak positive alignment (slightly toward goal) ~")
elif direction_alignment > -0.1:
    print("   → Near zero alignment (perpendicular to change) ~")
else:
    print("   → Negative alignment (moving away from goal) ✗")

print("\n🔧 GOAL-BASELINE REGULARIZED REWARDS:")
print("   " + "-" * 50)
print("   α (alpha) │ Reward  │ Change from α=0")
print("   " + "-" * 50)

baseline_reward = compute_vlmrm_reward(image_embed, goal_embed, baseline_embed, 0.0)

for alpha in ALPHA_VALUES:
    reward = compute_vlmrm_reward(image_embed, goal_embed, baseline_embed, alpha)
    change = reward - baseline_reward
    change_str = f"+{change:.4f}" if change >= 0 else f"{change:.4f}"
    
    # Add indicators for special values
    if alpha == 0.0:
        note = " (standard CLIP)"
    elif alpha == 1.0:
        note = " (full projection)"
    elif 0.4 <= alpha <= 0.6:
        note = " (recommended)"
    else:
        note = ""
    
    print(f"   {alpha:5.1f}     │ {reward:.4f}  │ {change_str}    {note}")

print("   " + "-" * 50)

# Find optimal alpha
optimal_alpha = max(ALPHA_VALUES, 
                   key=lambda a: compute_vlmrm_reward(image_embed, goal_embed, baseline_embed, a))
optimal_reward = compute_vlmrm_reward(image_embed, goal_embed, baseline_embed, optimal_alpha)

print(f"\n   Highest reward: {optimal_reward:.4f} at α = {optimal_alpha:.1f}")
