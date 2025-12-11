import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

# ----------------------------
# 1. Load InternVideo2.5 model
# ----------------------------
model_path = "OpenGVLab/InternVideo2_5_Chat_8B"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_path, trust_remote_code=True
).half().cuda().to(torch.bfloat16)
model.eval()

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments

    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)

    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)

    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        img_tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)

        pixel_values = [transform(tile) for tile in img_tiles]
        pixel_values = torch.stack(pixel_values)  # (n_tiles, 3, H, W)

        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)

    pixel_values = torch.cat(pixel_values_list, dim=0)  # (sum_tiles, 3, H, W)
    return pixel_values, num_patches_list

# ---------------------------------------
# 2. Encode video & text + cosine reward
# ---------------------------------------
@torch.no_grad()
def encode_video(pixel_values, num_patches_list):
    """
    Try multiple API names to get video embedding.
    """
    pixel_values = pixel_values.to(torch.bfloat16).to(model.device)

    # CLIP-style common APIs
    if hasattr(model, "get_video_features"):
        vid_emb = model.get_video_features(pixel_values, num_patches_list=num_patches_list)
    elif hasattr(model, "encode_video"):
        vid_emb = model.encode_video(pixel_values, num_patches_list=num_patches_list)
    elif hasattr(model, "get_visual_features"):
        # some InternVideo checkpoints use this
        vid_emb = model.get_visual_features(pixel_values, num_patches_list=num_patches_list)
    else:
        raise AttributeError(
            "This checkpoint doesn't expose a direct video-embedding API. "
            "Consider using InternVideo2.5 CLIP/retrieval checkpoint, or inspect model's forward() "
            "to locate video encoder output."
        )

    # ensure shape: (B, D)
    if vid_emb.dim() == 1:
        vid_emb = vid_emb.unsqueeze(0)
    elif vid_emb.dim() == 3:
        # (B, T, D) -> mean pool over T
        vid_emb = vid_emb.mean(dim=1)

    return vid_emb

@torch.no_grad()
def encode_text(text):
    """
    Try multiple API names to get text embedding.
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    if hasattr(model, "get_text_features"):
        txt_emb = model.get_text_features(**inputs)
    elif hasattr(model, "encode_text"):
        txt_emb = model.encode_text(**inputs)
    elif hasattr(model, "get_language_features"):
        txt_emb = model.get_language_features(**inputs)
    else:
        raise AttributeError(
            "This checkpoint doesn't expose a direct text-embedding API. "
            "Consider using InternVideo2.5 CLIP/retrieval checkpoint."
        )

    if txt_emb.dim() == 1:
        txt_emb = txt_emb.unsqueeze(0)
    elif txt_emb.dim() == 3:
        txt_emb = txt_emb.mean(dim=1)

    return txt_emb

@torch.no_grad()
def video_text_cosine_similarity(video_path, task_description, num_segments=128):
    pixel_values, num_patches_list = load_video(
        video_path,
        num_segments=num_segments,
        max_num=1
    )

    vid_emb = encode_video(pixel_values, num_patches_list)
    txt_emb = encode_text(task_description)

    # normalize then cosine
    vid_emb = F.normalize(vid_emb, dim=-1)
    txt_emb = F.normalize(txt_emb, dim=-1)

    sim = (vid_emb * txt_emb).sum(dim=-1)  # cosine similarity
    return sim.item(), vid_emb, txt_emb

# ----------------------------
# 3. Example
# ----------------------------
if __name__ == "__main__":
    video_path = "test_video/cup.mp4"
    task_description = "a robot arm picks up a cup"

    sim, vid_emb, txt_emb = video_text_cosine_similarity(
        video_path, task_description, num_segments=128
    )

    print("Cosine similarity (reward):", sim)
    print("Video embedding shape:", vid_emb.shape)
    print("Text embedding shape:", txt_emb.shape)
