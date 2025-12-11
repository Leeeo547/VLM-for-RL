import torch
import torch.nn as nn
from transformers import AutoModel

model_path = "OpenGVLab/InternVideo2_CLIP_S"
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).cuda().eval()

def list_projection_candidates(in_dim=768, out_dim=512):
    cands = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            if getattr(m, "in_features", None) == in_dim and getattr(m, "out_features", None) == out_dim:
                # exclude encoder block mlp
                if any(x in name.lower() for x in ["blocks", "mlp", "attn"]):
                    continue
                if any(k in name.lower() for k in ["proj", "projection", "visual", "vision", "text", "embed"]):
                    cands.append((name, m))
    return cands

cands = list_projection_candidates(768, 512)
print("=== Projection candidates (768 -> 512) ===")
for n, m in cands:
    print(n, "(", m.in_features, "->", m.out_features, ")")
