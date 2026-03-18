import os
from pathlib import Path


_CHECKPOINT_ROOT = Path(os.getenv("VIPO_CHECKPOINT_ROOT", "checkpoints"))

reward_model_path = {
    "video_align": os.getenv(
        "VIDEOALIGN_CHECKPOINT",
        str(_CHECKPOINT_ROOT / "videoalign"),
    ),
    "viclip": os.getenv(
        "VICLIP_CHECKPOINT",
        str(_CHECKPOINT_ROOT / "viclip" / "ViClip-InternVid-10M-FLT.pth"),
    ),
    "mps": os.getenv(
        "MPS_CHECKPOINT",
        str(_CHECKPOINT_ROOT / "mps" / "MPS_overall_checkpoint.pth"),
    ),
    "pick": os.getenv(
        "PICKSCORE_CHECKPOINT",
        str(_CHECKPOINT_ROOT / "pickscore"),
    ),
    "clip": os.getenv(
        "CLIP_CHECKPOINT",
        str(_CHECKPOINT_ROOT / "clip"),
    ),
}
