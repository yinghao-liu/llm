# annotate/__init__.py
from .pipeline import annotate_shots, annotate_video
from .subtitle import parse_srt, align_dialogue
from .vlm import VLMClient

__all__ = ["annotate_shots", "annotate_video", "parse_srt",
           "align_dialogue", "VLMClient"]