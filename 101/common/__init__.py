# common/__init__.py
"""video_rag 公共契约层。

只放跨包共享的数据模型与通用工具，不依赖任何业务包。
使用方式二选一（推荐前者，全库统一即可）：
    from common.models import Shot, Document
    from common import Shot, Document
"""
from .models import (  # noqa: F401
    TRANS_BLACK,
    TRANS_CUT,
    TRANS_GRADUAL,
    TRANS_START,
    TRANSITION_TYPES,
    Document,
    Shot,
    ShotAnnotation,
    atomic_write_json,
    dump_jsonl,
    format_timecode,
    load_jsonl,
    read_json,
)

__all__ = [
    "Shot", "ShotAnnotation", "Document",
    "TRANS_START", "TRANS_CUT", "TRANS_GRADUAL", "TRANS_BLACK",
    "TRANSITION_TYPES",
    "format_timecode", "read_json", "atomic_write_json",
    "dump_jsonl", "load_jsonl",
]