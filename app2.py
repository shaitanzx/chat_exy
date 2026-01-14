# File: server_gradio.py
# Полная замена веб-интерфейса на Gradio с сохранением оригинальных названий функций
# Основано на server.py, script.js и index.html

import os
from pathlib import Path
import gradio as gr
import torch
import numpy as np
import tempfile
import time
import json
import yaml
import logging
from typing import Dict, List, Any, Optional, Tuple
import io
import shutil
import uuid
import librosa
import unicodedata
import re
from datetime import datetime

# --- ОРИГИНАЛЬНЫЕ ИМПОРТЫ ИЗ SERVER.PY ---
# Импортируем config_manager ПЕРВЫМ ДЕЛОМ
from config import (
    config_manager,
    get_host,
    get_port,
    get_log_file_path,
    get_output_path,
    get_reference_audio_path,
    get_predefined_voices_path,
    get_ui_title,
    get_gen_default_temperature,
    get_gen_default_exaggeration,
    get_gen_default_cfg_weight,
    get_gen_default_seed,
    get_gen_default_speed_factor,
    get_gen_default_language,
    get_audio_sample_rate,
    get_full_config_for_template,
    get_audio_output_format,
)
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

from chatterbox.tts import ChatterboxTTS  # Main TTS engine class
from chatterbox.models.s3gen.const import (
    S3GEN_SR,
)  # Default sample rate from the engine

from chatterbox.models.t3 import T3
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.s3gen import S3Gen
from chatterbox.models.voice_encoder import VoiceEncoder
from chatterbox.models.tokenizers import MTLTokenizer
from chatterbox.mtl_tts import Conditionals, SUPPORTED_LANGUAGES # Need to import these too
from chatterbox.vc import ChatterboxVC

VC_MODEL = None

os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

# Select device: Apple Silicon GPU (MPS) if available, else fallback to CPU
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"🚀 Running on device: {DEVICE}")
# ---- Determinism (CUDA / PyTorch) ----
import os as _os, torch as _torch
_torch.backends.cudnn.benchmark = False
if hasattr(_torch.backends.cudnn, "deterministic"):
    _torch.backends.cudnn.deterministic = True
try:
    _torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass
_os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
if DEVICE == "cuda":
    _torch.backends.cuda.matmul.allow_tf32 = False
    _torch.backends.cudnn.allow_tf32 = False
# --------------------------------------


def get_or_load_model():
    global MODEL
    if MODEL is None:
        print("Model not loaded, initializing...")
        MODEL = ChatterboxTTS.from_pretrained(DEVICE)
        if hasattr(MODEL, 'to') and str(MODEL.device) != DEVICE:
            MODEL.to(DEVICE)
        if hasattr(MODEL, "eval"):
            MODEL.eval()
        print(f"Model loaded on device: {getattr(MODEL, 'device', 'unknown')}")
    return MODEL


MODEL = None

try:
    get_or_load_model()
except Exception as e:
    print(f"CRITICAL: Failed to load model. Error: {e}")


















