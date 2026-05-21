"""
WhisperEngine — local on-device transcription via HuggingFace Transformers.

Same tech stack as insanely-fast-whisper (Whisper Large v3, Flash Attn 2,
batched inference) but exposed as a clean poly-stt engine.

Install:
    pip install "poly-stt[whisper]"
    # pulls: transformers, torch, accelerate
"""

from __future__ import annotations

from typing import Iterator, Literal
import numpy as np

from .base import STTEngine


class WhisperEngine(STTEngine):
    """
    On-device Whisper transcription.

    Models (fastest → most accurate):
        distil-whisper/distil-small.en       English only, very fast
        distil-whisper/distil-medium.en      English only, balanced
        distil-whisper/distil-large-v2       Multilingual, fast
        openai/whisper-large-v3-turbo        Best quality/speed balance
        openai/whisper-large-v3             Best quality

    Args:
        model:           HuggingFace model ID.
        device:          "auto" | "cpu" | "mps" | "cuda"
                         "auto" picks MPS on Mac, CUDA if available, else CPU.
        dtype:           "float16" | "float32"
                         float16 is faster on GPU/MPS. Use float32 on CPU.
        batch_size:      Higher = faster on GPU. Use 4 on Mac, 24 on A100.
        language:        Force language (e.g. "english"). None = auto-detect.

    Example:
        engine = WhisperEngine(model="distil-whisper/distil-large-v2", device="mps")
        stt = PolySTT(engine=engine)
        text = stt.transcribe(pcm_bytes, sample_rate=16000)
    """

    def __init__(
        self,
        model: str = "openai/whisper-large-v3-turbo",
        device: str = "auto",
        dtype: str = "float16",
        batch_size: int = 4,
        language: str | None = None,
    ):
        self.model_id = model
        self.device = self._resolve_device(device)
        self.dtype = dtype
        self.batch_size = batch_size
        self.language = language
        self._pipe = None  # lazy load on first use

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda:0"
            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _load(self):
        if self._pipe is not None:
            return self._pipe
        try:
            import torch
            from transformers import pipeline
        except ImportError:
            raise ImportError(
                "WhisperEngine requires transformers and torch.\n"
                "Install with: pip install 'poly-stt[whisper]'"
            )

        dtype_map = {"float16": torch.float16, "float32": torch.float32}
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model_id,
            torch_dtype=dtype_map.get(self.dtype, torch.float16),
            device=self.device,
        )
        return self._pipe

    def _to_numpy(self, audio: bytes, sample_rate: int) -> dict:
        arr = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        return {"array": arr, "sampling_rate": sample_rate}

    def transcribe(self, audio: bytes, sample_rate: int = 16000, **kwargs) -> str:
        pipe = self._load()
        inputs = self._to_numpy(audio, sample_rate)
        gen_kwargs = {}
        if self.language:
            gen_kwargs["language"] = self.language
        result = pipe(
            inputs,
            batch_size=self.batch_size,
            generate_kwargs=gen_kwargs or None,
        )
        return result.get("text", "").strip()

    def stream_transcribe(
        self, audio: bytes, sample_rate: int = 16000, **kwargs
    ) -> Iterator[str]:
        # Whisper is batch-only — yield the full result once
        yield self.transcribe(audio, sample_rate, **kwargs)
