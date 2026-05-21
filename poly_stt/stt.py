from typing import Iterator
from .engines.base import STTEngine


class PolySTT:
    """
    Central STT interface. Pass any STTEngine at init and use the same
    API regardless of provider.

    Usage:
        engine = WhisperEngine(model="distil-whisper/distil-large-v2", device="mps")
        stt = PolySTT(engine=engine)

        # Full transcript
        text = stt.transcribe(pcm_bytes, sample_rate=16000)

        # Streaming (yields as results arrive)
        for chunk in stt.stream_transcribe(pcm_bytes):
            print(chunk, end="", flush=True)
    """

    def __init__(self, engine: STTEngine):
        if not isinstance(engine, STTEngine):
            raise TypeError(f"engine must be an STTEngine subclass, got {type(engine)}")
        self._engine = engine

    @property
    def engine(self) -> STTEngine:
        return self._engine

    @engine.setter
    def engine(self, new_engine: STTEngine):
        """Hot-swap the engine at runtime."""
        if not isinstance(new_engine, STTEngine):
            raise TypeError(f"engine must be an STTEngine subclass, got {type(new_engine)}")
        self._engine = new_engine

    def transcribe(self, audio: bytes, sample_rate: int = 16000, **kwargs) -> str:
        """
        Transcribe audio bytes and return the full transcript.
        Blocks until complete.

        Args:
            audio:       Raw PCM16 audio bytes (mono).
            sample_rate: Sample rate in Hz. Default 16000.
            **kwargs:    Passed through to the engine.

        Returns:
            Transcript string.
        """
        if not audio:
            raise ValueError("audio cannot be empty")
        return self._engine.transcribe(audio, sample_rate, **kwargs)

    def stream_transcribe(
        self, audio: bytes, sample_rate: int = 16000, **kwargs
    ) -> Iterator[str]:
        """
        Transcribe audio and yield text as it arrives.
        Batch engines yield once. Streaming engines yield partial results.

        Args:
            audio:       Raw PCM16 audio bytes (mono).
            sample_rate: Sample rate in Hz.
            **kwargs:    Passed through to the engine.

        Yields:
            Text chunks.
        """
        if not audio:
            raise ValueError("audio cannot be empty")
        yield from self._engine.stream_transcribe(audio, sample_rate, **kwargs)

    def transcribe_file(self, path: str, **kwargs) -> str:
        """
        Convenience: read an audio file and transcribe it.
        Requires soundfile: pip install soundfile

        Args:
            path: Path to audio file (wav, mp3, flac, etc.)
        """
        try:
            import soundfile as sf
            import numpy as np
        except ImportError:
            raise ImportError("transcribe_file requires soundfile: pip install soundfile")

        data, sample_rate = sf.read(path, dtype="int16")
        if data.ndim > 1:
            data = data.mean(axis=1).astype(np.int16)
        return self.transcribe(data.tobytes(), sample_rate, **kwargs)

    def __repr__(self) -> str:
        return f"PolySTT(engine={self._engine.name!r})"
