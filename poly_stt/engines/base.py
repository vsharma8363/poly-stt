from abc import ABC, abstractmethod
from typing import Iterator


class STTEngine(ABC):
    """
    Abstract base class for all STT engines.
    Subclass this and implement transcribe + stream_transcribe
    to plug in any provider.
    """

    @abstractmethod
    def transcribe(self, audio: bytes, sample_rate: int = 16000, **kwargs) -> str:
        """
        Transcribe audio bytes and return the full transcript as a string.
        Blocks until complete.

        Args:
            audio:       Raw PCM16 audio bytes (mono).
            sample_rate: Sample rate of the audio in Hz.
            **kwargs:    Engine-specific options (language, etc.)

        Returns:
            Transcript string.
        """
        ...

    @abstractmethod
    def stream_transcribe(
        self, audio: bytes, sample_rate: int = 16000, **kwargs
    ) -> Iterator[str]:
        """
        Transcribe audio and yield text chunks as they arrive.
        For batch models this may yield once. For streaming-native
        providers (e.g. Deepgram) this yields partial results in real time.

        Args:
            audio:       Raw PCM16 audio bytes (mono).
            sample_rate: Sample rate of the audio in Hz.
            **kwargs:    Engine-specific options.

        Yields:
            Text chunks (partial or final).
        """
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__
