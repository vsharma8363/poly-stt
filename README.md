# poly-stt

> Speech-to-text for the [`poly-` ecosystem](https://github.com/vikramsharma/poly-bidi). Plug in any engine. Get text out.

```python
from poly_stt import PolySTT, WhisperEngine

stt = PolySTT(WhisperEngine(model="distil-whisper/distil-large-v2"))
text = stt.transcribe(pcm_bytes, sample_rate=16000)
print(text)  # "hey what's up"
```

---

## Part of the `poly-` ecosystem

`poly-stt` is one piece of a larger vision:

```
poly-tts   →  text in,  audio out         (speak)
poly-stt   →  audio in, text out          (listen)
poly-bidi  →  bidirectional streaming     (converse)
```

The goal: make it trivially easy to give any Python program — an AI agent,
a script, a toy chatbot — a real voice interface. Like giving it a phone number.

```python
# The dream: three lines to talk to anything
server = SpeechBiDiServer(brain=MyAgent(), tts=PolyTTS(...), stt=PolySTT(...))
client = MicSpeakerClient()   # or TwilioClient(), or BrowserClient()
client.connect("ws://localhost:8765")
```

---

## Install

```bash
pip install poly-stt

# With on-device Whisper (recommended)
pip install "poly-stt[whisper]"
```

---

## Engines

| Engine | Extra | Backend | Notes |
|---|---|---|---|
| `WhisperEngine` | `whisper` | HuggingFace Transformers | On-device. Works offline. MPS on Mac, CUDA on GPU, CPU fallback. |
| `DeepgramEngine` *(soon)* | `deepgram` | Deepgram API | Streaming-native, word timestamps, very low latency |
| `OpenAIWhisperEngine` *(soon)* | `openai` | OpenAI API | Cloud Whisper, dead simple |

---

## Usage

### Transcribe raw audio bytes

```python
from poly_stt import PolySTT, WhisperEngine

stt = PolySTT(WhisperEngine())
text = stt.transcribe(pcm_bytes, sample_rate=16000)
```

### Transcribe a file

```python
text = stt.transcribe_file("recording.wav")
text = stt.transcribe_file("interview.mp3")   # requires soundfile
```

### Streaming (yields as results arrive)

```python
for chunk in stt.stream_transcribe(pcm_bytes):
    print(chunk, end="", flush=True)
```

Batch engines (like Whisper) yield once when done. Streaming-native engines
(like Deepgram) yield partial results in real time as you speak.

### Hot-swap engines at runtime

```python
stt = PolySTT(WhisperEngine())
# ... later ...
stt.engine = DeepgramEngine(api_key="...")   # swap without restarting
```

### Bring your own engine

```python
from poly_stt import STTEngine, PolySTT

class MyEngine(STTEngine):
    def transcribe(self, audio: bytes, sample_rate: int = 16000, **kwargs) -> str:
        return my_transcription_service(audio)

    def stream_transcribe(self, audio: bytes, sample_rate: int = 16000, **kwargs):
        yield self.transcribe(audio, sample_rate)

stt = PolySTT(MyEngine())
```

---

## WhisperEngine models

| Model | Speed | Quality | Language |
|---|---|---|---|
| `distil-whisper/distil-small.en` | ⚡⚡⚡ | ★★☆ | English only |
| `distil-whisper/distil-medium.en` | ⚡⚡⚡ | ★★★ | English only |
| `distil-whisper/distil-large-v2` | ⚡⚡ | ★★★★ | Multilingual |
| `openai/whisper-large-v3-turbo` | ⚡⚡ | ★★★★★ | Multilingual |
| `openai/whisper-large-v3` | ⚡ | ★★★★★ | Multilingual |

```python
# Fast + good enough for most use cases
WhisperEngine(model="distil-whisper/distil-large-v2", device="mps")   # Mac
WhisperEngine(model="distil-whisper/distil-large-v2", device="cuda")  # GPU
WhisperEngine(model="distil-whisper/distil-large-v2", device="cpu")   # anywhere
```

First run downloads the model from HuggingFace (~750MB for distil-large-v2,
~3GB for large-v3). Cached after that.

---

## The bigger picture

`poly-stt` + `poly-tts` + `poly-bidi` = a complete voice I/O stack for any
Python application. The architecture is deliberately simple:

- **No framework to adopt.** Drop these libraries in like you'd drop in `requests`.
- **No platform lock-in.** Your brain/agent/logic is just a function. Everything
  else is swappable.
- **Runs local.** On-device Whisper means no API keys, no latency, no data leaving
  your machine. 

The end game is something like this being real:

```python
# Call your friend. Have your AI talk to them.
server = SpeechBiDiServer(brain=ClaudeAgent(), tts=PolyTTS(...), stt=PolySTT(...))
TwilioClient(to="+14155551234").connect(server)
```

```python
# Or just... talk to your AI on your laptop like it's in the room.
MicSpeakerClient().connect(server)
```

```python
# Or video chat with it.
WebcamClient().connect(server)   # frames → vision model → voice response
```

That's the dream. This library is the ear.

---

## License

MIT
