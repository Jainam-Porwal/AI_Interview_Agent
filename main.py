import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix OpenMP conflict on Windows

import pyaudio
import numpy as np
from faster_whisper import WhisperModel

# =========================
# CONFIG
# =========================
MODEL_SIZE = "small"   # tiny, base, small, medium, large-v3
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 10   # bigger = more context, better accuracy

# =========================
# LOAD MODEL
# =========================
model = WhisperModel(
    MODEL_SIZE,
    device="cuda",         # GPU enabled
    compute_type="float16" # float16 for GPU (faster + accurate)
)

# =========================
# AUDIO SETUP
# =========================
p = pyaudio.PyAudio()

stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK
)

print("[MIC] Listening... Speak now (Ctrl+C to stop)\n")

# =========================
# MAIN LOOP
# =========================
try:
    while True:
        frames = []

        # Collect audio chunk
        for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)

        # Convert to numpy
        audio_bytes = b"".join(frames)
        audio_np = np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768.0

        # Transcribe
        segments, info = model.transcribe(
            audio_np,
            language="en",   # change if needed
            beam_size=1      # faster
        )

        # Print output
        for segment in segments:
            text = segment.text.strip()
            if text:
                print("[TEXT]", text)

except KeyboardInterrupt:
    print("\n[STOPPED]")

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()