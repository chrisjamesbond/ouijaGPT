import sounddevice as sd
import numpy as np
from openai import OpenAI
import queue
import threading
import time
import io
import wave

# Initialize the OpenAI client
client = OpenAI()

# Parameters
sample_rate = 16000  # Whisper API uses 16 kHz audio
chunk_duration = 5  # Each audio chunk will be 5 seconds long
output_file = "output_audio.wav"  # Local file to save the captured audio

# Queue to hold audio chunks
audio_queue = queue.Queue()

# Function to capture audio in real-time
def audio_callback(indata, frames, time, status):
    if status:
        print(status, flush=True)
    audio_queue.put(indata.copy())

# Function to transcribe audio using the Whisper API
def transcribe_audio():
    print("Starting transcription...")
    
    while True:
        audio_data = audio_queue.get()

        # Normalize audio data to 16-bit PCM
        audio_data = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)

        # Save the audio data to a local WAV file
        with wave.open(output_file, "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit audio
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        print(f"Saved audio to {output_file}")
        
        # Reopen the saved file for reading to send to the API
        with open(output_file, "rb") as wav_buffer:
            # Send the WAV file to the Whisper API
            transcript = client.audio.transcriptions.create(model="whisper-1", file=wav_buffer)
            print("Transcription:", transcript.text)

# Start the transcription thread
transcription_thread = threading.Thread(target=transcribe_audio)
transcription_thread.daemon = True
transcription_thread.start()

# Start recording with a slightly longer chunk duration
with sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate, dtype='float32', blocksize=int(sample_rate * chunk_duration)):
    print("Listening... Press Ctrl+C to stop.")
    while True:
        time.sleep(1)