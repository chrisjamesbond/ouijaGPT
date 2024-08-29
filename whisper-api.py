import sounddevice as sd
import numpy as np
from openai import OpenAI
import queue
import threading
import time
import wave
import os

# Initialize the OpenAI client
client = OpenAI()

# Parameters
sample_rate = 16000  # Whisper API uses 16 kHz audio
chunk_duration = 0.5  # Analyze audio in 0.5-second chunks
silence_threshold = 10  # Adjust this value based on your environment
silence_duration = 1  # Number of seconds to consider as the end of a phrase
output_file = "output_phrase.wav"  # Local file to save the captured audio

# Queue to hold audio chunks
audio_queue = queue.Queue()
silence_counter = 0
recording = False
audio_buffer = []

# Function to capture audio in real-time and detect phrases
def audio_callback(indata, frames, time, status):
    global silence_counter, recording, audio_buffer
    if status:
        print(status, flush=True)

    # Calculate the volume of the current audio chunk
    volume_norm = np.linalg.norm(indata) * 10

    if volume_norm > silence_threshold:
        if not recording:
            #print("Starting capture...")
            recording = True
        silence_counter = 0
        audio_buffer.append(indata.copy())
    else:
        if recording:
            silence_counter += chunk_duration
            if silence_counter >= silence_duration:
                #print("Phrase detected. Saving to file and sending to API...")
                audio_queue.put(np.concatenate(audio_buffer))
                audio_buffer = []
                recording = False
                silence_counter = 0

# Function to transcribe audio using the Whisper API
def transcribe_audio():
    #print("Starting transcription...")
    
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

        #print(f"Saved audio to {output_file}")
        
        # Reopen the saved file for reading to send to the API
        with open(output_file, "rb") as wav_buffer:
            # Send the WAV file to the Whisper API
            transcript = client.audio.transcriptions.create(model="whisper-1", file=wav_buffer)
            print(transcript.text)

        # Optionally, clean up the file after processing
        os.remove(output_file)

# Start the transcription thread
transcription_thread = threading.Thread(target=transcribe_audio)
transcription_thread.daemon = True
transcription_thread.start()

# Start recording and analyzing audio in chunks
with sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate, dtype='float32', blocksize=int(sample_rate * chunk_duration)):
    print("Listening for phrases... Press Ctrl+C to stop.")
    while True:
        time.sleep(1)