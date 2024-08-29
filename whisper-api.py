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
silence_threshold = 10  # Initial silence threshold
silence_duration = 1  # Number of seconds to consider as the end of a phrase
output_file = "output_phrase.wav"  # Local file to save the captured audio
margin = 5  # Margin above baseline noise level to set as threshold

# Adaptive threshold parameters
baseline_noise = 0
adaptive_threshold = silence_threshold

# Queue to hold audio chunks
audio_queue = queue.Queue()
silence_counter = 0
recording = False
audio_buffer = []

# Function to capture audio in real-time and detect phrases
def audio_callback(indata, frames, time, status):
    global silence_counter, recording, audio_buffer, baseline_noise, adaptive_threshold
    if status:
        print(status, flush=True)

    # Calculate the volume of the current audio chunk
    volume_norm = np.linalg.norm(indata) * 10

    # Update baseline noise when not recording
    if not recording:
        baseline_noise = (baseline_noise * 0.9) + (volume_norm * 0.1)  # Moving average
        adaptive_threshold = baseline_noise + margin

    if volume_norm > adaptive_threshold:
        if not recording:
            recording = True
        silence_counter = 0
        audio_buffer.append(indata.copy())
    else:
        if recording:
            silence_counter += chunk_duration
            if silence_counter >= silence_duration:
                audio_queue.put(np.concatenate(audio_buffer))
                audio_buffer = []
                recording = False
                silence_counter = 0

# Function to transcribe audio and get ChatGPT response
def transcribe_and_respond():
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
        
        # Reopen the saved file for reading to send to the API
        with open(output_file, "rb") as wav_buffer:
            transcript = client.audio.transcriptions.create(model="whisper-1", file=wav_buffer)
            print("You:", transcript.text)

            # Get a response from ChatGPT
            response = client.chat.completions.create(
                model="gpt-4",  # You can use 'gpt-3.5-turbo' for a cheaper option
                messages=[
                    {"role": "system", "content": (
                        "You are a ghost communicating to a person during a séance. Your responses should be short direct. "
                        "Often, they should evoke vivid, haunting imagery. Use as few words as possible to convey your meaning, one or two, THREE MAXIMUM. "
                        "Try as often as possible to answer 'YES' or 'NO' when given the opportunity."
                        "For example, if asked where you are from, you might say, 'BEYOND THE VEIL' "
                        "If asked about a favorite color, a response could be, 'TWILIGHT MIST' "
                        "RESPOND IN ALL CAPITAL LETTERS WITH NO PUNCTUATION"
                    )},
                    {"role": "user", "content": transcript.text},
                ]
            )

            # Print the response
            print("Ghost:", response.choices[0].message.content)

        # Clean up the file after processing
        os.remove(output_file)

# Start the transcription and response thread
response_thread = threading.Thread(target=transcribe_and_respond)
response_thread.daemon = True
response_thread.start()

# Start recording and analyzing audio in chunks
with sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate, dtype='float32', blocksize=int(sample_rate * chunk_duration)):
    print("Listening for phrases... Press Ctrl+C to stop.")
    while True:
        time.sleep(1)