import whisper

# Load the pre-trained Whisper model
model = whisper.load_model("base")

# Transcribe an audio file
result = model.transcribe("test.wav")

# Print the transcription
print(result["text"])