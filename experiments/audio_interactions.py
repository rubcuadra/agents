import sounddevice as sd
import numpy as np
import whisper
import scipy.io.wavfile as wavfile
import os
import tempfile
from datetime import datetime

def record_and_transcribe_voice_note_with_timestamp(duration=5, samplerate=16000):
    """
    Records a short voice note, saves it with a timestamped filename in a temp folder,
    transcribes it using OpenAI's Whisper, and identifies the spoken language.

    Args:
        duration (int): The duration of the recording in seconds.
        samplerate (int): The sample rate for the audio recording (Whisper prefers 16000 Hz).
    """
    # Create a timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Use tempfile to get a temporary directory and create the full file path
    temp_dir = tempfile.gettempdir()
    filename = os.path.join(temp_dir, f"voice_note_{timestamp}.wav")

    print(f"Recording a voice note for {duration} seconds...")
    print(f"Audio will be temporarily saved as '{filename}'")

    try:
        # Record audio
        audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
        sd.wait()  # Wait until recording is finished
        print("Recording finished.")

        # Save the recorded audio to the temporary WAV file
        wavfile.write(filename, samplerate, (audio_data * 32767).astype(np.int16)) # Convert to int16 for WAV
        print(f"Voice note saved temporarily.")

        # Load the Whisper model
        print("Loading Whisper model (this may take a moment)...")
        model = whisper.load_model("base") # Consider 'base' for faster processing or 'small'/'medium' for better accuracy
        print("Whisper model loaded.")

        # Transcribe the audio
        print("Transcribing audio...")
        result = model.transcribe(filename)

        # Print the transcription and detected language
        print("\n--- Transcription Result ---")
        print(f"Transcription: {result['text']}")
        print(f"Detected Language: {result['language']}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Clean up the recorded file from the temporary directory
        if os.path.exists(filename):
            os.remove(filename)
            print(f"Cleaned up temporary file '{filename}'")

if __name__ == "__main__":
    # Record for 7 seconds by default
    record_and_transcribe_voice_note_with_timestamp(duration=7)