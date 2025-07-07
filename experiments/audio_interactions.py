import sounddevice as sd
import numpy as np
import whisper
import scipy.io.wavfile as wavfile
import os
import tempfile
from datetime import datetime
import threading
import time
from pynput import keyboard

# Global variables for controlling recording
is_recording = False
audio_buffer = []
samplerate = 16000 # Whisper prefers 16000 Hz
stream = None
recording_thread = None
stop_event = threading.Event() # Event to signal the recording thread to stop

def audio_callback(indata, frames, time_info, status):
    """This is called (potentially in a separate thread) for each audio block."""
    global audio_buffer, is_recording
    if status:
        print(status)
    if is_recording:
        audio_buffer.append(indata.copy())

def start_recording_threaded(max_duration):
    """Starts the audio recording in a separate thread."""
    global is_recording, audio_buffer, stream, stop_event

    audio_buffer = [] # Clear buffer for new recording
    stop_event.clear() # Clear stop event for new recording

    try:
        with sd.InputStream(samplerate=samplerate, channels=1, dtype='float32', callback=audio_callback) as s:
            stream = s # Store the stream object to close it later
            is_recording = True
            print("Recording started (spacebar pressed). Speak now...")

            start_time = time.time()
            while is_recording and not stop_event.is_set() and (time.time() - start_time) < max_duration:
                time.sleep(0.1) # Check state periodically

            is_recording = False # Ensure flag is set to false
            if stream.active:
                stream.stop() # Explicitly stop the stream
            print("Recording stopped.")

    except Exception as e:
        print(f"Error during recording: {e}")
        is_recording = False # Ensure flag is reset on error

def on_press(key):
    """Callback for keyboard press events from pynput."""
    global is_recording, recording_thread, stop_event

    try:
        if key == keyboard.Key.space:
            if not is_recording:
                print("Spacebar pressed. Starting recording...")
                stop_event.clear() # Ensure stop event is clear for new recording
                recording_thread = threading.Thread(target=start_recording_threaded, args=(10,)) # Max 10 seconds
                recording_thread.start()
            elif is_recording:
                print("Spacebar pressed again. Stopping recording...")
                stop_event.set() # Signal the recording thread to stop
                # Give the recording thread a moment to finish
                if recording_thread and recording_thread.is_alive():
                    recording_thread.join(timeout=2)
                    if recording_thread.is_alive():
                        print("Warning: Recording thread did not terminate gracefully.")
        elif key == keyboard.KeyCode.from_char('q'): # Listen for 'q' to quit
            print("Quitting...")
            # Signal to stop any ongoing recording and then exit listener
            if is_recording:
                stop_event.set()
                if recording_thread and recording_thread.is_alive():
                    recording_thread.join(timeout=2)
            return False # This stops the pynput keyboard listener
    except AttributeError:
        # Handle special keys (like space, enter, shift, etc.) which don't have a .char attribute
        # print(f"Special key {key} pressed")
        pass


def transcribe_recorded_audio(filename):
    """Transcribes the audio file using Whisper."""
    print("Loading Whisper model (this may take a moment)...")
    # You can choose a different model size like 'base', 'small', 'medium', 'large'
    model = whisper.load_model("base")
    print("Whisper model loaded.")

    print("Transcribing audio...")
    result = model.transcribe(filename)

    # Print the transcription and detected language
    print("\n--- Transcription Result ---")
    print(f"Transcription: {result['text']}")
    print(f"Detected Language: {result['language']}")

def main():
    global audio_buffer, is_recording

    print("Press the SPACEBAR to start recording.")
    print("Press the SPACEBAR again to stop, or it will stop automatically after 10 seconds.")
    print("Press 'q' to quit at any time.")

    # Setup the pynput keyboard listener in the main thread
    # The listener will call on_press for key press events
    listener = keyboard.Listener(on_press=on_press)
    listener.start() # Start the listener thread

    # Keep the main thread alive to process audio and allow keyboard events
    try:
        while listener.is_alive(): # Keep running as long as the listener is active
            # Check if recording has stopped and buffer needs processing
            if not is_recording and audio_buffer:
                print("Processing recorded audio...")
                # Concatenate all recorded blocks
                recorded_audio = np.concatenate(audio_buffer, axis=0)

                # Create a timestamped filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                temp_dir = tempfile.gettempdir()
                filename = os.path.join(temp_dir, f"voice_note_{timestamp}.wav")

                try:
                    # Save the recorded audio to the temporary WAV file
                    wavfile.write(filename, samplerate, (recorded_audio * 32767).astype(np.int16)) # Convert to int16 for WAV
                    print(f"Voice note saved temporarily to '{filename}'")

                    transcribe_recorded_audio(filename)

                except Exception as e:
                    print(f"Error saving or transcribing audio: {e}")
                finally:
                    # Clean up the recorded file from the temporary directory
                    if os.path.exists(filename):
                        os.remove(filename)
                        print(f"Cleaned up temporary file '{filename}'")
                    audio_buffer.clear() # Clear buffer after processing
                    print("\nReady for next recording.")
            
            time.sleep(0.1) # Prevent busy-waiting

    except KeyboardInterrupt:
        print("Program interrupted by user (Ctrl+C).")
    finally:
        # Ensure listener and recording thread are stopped on exit
        if listener.is_alive():
            listener.stop() # Stop the pynput listener
        if is_recording: # If still recording, try to stop it
            stop_event.set()
            if recording_thread and recording_thread.is_alive():
                recording_thread.join(timeout=2)
        print("Exiting program.")

if __name__ == "__main__":
    '''
        1. Press SPACEBAR to start recording.
        2. Speak into your microphone.
        3. Press SPACEBAR again to stop, or wait 10 seconds for it to stop automatically.
        4. The transcription will appear.
        5. You can repeat this process.
        6. Press 'q' to quit the application.
    '''
    main()