import openai
import os
import json
from dotenv import load_dotenv
from pydub import AudioSegment

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

from openai import OpenAI
client = OpenAI()

def split_audio(file_path, chunk_duration_ms=90000):
    # Load the audio file
    audio = AudioSegment.from_file(file_path)
    # Split the audio into chunks of the specified duration
    chunks = [audio[i:i + chunk_duration_ms] for i in range(0, len(audio), chunk_duration_ms)]
    return chunks

def whisper_transcription(audio_chunks):
    all_transcriptions = []  # List to store transcriptions from all chunks

    for i, chunk in enumerate(audio_chunks):
        # Export each chunk as a .wav file
        chunk_filename = f"chunk_{i}.wav"
        chunk.export(chunk_filename, format="wav")
        
        # Transcribe the audio chunk
        with open(chunk_filename, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="json"
            )
            # Extract the 'text' field from the response
            transcription_text = transcription.text

        # Append the transcription to the list
        all_transcriptions.append({
            "chunk_id": i,
            "transcription": transcription_text
        })
        
        print(f"Transcription for {chunk_filename} completed.")

    # Save all transcriptions to a single JSON file
    with open("transcriptions_generated.json", "w") as json_file:
        json.dump(all_transcriptions, json_file, indent=4)

    print("All transcriptions saved to 'transcriptions_generated.json'.")

# Split the audio into smaller parts and transcribe
audio_chunks = split_audio("test_audio_1.wav")
whisper_transcription(audio_chunks)
