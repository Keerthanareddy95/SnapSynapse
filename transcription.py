import os
from pyannote.audio import Pipeline
import openai
import json
from dotenv import load_dotenv
from pydub import AudioSegment

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()

# Load the speaker diarization pipeline
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

def perform_diarization(file_path):
    # Run speaker diarization on the audio file
    diarization_result = diarization_pipeline(file_path)
    return diarization_result

def transcribe_with_diarization(file_path):
    diarization_result = perform_diarization(file_path)
    audio = AudioSegment.from_file(file_path)
    transcriptions = []

    for segment, _, speaker in diarization_result.itertracks(yield_label=True):
        start_time_ms = int(segment.start * 1000)
        end_time_ms = int(segment.end * 1000)
        chunk = audio[start_time_ms:end_time_ms]
        
        chunk_filename = f"{speaker}_segment_{int(segment.start)}.wav"
        chunk.export(chunk_filename, format="wav")

        with open(chunk_filename, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="json"
            )
            transcriptions.append({
                "speaker": speaker,
                "start_time": segment.start,
                "end_time": segment.end,
                "transcription": transcription.text
            })
        print(f"Transcription for {chunk_filename} by {speaker} completed.")

    # Save the transcriptions to a JSON file
    with open("diarized_transcriptions.json", "w") as json_file:
        json.dump(transcriptions, json_file, indent=4)

    print("Diarized transcriptions saved to 'diarized_transcriptions.json'.")

# Run the function on your audio file
transcribe_with_diarization("test_audio_1.wav")
