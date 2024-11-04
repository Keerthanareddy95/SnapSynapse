import json
import re

# Load the transcriptions from the JSON file
with open("diarized_transcriptions.json", "r") as json_file:
    transcriptions = json.load(json_file)

# function to clean the transcription text
def clean_transcription(text):
    # List of common filler words
    filler_words = [
        "um", "uh", "like", "you know", "actually", "basically", "I mean",
        "sort of", "kind of", "right", "okay", "so", "well", "just"
    ]
    
    # regex pattern to match filler words (case insensitive)
    filler_pattern = re.compile(r'\b(' + '|'.join(filler_words) + r')\b', re.IGNORECASE)
    
    # Remove filler words
    cleaned_text = filler_pattern.sub('', text)
    
    # Remove extra whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text

# Clean the transcriptions
cleaned_transcriptions = []
for entry in transcriptions:
    cleaned_entry = {
        "speaker": entry["speaker"],
        "start_time": entry["start_time"],
        "end_time": entry["end_time"],
        "transcription": clean_transcription(entry["transcription"])
    }
    cleaned_transcriptions.append(cleaned_entry)

# Save the cleaned transcriptions to a new JSON file
with open("cleaned_transcription.json", "w") as cleaned_json_file:
    json.dump(cleaned_transcriptions, cleaned_json_file, indent=4)

print("Cleaned transcriptions saved to 'cleaned_transcription.json'.")
