import google.generativeai as genai
import json

genai.configure(api_key="your-gemini-api")
model = genai.GenerativeModel("gemini-1.5-pro-latest")
# Define different prompt formats
prompts = {
    "1": "Analyze the following customer support call and provide an expert-level summary. Capture the core issues raised by the customer, the responses provided, and the outcomes. Include specific follow-up actions as required:\n\n"
         "1. **Summary**: Provide a concise overview that identifies key issues raised by the customer and the responses or solutions provided by the support team. Ensure this summary is thorough and covers both explicit and implied insights.\n\n"
         "2. **Action Items**: List any follow-up actions required, specifying who is responsible and the significance of each task.\n\n"
         "3. **Keywords**: Extract 5 to 7 keywords that reflect the main themes, concerns, or resolutions discussed.\n\n"
         "Support call transcription:\n\"{text}\"",

    "2": "Review the customer support call transcription and generate a structured response based on each speaker's contributions:\n\n"
         "1. **Summary**: Summarize the main points raised by each speaker, focusing on customer concerns and the support team’s responses.\n\n"
         "2. **Action Items**: List any follow-up items assigned to the support team or requiring further clarification.\n\n"
         "3. **Keywords**: Extract a list of 5 to 7 keywords that capture the call’s main topics.\n\n"
         "Support call transcription:\n\"{text}\"",

    "3": "Summarize this customer support call with an emphasis on any complaints and suggestions:\n\n"
         "1. **Summary**: Outline the main complaints or issues brought up by the customer and describe the support team's responses or solutions.\n\n"
         "2. **Keywords**:  List 5 to 7 keywords that represent the central themes discussed.\n\n"
         "Transcription text:\n\"{text}\"",

    "4": "Provide a detailed summary of this support call, focusing on any escalation needs and follow-up actions:\n\n"
         "1. **Summary**: Capture any escalations discussed, including reasons for the escalation and expected outcomes.\n\n"
         "2. **Action Items**: Specify follow-up actions required for escalation resolution, with details on responsible teams and timelines."
         "2. **Keywords**:  Include 5 to 7 keywords that summarize the key topics covered.\n\n"
         "Review the transcription here:\n\"{text}\"",

    "5": "Summarize this call with a focus on technical support, outlining the issue, troubleshooting steps, and resolution:\n\n"
         "1. **Summary**: Describe the technical issue, the troubleshooting steps taken, and the resolution or next steps.\n\n"
         "2. **Action Items**: List any required follow-up or additional technical support steps.\n\n"
         "3. **Keywords**: Include 5 to 7 keywords relevant to the technical issue discussed.\n\n"
         "Meeting transcription:\n\"{text}\"",
}

# Function to generate structured content including summary, action items, and keywords
def generate_analysis(text, prompt):
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    response = model.generate_content([prompt.format(text=text)])
    
    # Assuming response comes formatted with headings for each section
    response_text = response.text

    # Sample splitting, may need adjusting depending on model's actual response formatting
    parts = response_text.split("\n")
    summary = []
    action_items = []
    keywords = []

    # Loop through parts to segment the response into summary, action items, and keywords
    current_section = None
    for line in parts:
        if "Summary" in line:
            current_section = "summary"
        elif "Action Items" in line:
            current_section = "action_items"
        elif "Keywords" in line:
            current_section = "keywords"
        elif current_section == "summary":
            summary.append(line.strip())
        elif current_section == "action_items":
            action_items.append(line.strip())
        elif current_section == "keywords":
            keywords.append(line.strip())
    
    # Construct the final structured JSON data
    structured_output = {
        "Summary": " ".join(summary),
        "Action Items": [item for item in action_items if item],
        "Keywords": [keyword for keyword in keywords if keyword]
    }
    
    return structured_output

# Save the structured output to a JSON file
def save_output_to_json(data, output_file_path):
    with open(output_file_path, 'w') as file:
        json.dump(data, file, indent=4)

# Main function to handle the flow
def main():
    input_file = "cleaned_transcription.json"
    output_file = "summary_output.json"
    
    # Display options to user
    print("Choose a summary format:")
    print("1. General Call Summary with Key Takeaways and Actions")
    print("2. Customer and Support Agent Exchange Analysis")
    print("3. Complaint Resolution and Suggestions")
    print("4. Escalation and Follow-up Summary")
    print("5. Technical Issue Assistance Summary")
    

    # Add more options as needed...

    # Capture user choice
    choice = input("Enter the number corresponding to your choice: ")

    # Validate choice
    if choice in prompts:
        # Load text data from JSON
        text = load_text_from_json(input_file)
        
        # Generate structured analysis based on selected prompt
        prompt = prompts[choice]
        structured_output = generate_analysis(text, prompt)
        
        # Save structured analysis to JSON
        save_output_to_json(structured_output, output_file)
        print("Output saved to", output_file)
    else:
        print("Invalid choice. Please select a valid option.")

# Function to load the text from input JSON without timestamps
def load_text_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Combine entries in a structured format for each speaker, without timestamps
    combined_text = ""
    for entry in data:
        combined_text += f"{entry['speaker']}: {entry['transcription']}\n"
    
    return combined_text

# Run the main function
if __name__ == "__main__":
    main()
