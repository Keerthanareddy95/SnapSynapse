import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the model and tokenizer
model_name = "t5-small"  # You can use any other pre-trained T5 model
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Streamlit app title
st.title("Text Summarization with T5")

# User input for text to summarize
input_text = st.text_area("Enter the text you want to summarize", height=200)

# Text input for minimum and maximum length of the summary
min_length = st.text_input("Enter minimum length of the summary", value="30")
max_length = st.text_input("Enter maximum length of the summary", value="150")

# Button to trigger summarization
if st.button("Summarize"):
    if input_text:
        try:
            # Convert the input values to integers
            min_length = int(min_length)
            max_length = int(max_length)

            # Tokenize the input text
            inputs = tokenizer.encode("summarize: " + input_text, return_tensors="pt", max_length=512, truncation=True)

            # Generate summary with specified min and max length
            summary_ids = model.generate(
                inputs, 
                max_length=max_length, 
                min_length=min_length, 
                length_penalty=1.0, 
                num_beams=6, 
                early_stopping=True
            )
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            # Display the summary
            st.subheader("Summary")
            st.write(summary)

        except ValueError:
            st.error("Please enter valid numbers for min and max length.")
    else:
        st.error("Please enter some text to summarize!")
