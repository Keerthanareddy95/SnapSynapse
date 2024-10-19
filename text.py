import streamlit as st

# Import summarization function
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load pre-trained T5 model and tokenizer
model_name = "t5-small"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Function to perform summarization
def summarize(text):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(
        inputs, 
        max_length=150, 
        min_length=50, 
        length_penalty=1.0, 
        num_beams=6, 
        early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Streamlit app UI
st.title("📜 Text Summarization Tool 📑")

# Text input box
text_input = st.text_area("Enter the text you want to summarize:", height=300)

# If input is given, display the summarized result
if st.button("Summarize"):
    if text_input:
        with st.spinner("Summarizing..."):
            summary = summarize(text_input)
        st.subheader("Summary:")
        st.write(summary)
    else:
        st.error("Please enter the text to summarize.")
