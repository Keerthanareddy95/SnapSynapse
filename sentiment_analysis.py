import json
import numpy as np
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Ensure the VADER lexicon is downloaded
nltk.download('vader_lexicon')

def analyze_sentiment_vader(text):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)

def plot_sentiment_trend(sentiment_results):
    # Extract compound sentiment scores for plotting
    compound_scores = [entry['sentiment']['compound'] for entry in sentiment_results]

    # Create a single line plot showing sentiment trend
    plt.figure(figsize=(12, 6))
    plt.plot(compound_scores, color='purple', linestyle='-', marker='o', markersize=5, label="Sentiment Trend")
    plt.axhline(0, color='grey', linestyle='--')  # Add a zero line for neutral sentiment
    plt.title("Sentiment Trend Over the Customer Support Conversation", fontsize=16, fontweight='bold', color="darkblue")
    plt.xlabel("Segment Index")
    plt.ylabel("Compound Sentiment Score")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Load the cleaned customer support transcription
    cleaned_file_path = "cleaned_transcription.json"
    with open(cleaned_file_path, 'r', encoding='utf-8') as file:
        cleaned_transcription = json.load(file)

    # Perform sentiment analysis on each segment of the transcription
    sentiment_results = []
    total_compound = 0
    customer_sentiment = 0
    agent_sentiment = 0
    customer_count = 0
    agent_count = 0

    for entry in cleaned_transcription:
        sentiment_scores = analyze_sentiment_vader(entry['transcription'])
        sentiment_results.append({
            "speaker": entry['speaker'],
            "transcription": entry['transcription'],
            "sentiment": sentiment_scores
        })

        # Accumulate the compound scores for overall analysis
        total_compound += sentiment_scores['compound']

        # Determine sentiment for customer and agent separately based on speaker
        if 'SPEAKER_01' in entry['speaker']:  # Assuming customer speaker label
            customer_sentiment += sentiment_scores['compound']
            customer_count += 1
        else:  # Assuming support agent speaker label
            agent_sentiment += sentiment_scores['compound']
            agent_count += 1

    # Calculate the overall sentiment score
    overall_sentiment_score = total_compound / len(sentiment_results)

    # Calculate average sentiment for Customer and Agent
    average_customer_sentiment = customer_sentiment / customer_count if customer_count else 0
    average_agent_sentiment = agent_sentiment / agent_count if agent_count else 0

    # Determine the overall sentiment as positive, neutral, or negative
    if overall_sentiment_score > 0.05:
        overall_sentiment = "Positive"
    elif overall_sentiment_score < -0.05:
        overall_sentiment = "Negative"
    else:
        overall_sentiment = "Neutral"

    print("Overall Sentiment Score:", overall_sentiment_score)
    print("Overall Sentiment:", overall_sentiment)
    print("Average Customer Sentiment:", average_customer_sentiment)
    print("Average Support Agent Sentiment:", average_agent_sentiment)

    # Save the sentiment analysis results to a JSON file
    sentiment_file_path = "sentiment_analysis_results.json"
    with open(sentiment_file_path, 'w', encoding='utf-8') as file:
        json.dump(sentiment_results, file, ensure_ascii=False, indent=4)

    print(f"Sentiment analysis results saved to {sentiment_file_path}")

    # Plot sentiment trend over the transcript
    plot_sentiment_trend(sentiment_results)
