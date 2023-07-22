import nltk
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

def perform_sentiment_analysis(text):
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(text)
    return scores

if __name__ == "__main__":
    example_text = "I love how friendly and helpful the staff is! The food was amazing too."

    scores = perform_sentiment_analysis(example_text)

    # Plot the sentiment scores as a bar graph
    labels = ['Positive', 'Neutral', 'Negative']
    values = [scores['pos'], scores['neu'], scores['neg']]
    
    plt.bar(labels, values, color=['green', 'blue', 'red'])
    plt.xlabel('Sentiment')
    plt.ylabel('Score')
    plt.title('Sentiment Analysis')
    plt.show()
