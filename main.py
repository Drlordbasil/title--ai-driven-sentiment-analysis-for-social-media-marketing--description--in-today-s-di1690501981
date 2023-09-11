import tweepy
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

# Twitter API credentials
consumer_key = 'your_consumer_key'
consumer_secret = 'your_consumer_secret'
access_token = 'your_access_token'
access_token_secret = 'your_access_token_secret'

# Setup Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)


def scrape_social_media_data(keyword, count):
    tweets = []
    try:
        # Scrape tweets based on keyword
        fetched_tweets = tweepy.Cursor(
            api.search, q=keyword, lang='en', tweet_mode='extended').items(count)
        for tweet in fetched_tweets:
            tweets.append(tweet.full_text)
        return tweets
    except tweepy.TweepError as e:
        print("Error: " + str(e))


def preprocess_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove special characters and numbers
    text = re.sub('[^a-zA-Z]', ' ', text)

    # Convert to lowercase
    text = text.lower()

    # Tokenize text
    tokens = text.split()

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Join tokens back into a single string
    processed_text = ' '.join(tokens)

    return processed_text


def train_sentiment_analysis_model(X_train, y_train):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    encoded_data_train = tokenizer.batch_encode_plus(
        X_train,
        add_special_tokens=True,
        return_attention_mask=True,
        padding='max_length',
        truncation=True,
        max_length=256,
        return_tensors='tf'
    )
    input_ids_train = tf.convert_to_tensor(encoded_data_train['input_ids'])
    attention_masks_train = tf.convert_to_tensor(
        encoded_data_train['attention_mask'])
    labels_train = tf.convert_to_tensor(y_train)

    model = TFDistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased')
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    model.compile(optimizer=optimizer, loss=model.compute_loss,
                  metrics=['accuracy'])
    model.fit([input_ids_train, attention_masks_train],
              labels_train, epochs=2, batch_size=16)

    return model


def predict_sentiment(model, text):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    encoded_data = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        return_attention_mask=True,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )
    input_ids = tf.convert_to_tensor(encoded_data['input_ids'])
    attention_masks = tf.convert_to_tensor(encoded_data['attention_mask'])

    predictions = model.predict([input_ids, attention_masks])
    sentiment = predictions.logits[0].numpy().argmax()

    return sentiment


def visualize_sentiment_distribution(sentiment_data):
    sns.set(style='darkgrid')
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x='Sentiment', data=sentiment_data, palette='viridis')
    ax.set_xlabel('Sentiment', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Sentiment Distribution', fontsize=14)
    plt.show()


def run_analysis_tool(keyword, count):
    # Step 1: Data Collection
    tweets = scrape_social_media_data(keyword, count)

    # Step 2: Preprocessing
    processed_tweets = [preprocess_text(tweet) for tweet in tweets]

    # Step 3: Sentiment Analysis Model
    # Load labeled dataset
    labeled_data = pd.read_csv('labeled_data.csv')

    # Split data into features and labels
    X = labeled_data['tweet'].values
    y = labeled_data['sentiment'].values

    # Train-test split for model training and evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Train sentiment analysis model
    model = train_sentiment_analysis_model(X_train, y_train)

    # Step 4: Real-time Analysis
    sentiment_scores = []
    for tweet in processed_tweets:
        sentiment = predict_sentiment(model, tweet)
        sentiment_scores.append(sentiment)

    # Step 5: Visualization and Reporting
    sentiment_data = pd.DataFrame({'Sentiment': sentiment_scores})
    visualize_sentiment_distribution(sentiment_data)


run_analysis_tool('brand', 1000)
