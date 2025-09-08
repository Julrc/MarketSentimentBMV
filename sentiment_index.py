# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import praw
from pytrends.request import TrendReq
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import matplotlib.dates as mdates
from scipy import interpolate
import feedparser
from datetime import datetime, timedelta

nltk.download('punkt')
nltk.download('stopwords')
# -

nltk.download('stopwords')
stop_words = set(stopwords.words('spanish'))

# Arreglar google trends, reddit scores.

# +
end_date = datetime.today()

start_date = end_date - timedelta(days=720)

start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')

print("Start Date:", start_date_str)
print("End Date:", end_date_str)

ipc = yf.download('^MXX', start=start_date_str, end=end_date_str, interval='1d')
ipc.reset_index(inplace=True)
ipc['Date'] = pd.to_datetime(ipc['Date'])


# +
def get_google_news_rss_feed(feed_url):
    feed = feedparser.parse(feed_url)
    if feed.bozo:
        print("Error parsing feed:", feed.bozo_exception)
        return []
    if not feed.entries:
        print("No entries found in the feed.")
        return []
    articles = []
    for entry in feed.entries:
        title = entry.title if 'title' in entry else 'No title'
        summary = entry.summary if 'summary' in entry else 'No summary'
        published = datetime(*entry.published_parsed[:6]) if 'published_parsed' in entry else None
        articles.append({
            'title': title,
            'content': summary,
            'date': published
        })
    return articles

# Collect news articles
bolsa_rss_url = "https://news.google.com/rss/search?q=Bolsa+Mexicana+de+Valores&hl=es-419&gl=MX&ceid=MX:es-419"
news_articles = get_google_news_rss_feed(bolsa_rss_url)
if news_articles:
    print(f"Collected {len(news_articles)} articles.")
else:
    print("No articles were collected.")

def preprocess_text(text, stop_words):
    if not isinstance(text, str):
        text = ''
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-záéíóúñ\s]', '', text)
    tokens = word_tokenize(text, language='spanish')
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

#sentiment analysis model
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


def get_sentiment_score(text):
    if not text:
        return 0.0
    try:
        result = sentiment_pipeline(text[:512])[0]  # Truncate text to 512 tokens
        label = result['label']
        score = int(label.split()[0])  # Extract the number from '1 star', '2 stars', etc.
        normalized_score = (score - 3) / 2  # Normalize to range [-1, 1]
        return normalized_score
    except Exception as e:
        print(f"Error processing text: {e}")
        return 0.0

news_df = pd.DataFrame(news_articles)
if 'date' in news_df.columns:
    news_df['Date'] = pd.to_datetime(news_df['date']).dt.normalize()
    news_df.drop('date', axis=1, inplace=True)
else:
    news_df['Date'] = pd.to_datetime(news_df['published']).dt.normalize()
news_df['combined_text'] = news_df['title'].astype(str) + ' ' + news_df['content'].astype(str)
news_df['content_clean'] = news_df['combined_text'].apply(lambda x: preprocess_text(x, stop_words))
news_df['sentiment_score'] = news_df['content_clean'].apply(get_sentiment_score)

desired_columns = ['Date', 'title']

news_df[desired_columns].to_csv('news_titles.csv', index=False)

print("Date and titles have been saved to 'news_titles.csv'.")

# +

#Reddit API
reddit = praw.Reddit(
    client_id="fbtWruG8aopQ5chxNLpURw",
    client_secret="hGgoXi6sA0Qv0wgAezpdXqGvOgc_1Q",
    user_agent="marketsentimentbmv"
)

reddit.read_only = True

def get_reddit_posts(subreddits, query, limit=10):
    posts = []
    for subreddit in subreddits:
        print(f"Searching in subreddit: {subreddit}")
        subreddit_obj = reddit.subreddit(subreddit)
        for submission in subreddit_obj.search(query, limit=limit):
            posts.append({
                'title': submission.title,
                'content': submission.selftext,
                'created': pd.to_datetime(submission.created_utc, unit='s')
            })
    return posts

subreddits = ['MexicoBursatil', 'MexicoFinanciero']
query = 'Acciones' , 'comprar', 'vender'
reddit_posts = get_reddit_posts(subreddits, query)

# +

# reddit posts to DataFrame
reddit_df = pd.DataFrame(reddit_posts)

reddit_df['Date'] = reddit_df['created'].dt.normalize()
reddit_df.drop('created', axis=1, inplace=True)

# compute sentiment
reddit_df['content_clean'] = reddit_df.apply(lambda x: preprocess_text(x['title'] + ' ' + x['content'], stop_words), axis=1)
reddit_df['sentiment_score'] = reddit_df['content_clean'].apply(get_sentiment_score)


# +
from pytrends.request import TrendReq

def get_google_trends_data(keywords, timeframe='today 1-m'):
    pytrends = TrendReq(hl='es-MX', tz=360)
    pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo='MX', gprop='')
    data = pytrends.interest_over_time()
    return data


keywords = ['BMV', 'inversiones', 'Acciones', 'comprar acciones']
trends_data = get_google_trends_data(keywords)
# -

# Verificar si se obtuvieron datos
if trends_data.empty:
    print("No se obtuvieron datos de Google Trends. Verifica las palabras clave y el timeframe.")
else:

    trends_data.reset_index(inplace=True)

    print("Datos originales:")
    print(trends_data.head())

    # Calcular el cambio porcentual para cada palabra clave
    for keyword in keywords:
        trends_data[f'{keyword}_pct_change'] = trends_data[keyword].pct_change() * 100

    # Definir cambios significativos
    significant_threshold = 51

    # Crear columnas que indiquen si hubo un aumento significativo
    for keyword in keywords:
        trends_data[f'{keyword}_sig_increase'] = trends_data[f'{keyword}_pct_change'].apply(
            lambda x: 1 if x > significant_threshold else 0
        )
    # Crear una columna que sume los aumentos significativos de todas las palabras clave
    trend_sig_columns = [f'{keyword}_sig_increase' for keyword in keywords]
    trends_data['Google_Trends_Significant_Increase'] = trends_data[trend_sig_columns].sum(axis=1)

    trends_data['date'] = pd.to_datetime(trends_data['date']).dt.normalize()


trends_data['date'] = pd.to_datetime(trends_data['date']).dt.normalize()
trends_data.rename(columns={'date': 'Date'}, inplace=True)

trends_daily = trends_data[['Date', 'Google_Trends_Significant_Increase']]

#Aggregate News Sentiment by Date
news_daily_sentiment = news_df.groupby('Date')['sentiment_score'].mean().reset_index()
print(news_daily_sentiment)

#Aggregate Reddit Sentiment by Date
reddit_daily_sentiment = reddit_df.groupby('Date')['sentiment_score'].mean().reset_index()
print(reddit_daily_sentiment)


# +
#Volatility
ipc = yf.download('^MXX', start=start_date_str, end=end_date_str, interval='1d')
ipc.reset_index(inplace=True)
ipc['Date'] = pd.to_datetime(ipc['Date'])
ipc['Returns'] = ipc['Close'].pct_change()
ipc['Volatility'] = ipc['Returns'].rolling(window=2).std() * np.sqrt(2)

print(ipc['Volatility'])

default_volatility = 0.01
ipc['Volatility'].fillna(default_volatility, inplace=True)

print("ipc shape:", ipc.shape)
print(ipc[['Date', 'Volatility']].head())

# +
# Merge News and Reddit sentiment
sentiment_data = pd.merge(
    news_daily_sentiment,
    reddit_daily_sentiment,
    on='Date',
    how='outer',
    suffixes=('_news', '_reddit')
)

sentiment_data = pd.merge(
    sentiment_data,
    trends_daily,
    on='Date',
    how='outer'
)


sentiment_data.fillna(method='ffill', inplace=True)
sentiment_data.dropna(inplace=True)

# Merge w md
ipc_sentiment = pd.merge(sentiment_data, ipc[['Date', 'Volatility']], on='Date', how='left')
ipc_sentiment.dropna(inplace=True)

# +

ipc['Returns'] = ipc['Close'].pct_change()


ipc['Volatility'] = ipc['Returns'].rolling(window=30).std() * np.sqrt(30)
ipc['Volatility'].fillna(ipc['Volatility'].mean(), inplace=True)

ipc['Volume'] = ipc['Volume'].astype(float)

ipc['Volume_30d_avg'] = ipc['Volume'].rolling(window=30).mean()
ipc['Volume_vs_30d_avg'] = ipc['Volume'] / ipc['Volume_30d_avg']


ipc['Momentum'] = ipc['Close'] - ipc['Close'].shift(1)
ipc['Momentum_30d_avg'] = ipc['Momentum'].rolling(window=30).mean()
ipc['Momentum_vs_30d_avg'] = ipc['Momentum'] / ipc['Momentum_30d_avg']


print("Columns in ipc DataFrame:")
print(ipc.columns.tolist())

# selecting ipc_metrics
ipc_metrics = ipc[['Date', 'Volatility', 'Volume', 'Close', 'Momentum', 'Volume_30d_avg',
                   'Volume_vs_30d_avg', 'Momentum_30d_avg',
                   'Momentum_vs_30d_avg',]]

# merging ipc_metrics with sentiment_data
ipc_sentiment = pd.merge(
    sentiment_data,
    ipc_metrics,
    on='Date',
    how='outer'
)

ipc_sentiment.sort_values('Date', inplace=True)
ipc_sentiment.fillna(method='ffill', inplace=True)
ipc_sentiment.fillna(method='bfill', inplace=True)

# Invert Volatility
ipc_sentiment['Volatility_inv'] = 1 / ipc_sentiment['Volatility']
ipc_sentiment.replace([np.inf, -np.inf], np.nan, inplace=True)
ipc_sentiment['Volatility_inv'].fillna(ipc_sentiment['Volatility_inv'].mean(), inplace=True)


# +
features_to_normalize = [
    'sentiment_score_news',
    'sentiment_score_reddit',
    'Google_Trends_Significant_Increase',
    'Volatility',
    'Volume',
    'Momentum',
    'Volume_vs_30d_avg',
    'Momentum_vs_30d_avg',
    'Volatility_inv'
]

scaler = MinMaxScaler()

# Apply scaling
ipc_sentiment_scaled = ipc_sentiment.copy()
ipc_sentiment_scaled[features_to_normalize] = scaler.fit_transform(
    ipc_sentiment[features_to_normalize]
)

print(ipc_sentiment_scaled)

# +
weights = {
    'sentiment_score_news': 0.2,
    'sentiment_score_reddit': 0.2,
    'Google_Trends_Significant_Increase':0.1,
#25% Volatility
    'Volatility_inv': 0.25,
#25% market/volume
    'Volume_vs_30d_avg': 0.15,
    'Momentum_vs_30d_avg': 0.1,
}

features = list(weights.keys())

ipc_sentiment_scaled.set_index('Date', inplace=True)

def adjust_weights(row, weights):
    available_weights = {}
    total_weight = 0

    for feature, weight in weights.items():
        if pd.notna(row[feature]):
            available_weights[feature] = weight
            total_weight += weight

    # Normalize weights
    for feature in available_weights:
        available_weights[feature] /= total_weight

    return available_weights

def calculate_sentiment_index(row):
    adj_weights = adjust_weights(row, weights)
    sentiment_index = 0
    for feature, weight in adj_weights.items():
        sentiment_index += row[feature] * weight
    return sentiment_index * 100  # Scale to 0-100

ipc_sentiment_scaled['Sentiment_Index'] = ipc_sentiment_scaled.apply(calculate_sentiment_index, axis=1)

ipc_sentiment_scaled.reset_index(inplace=True)

# DataFrame with  'Date' and 'Sentiment_Index' csv
sentiment_index_df = ipc_sentiment_scaled[['Date', 'Sentiment_Index']]

today = datetime.now()

# Calculate the date one year ago
one_year_ago = today - timedelta(days=673)

# Filter the DataFrame for dates within the past year
sentiment_index_df = sentiment_index_df[sentiment_index_df['Date'] >= one_year_ago]

# Optional: Reset index if desired
sentiment_index_df.reset_index(drop=True, inplace=True)

sentiment_index_df.to_csv('sentiment_index.csv', index=False)

