import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from wordcloud import STOPWORDS
import nltk
import re
from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

def dataAnalysis():
    df_true = pd.read_csv('True.csv')
    df_fake = pd.read_csv('Fake.csv')
    df_fake['target'] = 0
    df_true['target'] = 1

    df = pd.concat([df_true, df_fake]).reset_index(drop=True)

    df.drop_duplicates(inplace=True)

    df = df.drop(columns=['date', 'subject'])

    df['content'] = df['title'] + ' ' + df['text']

    sns.countplot(x='target', data=df)
    plt.savefig('../static/vis/newsdist.jpg')
    plt.clf()

    label_counts = df['target'].value_counts()

    plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=90, colors=['#03fcec', '#0330fc'])
    plt.savefig('../static/vis/labelpct.jpg')

    # create number of sentences
    df['num_characters'] = df['content'].apply(len)

    # crete number of words
    df['num_words'] = df['content'].apply(lambda x: len(nltk.word_tokenize(x)))

    # create number of sentences
    df['num_sentences'] = df['content'].apply(lambda x: len(nltk.sent_tokenize(x)))

    df[['num_characters', 'num_words', 'num_sentences']].describe()

    wc = WordCloud(width=2000, height=800, min_font_size=10, background_color='white',
                   colormap='rainbow')

    spam_wc = wc.generate(df[df['target'] == 0]['content'].str.cat(sep=' '))

    spam_wc.to_file('../static/vis/spamcloud.jpg')

    wc = WordCloud(width=2000, height=800, min_font_size=10, background_color='white',
                   colormap='rainbow')

    true_wc = wc.generate(df[df['target'] == 1]['content'].str.cat(sep=' '))

    true_wc.to_file('../static/vis/truecloud.jpg')

    corpus = []
    for news in df['content'].tolist():
        for word in news.split():
            corpus.append(word)

    word_counts = Counter(corpus).most_common(10)
    words, counts = zip(*word_counts)

    df = df.drop(columns=['title', 'text', 'num_characters', 'num_words', 'num_sentences'])

    # Initialize stop words
    stop_words = set(stopwords.words('english'))

    def preprocess_text(text):
        # Step 1: Convert to lowercase
        text = text.lower()
        # Step 2: Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
        # Step 3: Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Step 4: Remove special characters, punctuation, and numbers
        text = re.sub(r"[^a-zA-Z\s]", '', text)
        # Step 5: Tokenize text
        words = word_tokenize(text)
        # Step 6: Remove stop words
        words = [word for word in words if word not in stop_words]
        # Step 7: Join words back into a single string
        processed_text = ' '.join(words)

        return processed_text

    df['clean_text'] = df['content'].apply(preprocess_text)

    df = df.drop(columns='content')

    df.to_csv('../clean_content.csv', index=False)

    df = pd.read_csv('../clean_content.csv')

    corpus = []
    for news in df['clean_text'].tolist():
        if isinstance(news, str):  # Check if the value is a string
            for word in news.split():
                corpus.append(word)

    # Count the most common 30 words
    word_counts = Counter(corpus).most_common(10)
    words, counts = zip(*word_counts)

    fake_corpus = []
    for news in df[df['target'] == 0]['clean_text'].dropna().tolist():
        if isinstance(news, str):  # Check if news is a string
            for word in news.split():
                fake_corpus.append(word)

    # Count the most common 30 words
    word_counts = Counter(fake_corpus).most_common(10)
    words, counts = zip(*word_counts)

    sns.barplot(x=list(words), y=list(counts), palette="viridis")
    plt.savefig('../static/vis/fake10.jpg')
    plt.clf()

    real_corpus = []
    for news in df[df['target'] == 1]['clean_text'].dropna().tolist():
        if isinstance(news, str):  # Check if news is a string
            for word in news.split():
                real_corpus.append(word)

    # Count the most common 30 words
    word_counts = Counter(real_corpus).most_common(10)
    words, counts = zip(*word_counts)

    sns.barplot(x=list(words), y=list(counts), palette="viridis")
    plt.savefig('../static/vis/true10.jpg')
    plt.clf()

    wc = WordCloud(width=2000, height=800, min_font_size=10, background_color='white',
                   colormap='rainbow')

    spam_wc = wc.generate(df[df['target'] == 0]['clean_text'].str.cat(sep=' '))

    spam_wc.to_file('../static/vis/fakewc_clean.jpg')

    wc = WordCloud(width=2000, height=800, min_font_size=10, background_color='white',
                   colormap='rainbow')

    true_wc = wc.generate(df[df['target'] == 1]['clean_text'].str.cat(sep=' '))

    true_wc.to_file('../static/vis/truewc_clean.jpg')

    df = pd.read_csv('../clean_content.csv')











#dataAnalysis()

