import lyricsgenius
import os.path
import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.corpus import stopwords
from textblob import TextBlob
from textblob import Word
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import matplotlib.pyplot as plt


def s_analysis(file_df):
    lyrics = pd.DataFrame(data=file_df)
    album = lyrics['songs'][0]['album']
    artist = lyrics['artist'][0]
    lyrics['title'] = lyrics['songs'][0]['title']

    n = 0

    while n < len(lyrics):
        lyrics['title'][n] = lyrics['songs'][n]['title']
        lyrics['songs'][n] = lyrics['songs'][n]['lyrics'].replace('\n', ' ')
        n += 1

    # Takes out stop words
    stop = stopwords.words('english')
    lyrics['stopwords'] = lyrics['songs'].apply(
        lambda x: len([x for x in x.split() if x in stop]))

    # 2) Basic Pre-Processing

    # Takes out anything that isn't a word/number/space
    lyrics['songs'] = lyrics['songs'].apply(
        lambda x: " ".join(x.lower() for x in x.split()))
    lyrics['songs'] = lyrics['songs'].str.replace('[^\w\s]', '')
    lyrics['songs'] = lyrics['songs'].apply(
        lambda x: " ".join(x for x in x.split() if x not in stop))

    bars = TextBlob(lyrics['songs'][0]).words
    print(bars)

    # Generate a word cloud image
    wordcloud = WordCloud(background_color="white").generate(lyrics['songs'][0])

    # Display the generated image:
    # the matplotlib way:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    """
    TextBlob(lyrics['songs'][0]).words
    lyrics['songs'] = lyrics['songs'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    lyrics['sentimentTxtB'] = lyrics['songs'].apply(lambda x: TextBlob(x).sentiment[0])
    lyrics['sentimentTxtB'] = lyrics['sentimentTxtB'].round(3)
    # lyrics = lyrics.sort_values('sentimentTxtB', ascending=False)

    lyrics['sentimentVaderPos'] = lyrics['songs'].apply(lambda x: analyzer.polarity_scores(x)['pos'])
    lyrics['sentimentVaderNeg'] = lyrics['songs'].apply(lambda x: analyzer.polarity_scores(x)['neg'])

    lyrics['Vader'] = lyrics['sentimentVaderPos'] - lyrics['sentimentVaderNeg']

    nb_rows = len(lyrics.index)
    total_sent = sum(lyrics['sentimentTxtB'])
    avg = total_sent / nb_rows

    total_vad = sum(lyrics['Vader'])
    avg_vad = total_vad / nb_rows

    print(lyrics[['title', 'songs', 'sentimentTxtB', 'Vader']])
    avg = round(avg, 3)
    avg_vad = round(avg_vad, 3)
    print(album + ' by ' + artist + ' - TextBlob: ' + str(avg) + ", Vader: " + str(avg_vad))
    return album, artist, avg, avg_vad
    """

index = len([filename for filename in os.listdir(
    os.getcwd()) if filename.endswith(".json")])

avg_df = pd.DataFrame(index=range(index), columns=['album', 'artist'])

for filename in os.listdir(os.getcwd()):
    if filename.endswith(".json"):
        df = pd.read_json(filename)
        s_analysis(df)
