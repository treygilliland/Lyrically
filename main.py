import lyricsgenius
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import config

from os import path
from PIL import Image

# Method to clean and format lyrics
def preprocess_lyrics(lyrics):
	REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
	REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

	lyrics = [REPLACE_NO_SPACE.sub("", line.lower()) for line in lyrics]
	lyrics = [REPLACE_WITH_SPACE.sub(" ", line) for line in lyrics]
	
	return lyrics

# Save a song in filepath Songs/Artist/Album/Title
def save_song_lyrics(song):
	filepath = "Songs/" + song.artist + "/" + song.album + "/"

	if not os.path.exists(filepath):
		os.makedirs(filepath)

	song.save_lyrics(filename=filepath + song.title)

# Connect to Genius API
genius = lyricsgenius.Genius(config.API_KEY)

# Pull song list from Genius
artists = ["Drake", "Playboi Carti", "DJ Khaled", "Justin Bieber", "Usher", "Kesha"]
# artist = genius.search_artist("DJ Khaled", max_songs=10, sort="popularity")
# print(artist.songs)

genius.remove_section_headers = True

# Add training songs to lyrics list
song_lyrics = []

for artist in artists:
	top_10 = genius.search_artist(artist, max_songs=10, sort="popularity")
	for song in top_10.songs:
		song_lyrics.append(song.lyrics.replace("\n", " "))

# Add test songs to test lyrics list
song_lyrics_test = []
test_artists = ["Lady Gaga", "Queen", "Nav", "Eminem", "Taylor Swift", "Rihanna"]
for artist in test_artists:
	top_10 = genius.search_artist(artist, max_songs=10, sort="popularity")
	for song in top_10.songs:
		song_lyrics_test.append(song.lyrics.replace("\n", " "))

# Clean lyrics
lyrics_clean = preprocess_lyrics(song_lyrics)
lyrics_clean_test = preprocess_lyrics(song_lyrics_test)
print(lyrics_clean)

# Vectorization
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(binary=True)
cv.fit(lyrics_clean)
X = cv.transform(lyrics_clean)
X_test = cv.transform(lyrics_clean_test)

# Build Classifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

target = [1 if i < 30 else 0 for i in range(60)]

X_train, X_val, y_train, y_val = train_test_split(
    X, target, train_size = .75
)

#find c

#for c in [0.01, 0.05, 0.25, 0.5, 1]:
lowest_c = 1
best_acc = .5
for c in [0.001, 0.005, 0.01, 0.05, 0.1]:
    lr = LogisticRegression(C=c)
    lr.fit(X_train, y_train)
    acc = accuracy_score(y_val, lr.predict(X_val))
    print ("Accuracy for C=%s: %s" 
           % (c, acc))

final_model = LogisticRegression(C=0.05)
final_model.fit(X, target)
print ("Final Accuracy: %s" 
       % accuracy_score(target, final_model.predict(X_test)))
# Final Accuracy: 0.88128


feature_to_coef = {
    word: coef for word, coef in zip(
        cv.get_feature_names(), final_model.coef_[0]
    )
}
for best_positive in sorted(
    feature_to_coef.items(), 
    key=lambda x: x[1], 
    reverse=True)[:5]:
    print (best_positive)
    
#     ('excellent', 0.9288812418118644)
#     ('perfect', 0.7934641227980576)
#     ('great', 0.675040909917553)
#     ('amazing', 0.6160398142631545)
#     ('superb', 0.6063967799425831)
    
for best_negative in sorted(
    feature_to_coef.items(), 
    key=lambda x: x[1])[:5]:
    print (best_negative)
    
#     ('worst', -1.367978497228895)
#     ('waste', -1.1684451288279047)
#     ('awful', -1.0277001734353677)
#     ('poorly', -0.8748317895742782)
#     ('boring', -0.8587249740682945)