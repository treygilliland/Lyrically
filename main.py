import lyricsgenius
import os.path
import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

#Client Access token
genius = lyricsgenius.Genius("xLRuSen3RJZ2MNt5suRH9mrZsKWwocDzVQTKqbqcKu4BMG1s-PGN4AEyngzI69Eo")
#artist = genius.search_artist("Kodak Black", max_songs=3, sort="popularity")
#print(artist.songs)

genius.remove_section_headers = True

song = genius.search_song("Tunnel Vision", "Kodak Black")
#print(song.lyrics)


#This will save a song in the form Songs/Artist/Album/Title
def save_song_lyrics_filepath(self, song):
	filepath = "Songs/" + song.artist + "/" + song.album + "/"

	if not os.path.exists(filepath):
		os.makedirs(filepath)

	song.save_lyrics(filename=filepath + song.title)