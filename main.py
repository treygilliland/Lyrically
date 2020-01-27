import lyricsgenius
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

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

#Client Access token
genius = lyricsgenius.Genius("xLRuSen3RJZ2MNt5suRH9mrZsKWwocDzVQTKqbqcKu4BMG1s-PGN4AEyngzI69Eo")
#artist = genius.search_artist("Kodak Black", max_songs=3, sort="popularity")
#print(artist.songs)

genius.remove_section_headers = True


song = genius.search_song("Tunnel Vision", "Kodak Black")
#print(song.lyrics)

print(type(song.lyrics))

lyrics_clean = preprocess_lyrics(song.lyrics)
print(lyrics_clean)