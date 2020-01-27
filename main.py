import lyricsgenius
import os.path
import re
import config

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# prompts user for API KEY if automatic setup isn't already configured
def prompt_API_KEY():
	try:
		genius = lyricsgenius.Genius(config.API_KEY)
	except:
		print("No config.py file found. To setup automatic key, input 'y'. Otherwise, input Genius API key for one time use:")
		key = input()

		if key == 'y':
			print("Input API_KEY now: ")
			api_key = input()

			f = open("config.py", "w")
			f.write("API_KEY = \"" + api_key + "\"")
			f.close()
		else:
			genius = lyricsgenius.Genius(config.API_KEY)

	genius.remove_section_headers = True
	return genius

# Generates song list from list of artists
def generate_song_list(artists)
	# artist = genius.search_artist("DJ Khaled", max_songs=10, sort="popularity")
	# print(artist.songs)
	# Add training songs to lyrics list
	song_lyrics = []

	for artist in artists:
		top_10 = genius.search_artist(artist, max_songs=10, sort="popularity")
		for song in top_10.songs:
			song_lyrics.append(song.lyrics.replace("\n", " "))

	return song_lyrics

# Save a song in filepath Songs/Artist/Album/Title
def save_song_lyrics(song):
	filepath = "Songs/" + song.artist + "/" + song.album + "/"

	if not os.path.exists(filepath):
		os.makedirs(filepath)

	song.save_lyrics(filename=filepath + song.title)

# Method to clean and format lyrics
def preprocess_lyrics(lyrics):
	REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
	REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

	lyrics = [REPLACE_NO_SPACE.sub("", line.lower()) for line in lyrics]
	lyrics = [REPLACE_WITH_SPACE.sub(" ", line) for line in lyrics]
	
	return lyrics

def __main__():
	genius = prompt_API_KEY()

	artists = ["Drake", "Playboi Carti", "DJ Khaled", "Justin Bieber", "Usher"]
	test_artists = ["Lady Gaga", "Queen", "Nav", "Eminem", "Taylor Swift"]

	song_lyrics = generate_song_list(artists)
	song_lyrics_test = generate_song_list(test_artists)

	lyrics_clean = preprocess_lyrics(song_lyrics)
	lyrics_clean_test = preprocess_lyrics(song_lyrics_test)

	# Vectorization

	cv = CountVectorizer(binary=True)
	cv.fit(lyrics_clean)
	X = cv.transform(lyrics_clean)
	X_test = cv.transform(lyrics_clean_test)

	# Build Classifier

	target = [1 if i < 30 else 0 for i in range(50)]

	X_train, X_val, y_train, y_val = train_test_split(
	    X, target, train_size = .80
	)

	#find most optimal c value

	#for c in [0.01, 0.05, 0.25, 0.5, 1]:
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

	#Evaluate model

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
	    
	for best_negative in sorted(
	    feature_to_coef.items(), 
	    key=lambda x: x[1])[:5]:
	    print (best_negative)