import os

import IPython.display as ipd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm
import librosa
import librosa.display

import utils

plt.rcParams['figure.figsize'] = (17, 5)

# Path to the audio file
audio_file_path = 'data/fma_dataset/fma_small/008/'


# Load metadata and features.
tracks = utils.load('data/fma_metadata/tracks.csv')
genres = utils.load('data/fma_metadata/genres.csv')

print(f"The dataset in total has {tracks.shape[0]} tracks with {tracks.shape[1]} features.")
print(f"The dataset in total has {genres.shape[0]} genres.")

# getting the 8 most common genres in the dataset
small_genres = genres[['#tracks', 'title']].sort_values(by='#tracks', ascending=False).head(8)
print(small_genres)

small_data = tracks[tracks['set', 'subset'] <= 'small']
columns = small_data.columns.tolist()
print(columns)
small_data = small_data["album"]["tags"]
print(small_data.head())
