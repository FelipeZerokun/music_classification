import soundfile as sf
import openl3

import numpy as np
from pathlib import Path
import os

def create_music_embeddings(audio_file_path, batch_size, model):

    if not os.path.exists(audio_file_path):
        print("Audio file not found.")
        return None

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load audio file
    print("Loading audio file: ", audio_file_path)
    audio, sr = sf.read(audio_file_path)
    print("Audio loaded successfully.")

    # checking the shape of the audio file
    if len(audio.shape) > 1:
        print("Audio file has more than one channel. Only the first channel will be used.")
        audio = np.mean(audio, axis=1)

    # Compute embeddings
    print("Computing embeddings...")
    embeddings, timestamps = openl3.get_audio_embedding(audio, sr,
                                         content_type="music",
                                         input_repr="mel128",
                                         embedding_size=512)

    print("done")

    return embeddings, timestamps

def process_audio_embedding(audio_file_path, file_suffix, output_dir, model):

    if not os.path.exists(audio_file_path):
        print("Audio file not found.")
        return None

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the embedding to '/different/dir/file_suffix.npz'
    openl3.process_audio_file(audio_file_path,
                              model=model,
                              suffix=file_suffix,
                              output_dir=output_dir)


def load_embedding_file(embedding_file_path):
    # Load the embedding file
    print("Loading embedding file: ", embedding_file_path)
    data = np.load(embedding_file_path)

    embeddings, timestamps = data['embedding'], data['timestamps']

    print("Embedding file loaded successfully.")

    return embeddings, timestamps

def read_files(data_dir):
    # Read the files in the data directory
    audio_files = os.listdir(data_dir)

    return audio_files


if __name__ == '__main__':
    # Load the pre-trained model
    model = openl3.models.load_audio_embedding_model(input_repr="mel128",
                                                     content_type="music",
                                                     embedding_size=512)

    # Path to the audio file
    datasets_path = 'data/fma_dataset/fma_small/'
    test_folder = '008/'

    output_dir = 'data/fma_dataset/embeddings/'
    file_suffix = 'test_' + test_folder + '_01'

    music_files = read_files(datasets_path + test_folder)

    for sample in music_files:
        file_path = datasets_path + test_folder + sample
        # print(file_path)
        # process_audio_embedding(file_path, file_suffix, output_dir, model)
        print("Embedding for ", sample.split('.')[0], " has been saved.")
        print("track id: ", int(sample.split('.')[0]))