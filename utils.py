# Sound event detection using transfer learning with a pretrained model
# The classes are: 'whistle', 'cetaceans_allfreq', 'click', 'allfreq'
# The model is a pretrained ResNet50 with a custom head
# The model is trained on the 4 classes above

# We have a csv file with the following columns: path,start,duration,end,label
# The path is the path to the audio file
# The start and end are the start and end of the sound event in seconds
# The label is the class of the sound event

# The audio files are in the url: https://storage.googleapis.com/datathon2022/dataset1/{filename}.ogg with filename the name of the file in the csv (path column)
# The audio files are .ogg files with 50000 Hz sampling rate in mono

import pandas as pd
import numpy as np
import torch
import librosa, librosa.display
import torchlibrosa
import matplotlib.pyplot as plt
import urllib.request
import zipfile
import os

def download_audio():
    # Download the audio files from the url
    # Download all the audio files from https://storage.googleapis.com/datathon2022/dataset1_all_ogg.zip and save unzip them in the folder 'audio'
    if not os.path.exists('audio'):
        os.makedirs('audio')
        urllib.request.urlretrieve('https://storage.googleapis.com/datathon2022/dataset1_all_ogg.zip', 'audio.zip')
        with zipfile.ZipFile('audio.zip', 'r') as zip_ref:
            zip_ref.extractall('audio')
        os.remove('audio.zip')
    else:
        print('Audio files already downloaded')

def get_audio(path, start, end):
    # Get the audio from the audio folder and the path
    # The path is the path to the audio file
    # The start and end are the start and end of the sound event in seconds
    # The audio is returned
    audio, sr = librosa.load(f'audio/datathon2022/dataset1/{path}.ogg', sr=50000, offset=start, duration=end-start)

    return audio, sr

def get_features(audio, sr):
    # Input: audio and sampling rate
    # Get the features from the audio
    # The features are the log-mel spectrogram
    # The logmel consist on 3 dimensions: x, y, z, where x is the time, y is the mel frequency, and z is the dB scale
    # Since sound events have different durations(number of samples), the 2-d feature arrays are flattened using mean on the frequency axis
    features = get_logmel(audio, sr)
    
    # Time axis will not be used because of the different durations of the sounds, so we mean the features on the frequency
    features = np.mean(features, axis=2)
    
    return features

def get_logmel(audio, sr):
    # Input: audio and sampling rate
    # Get the log-mel spectrogram from the audio
    # The spectrogram is computed with 50000 Hz sampling rate and 2048 window size
    # The spectrogram is converted to log-mel spectrogram with 128 mel bands
    # The features are then normalized with the mean and std of the training set
    # The log-mel spectrogram is returned
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512,
                            win_length=2048, window='hann', center=True, pad_mode='reflect')
    logmel = librosa.power_to_db(spectrogram, ref=1.0, amin=1e-10, top_db=80.0)
    logmel = np.stack([logmel, logmel, logmel])
    logmel = (logmel - 0.5) / 0.5
    
    return logmel

def get_labels(df):
    # Get the labels from the dataframe
    # The labels are the class of the sound event
    # The labels are converted to integers using the following mapping:
    # 'whistle': 0
    # 'cetaceans_allfreq': 1
    # 'click': 2
    # 'allfreq': 3
    # The labels are returned
    labels = df['label'].values
    labels = np.where(labels == 'whistle', 0, labels)
    labels = np.where(labels == 'cetaceans_allfreq', 1, labels)
    labels = np.where(labels == 'click', 2, labels)
    labels = np.where(labels == 'allfreq', 3, labels)
    labels = labels.astype(int)

    return labels

def plot_logmel(logmel):
    # Plot the features with librosa.display.specshow
    # The features are the log-mel spectrogram

    fig, ax = plt.subplots()
    # The Z axis is the dB scale (decibels), the Y axis is the mel frequency, and the X axis is the time
    img = librosa.display.specshow(logmel[:][:][0], x_axis='time', y_axis='mel', sr=50000, hop_length=512, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Log-Mel spectrogram')
    plt.savefig("temp.png")

def split_dataset(df, train_size=0.8):
    # Split the dataset into train and test
    # The train_size is the percentage of the dataset used for training
    # The test_size is the percentage of the dataset used for testing
    # The train and test datasets are returned
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_df = df[:int(len(df)*train_size)]
    test_df = df[int(len(df)*train_size):]
    
    return train_df, test_df

def save_predictions(predictions, filename):
    # Save the predictions to a csv file
    # The predictions are the predicted labels
    # The filename is the name of the csv file
    # The csv file is saved
    df = pd.DataFrame({'label': predictions})
    df.to_csv(filename, index=False)

class SubmarineAudioDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio, sr = get_audio(row['path'], row['start'], row['end'])
        features = get_features(audio, sr)
        label = get_labels(self.df)[idx]

        # Prepare the features to be fed to mobileNet (3 channels)
        features = np.stack([features, features, features])

        return features, label
