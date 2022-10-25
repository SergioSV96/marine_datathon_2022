# Sound event detection using transfer learning with a pretrained model
# The classes are: 'whistle', 'cetaceans_allfreq', 'click', 'allfreq'
# The model is a pretrained mobilenet with a custom head
# The model is trained on the 4 classes above

# We have a csv file with the following columns: path,start,duration,end,label
# The path is the path to the audio file
# The start and end are the start and end of the sound event in seconds
# The label is the class of the sound event

# The audio files are in the url: https://storage.googleapis.com/datathon2022/dataset1/{filename}.ogg with filename the name of the file in the csv (path column)
# The audio files are .ogg files with 50000 Hz sampling rate in mono

import utils
import model
import pandas as pd
import torch

# Download the audio files from the url
utils.download_audio()

# Load the csv file
df = pd.read_csv('labels_dataset1_v2.csv')

# Delete all the rows with label 'volcano'
df = df[df['label'] != 'volcano']

# Split the dataset into train and test
# The train_size is the percentage of the dataset used for training

train_df, test_df = utils.split_dataset(df, train_size=0.8)

# To train the model, we need the features and the labels
# The features are the log-mel spectrogram of the audio
# The labels are the class of the sound event

# Load the data for the model using SubmarineAudioDataset (torch.utils.data.Dataset)
train_dataset = utils.SubmarineAudioDataset(train_df)
test_dataset = utils.SubmarineAudioDataset(test_df)
  
# DataLoaders are used to load the data in batches
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Create the pytorch model
model = model.SubmarineAudioModel()

# Train the model
model.fit(train_loader, test_loader, epochs=10, lr=0.001)

# Evaluate the model
model.evaluate(test_loader)

# Predict the labels for the test dataset
predictions = model.predict(test_loader)

# Save the predictions to a csv file
utils.save_predictions(predictions, 'predictions.csv')