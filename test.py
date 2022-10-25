import utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa

# Test the get_audio function and get_features function

# Load the csv file
df = pd.read_csv('labels_dataset1_v2.csv')

# Delete all the rows with label 'volcano'
df = df[df['label'] != 'volcano']

# row index example is 34
row_index = 64

# Get the audio from the first row (get_audio(path, start, end))
audio, sr = utils.get_audio(df.iloc[row_index]['path'], df.iloc[row_index]['start'], df.iloc[row_index]['end'])

# Get the logmel from the audio
logmel = utils.get_logmel(audio, sr)

# Print the shape of the logmel
print(logmel.shape)
print(logmel)

# Plot the log-mel spectrogram
utils.plot_logmel(logmel)

# Print the row
print(df.iloc[row_index])

# Explain the shape of the logmel
# The shape of the logmel is (3, 128, variable)
# The first dimension is the number of channels
# The second dimension is the number of mel frequency bins
# The third dimension is the number of frames

# We don't know the number of frames because the duration of the sound event is different for each row
# The number of frames is computed using the formula: (duration * sampling_rate) / hop_length
# The hop_length is 512, so the number of frames is (duration * 50000) / 512

# The number of frames is different for each row, so the shape of the logmel is (3, 128, variable)
# The number of frames must be the same for each row, so we need to transform the logmel to a 2-d array
# The dimension of the 2-d array is (3, 128)
# The 2-d array is computed using the mean on the time axis

logmel = np.mean(logmel, axis=2)
print(logmel.shape)

# The shape of the logmel is now (3, 128)
# lineplot
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
ax.plot(logmel[0])
ax.plot(logmel[1])
ax.plot(logmel[2])
# title
ax.set_title('Log-mel spectrogram')
# x-axis
ax.set_xlabel('Mel frequency')
# y-axis
ax.set_ylabel('dB')
# legend
ax.legend(['Channel 1', 'Channel 2', 'Channel 3'])
plt.savefig('lineplot.png')