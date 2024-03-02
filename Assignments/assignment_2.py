import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA
import sys
from scipy.io import wavfile

SOURCE_DIRECTORY = "/mnt/research/NOS_mri/CSE801A_Spring2024_A2/"
sys.path.append(SOURCE_DIRECTORY)
from plot_joint import plot_joint

print("Start problem 2")
df = pd.read_csv("/mnt/research/NOS_mri/CSE801A_Spring2024_A2/online_shoppers_intention.csv")

# Filter out bounce rates of 0
filtered_df = df[df['BounceRates'] != 0]

# Create arrays for bounce rates of new visitors and returning visitors
new_visitor_bounce_rates = filtered_df[filtered_df['VisitorType'] == 'New_Visitor']['BounceRates']
returning_visitor_bounce_rates = filtered_df[filtered_df['VisitorType'] == 'Returning_Visitor']['BounceRates']

# Create boxplot
plt.boxplot([new_visitor_bounce_rates, returning_visitor_bounce_rates])

# Set x-axis labels
plt.xticks([1, 2], ['New Visitor', 'Returning Visitor'])

# Set y-axis label
plt.ylabel('Bounce Rate')

# Show the plot
# plt.show()

# Group the data by month and calculate the mean bounce rate
monthly_bounce_rate = df.groupby('Month')['BounceRates'].mean()

# Create a bar plot
monthly_bounce_rate.plot(kind='bar')

# Set x-axis label
plt.xlabel('Month')

# Set y-axis label
plt.ylabel('Mean Bounce Rate')

# Show the plot
# plt.show()

print("Completed 2")

print("Start problem 3")

# Load dist2d.csv into a variable X 
X = pd.read_csv(SOURCE_DIRECTORY + "dist2d.csv", header = None)

# Scatter plot X and the corresponding projected distributions of the data
plot_joint(X.values)

# Center X
X_centered = X - X.mean()

# Scatter plot centered X
plot_joint(X_centered.values)

# Perform PCA on centered X
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_centered)

# Scatter plot PCA result
plot_joint(X_pca)

# Perform FastICA on centered X
ica = FastICA(n_components=2)
X_ica = ica.fit_transform(X_centered)

# Scatter plot FastICA result
plot_joint(X_ica)


print("Completed 3")

print("Start problem 4")

# Load the wave files
SOURCE_WAVES_DIRECTORY = "/mnt/home/doggalok/Documents/ITM891_Lokesh/Assignments/Waves/"
wave_files = ['mix_1.wav', 'mix_2.wav', 'mix_3.wav', 'mix_4.wav', 'mix_5.wav']
wave_data = []
for file in wave_files:
    file = SOURCE_WAVES_DIRECTORY + file
    sample_rate, data = wavfile.read(file)
    wave_data.append(data)

# Convert the wave data to a numpy array
wave_data = np.array(wave_data)

# Apply FastICA to unmix the signals
ica = FastICA(n_components=5)
unmixed_signals = ica.fit_transform(wave_data)

# Rescale the unmixed signals to a scale from -1 to 1
rescaled_signals = []
for signal in unmixed_signals.T:
    rescaled_signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) * 2 - 1
    rescaled_signals.append(rescaled_signal)

# Convert the rescaled signals to float32
rescaled_signals = np.array(rescaled_signals, dtype=np.float32)

# Write out the unmixed signals as wave files
for i, signal in enumerate(rescaled_signals):
    wavfile.write(f'unmixed_song{i+1}.wav', sample_rate, signal)

# Plot the time courses of the different unmixed songs
time = np.arange(len(rescaled_signals[0])) / sample_rate
plt.figure(figsize=(10, 6))
for i, signal in enumerate(rescaled_signals):
    plt.plot(time, signal, label=f'Song {i+1}')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
# plt.show()

print("Completed 4")