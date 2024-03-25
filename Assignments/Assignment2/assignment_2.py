import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA
import sys
from scipy.stats import ttest_ind
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

# Perform the t-test on the two independent samples
t_stat, p_value = ttest_ind(new_visitor_bounce_rates, returning_visitor_bounce_rates, equal_var=False)

# Print the t-statistic and the p-value
print(f'T-statistic: {t_stat}, P-value: {p_value}')

# Conclusions
alpha = 0.05  # Significance level
if p_value < alpha:
    print("There is a significant difference in bounce rates between new and returning visitors.")
else:
    print("There is no significant difference in bounce rates between new and returning visitors.")

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
fs, mix1 = wavfile.read('/mnt/home/doggalok/Documents/ITM891_Lokesh/Assignments/Waves/mix_1.wav')
fs, mix2 = wavfile.read('/mnt/home/doggalok/Documents/ITM891_Lokesh/Assignments/Waves/mix_2.wav')
fs, mix3 = wavfile.read('/mnt/home/doggalok/Documents/ITM891_Lokesh/Assignments/Waves/mix_3.wav')
fs, mix4 = wavfile.read('/mnt/home/doggalok/Documents/ITM891_Lokesh/Assignments/Waves/mix_4.wav')
fs, mix5 = wavfile.read('/mnt/home/doggalok/Documents/ITM891_Lokesh/Assignments/Waves/mix_5.wav')
 
# Combine the mixed signals into a single matrix
mixed_signals = np.column_stack((mix1, mix2, mix3, mix4, mix5))
 
# Perform FastICA to unmix the signals
ica = FastICA(n_components=5)
unmixed_signals = ica.fit_transform(mixed_signals)
 
# Rescale the unmixed signals to a scale from -1 to 1
unmixed_signals = unmixed_signals / np.max(np.abs(unmixed_signals))
 
# Write out the unmixed signals as wave files
wavfile.write('unmixed1.wav', fs, unmixed_signals[:, 0].astype(np.float32))
wavfile.write('unmixed2.wav', fs, unmixed_signals[:, 1].astype(np.float32))
wavfile.write('unmixed3.wav', fs, unmixed_signals[:, 2].astype(np.float32))
wavfile.write('unmixed4.wav', fs, unmixed_signals[:, 3].astype(np.float32))
wavfile.write('unmixed5.wav', fs, unmixed_signals[:, 4].astype(np.float32))
  
# Plot the time courses of the different unmixed songs
plt.figure(figsize=(10, 6))
# Iterate over the columns of unmixed_signals
for i in range(unmixed_signals.shape[1]):
    plt.figure(figsize=(10, 6))  # Create a new figure for each signal
    plt.plot(np.arange(len(unmixed_signals)), unmixed_signals[:, i])
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title(f'Unmixed Signal {i+1}')
    plt.show()

plt.tight_layout()
# plt.show()

print("Completed 4")