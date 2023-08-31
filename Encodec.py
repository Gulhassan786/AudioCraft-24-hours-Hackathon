import subprocess

# /////////////////////Compression////////////////
def compress_audio(input_file_path, output_file):
    # Define the command you want to run
    command = [
        "encodec",
        "-b", "1.5",
        "-f",
        input_file_path,
        output_file
    ]

    # Run the command
    try:
        subprocess.run(command, check=True)
        print("encodec command executed successfully.")
    except subprocess.CalledProcessError as e:
        print("Error: encodec command failed with exit code", e.returncode)

# Example usage of the function
input_file_path = "/content/train_audio/abethr1/XC363501.ogg"
output_file = "compress_audio.ecdc"
compress_audio(input_file_path, output_file)


#///////////////////// Decompression///////////////
def decompress_audio(input_file_path, output_file):
    # Define the command you want to run
    command = [
        "encodec",
        "-f",
        "-r",
        input_file_path,
        output_file
    ]

    # Run the command
    try:
        subprocess.run(command, check=True)
        print("encodec command executed successfully.")
    except subprocess.CalledProcessError as e:
        print("Error: encodec command failed with exit code", e.returncode)

# Example usage of the function
input_file_path = "compress_audio.ecdc"
output_file = "decompress_audio.wav"
decompress_audio(input_file_path, output_file)


# /////////////////////////////Discrete Waveform//////////////////
from encodec import EncodecModel
from encodec.utils import convert_audio
import torchaudio
import torch
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt



# Instantiate a pretrained EnCodec model
model = EncodecModel.encodec_model_24khz()
# The number of codebooks used will be determined bythe bandwidth selected.
# E.g. for a bandwidth of 6kbps, `n_q = 8` codebooks are used.
# Supported bandwidths are 1.5kbps (n_q = 2), 3 kbps (n_q = 4), 6 kbps (n_q = 8) and 12 kbps (n_q =16) and 24kbps (n_q=32).
# For the 48 kHz model, only 3, 6, 12, and 24 kbps are supported. The number
# of codebooks for each is half that of the 24 kHz model as the frame rate is twice as much.
model.set_target_bandwidth(6.0)

# Load and pre-process the audio waveform
wav, sr = torchaudio.load("/content/train_audio/abethr1/XC363501.ogg")
wav = convert_audio(wav, sr, model.sample_rate, model.channels)
wav = wav.unsqueeze(0)

# Extract discrete codes from EnCodec
with torch.no_grad():
    encoded_frames = model.encode(wav)
codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]

# Generate some data
codes = np.random.rand(10, 100, 1)

# Create a line plot of the data
sns.lineplot(x=np.arange(codes.shape[1]), y=codes[0, :, 0])

# Add some customizations
sns.set_style("darkgrid")
plt.xlabel("Time")
plt.ylabel("Discrete code")
plt.title("Discrete waveform")
plt.show()
