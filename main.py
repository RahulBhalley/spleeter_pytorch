from separator_torch import Separator, params

# Define input audio file name and directory
audio_filename = "dubbed_hi_trimmed.wav"
audio_dir = '.'

# Create Separator instance with given parameters
separator = Separator(params)

# Separate audio into components
# The separate() method takes input audio path and output directory as arguments
separator.separate(f"{audio_dir}/{audio_filename}", "./output")