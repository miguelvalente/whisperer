seed = 42

# batch size for OpenAI's Whisper
whisper_model = "base.en"
batch_size = 16


# Configurations to sample audio lengths
loc = 7  # mean of the normal gaussian for sampling
scale = 2
lower_bound = 1  # lower bound for truncated gaussian
upper_bound = 10  # upper bound for truncated gaussian

# Configuration for audio splitter
frame_lenght = 300
top_db = 40
hop_length = 512

# Values for min and max len text cutoff
min_len = 5
max_len = 180
