seed = 42
dataset_name = "test"

# batch size for OpenAI's Whisper
whisper_model = "base.en"
batch_size = 32


# Configurations to sample audio lengths
loc = 7
scale = 2
lower_bound = 1
upper_bound = 10

# Configuration for audio splitter
frame_lenght = 300
top_db = 40
hop_length = 512

# Values for min and max len text cutoff
min_len = 5
max_len = 180
