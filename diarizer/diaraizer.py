from simple_diarizer.diarizer import Diarizer
from sklearn.cluster import AgglomerativeClustering
import numpy as np


def read_speaker_config(speaker_config):
    speakers = []
    with open(speaker_config, "r") as f:
        for line in f:
            speakers.append(line.strip())
    return speakers




def clusterize_speakers(total_embeds, num_speakers):
    model = AgglomerativeClustering(
        n_clusters=num_speakers,
        linkage='ward'
        ).fit(total_embeds)

    return model.labels_

def diarize_audio(diar, wav_file, num_speakers):
     
    cleaned_segments, embeds, _, cluster_labels = diar.diarize(
        wav_file, num_speakers=num_speakers, extra_info=True
    ).values()

    speaker_embeds = []
    for speaker_idx in range(num_speakers):

        speaker_embeds.append(embeds[cluster_labels == speaker_idx].mean(axis=0))

    return speaker_embeds, cleaned_segments


# a function that asks for input and valites it as int
def get_int_input(audio_file):
    prompt = f"Please enter the number of speakers in: {audio_file.name}"
    while True:
        try:
            user_input = int(input(prompt))
        except ValueError:
            print("Please enter a valid integer")
            continue
        else:
            return user_input
    

def diarize(audio_files):
    #lambda to get the first element of a tuple
    get_embed = lambda x: x[0]
    get_clean_segmetn = lambda x: x[0]

    diar = Diarizer(
        embed_model="xvec",  # 'xvec' and 'ecapa' supported
        cluster_method="sc",  # 'ahc' and 'sc' supported
    )

    audios_diarized = {}
    for audio_file in audio_files:
        if audio_file.name == 'lex_debate_full.wav':
            audios_diarized[audio_file.name] = diarize_audio(diar, audio_file, 3)
        else:
            audios_diarized[audio_file.name] = diarize_audio(diar, audio_file, 2)
        

    embeds = list(map(get_embed, audios_diarized.values()))
    embeds = list(map(get_embed, audios_diarized.values()))
    total_embeds = np.concatenate(embeds, axis=0)

    cluster_labels = clusterize_speakers(total_embeds, 5)





