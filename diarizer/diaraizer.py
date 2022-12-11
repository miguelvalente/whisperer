from simple_diarizer.diarizer import Diarizer
from sklearn.cluster import AgglomerativeClustering
import numpy as np


def read_speaker_config(speaker_config):
    speakers = []
    with open(speaker_config, "r") as f:
        for line in f:
            speakers.append(line.strip())
    return speakers


def diarize(speakers, audio_files):

    diar = Diarizer(
        embed_model="xvec",  # 'xvec' and 'ecapa' supported
        cluster_method="sc",  # 'ahc' and 'sc' supported
    )

    embeds = {}
    for audio_file in audio_files:
        embeds[audio_file] = get_embeds(diar, audio_file, len(speakers))

    total_embeds = np.concatenate(list(embeds.values()), axis=0)


def clusterize_speakers(total_embeds, num_speakers):
    model = AgglomerativeClustering(
        n_clusters=num_speakers,
        linkage='ward'
        ).fit(total_embeds)

    return model.labels_

def get_embeds(diar, wav_file, num_speakers):
    cleaned_segments, embeds, _, cluster_labels = diar.diarize(
        wav_file, num_speakers=num_speakers, extra_info=True
    ).values()

    speaker_embeds = []
    for speaker_idx in sorted(set(cluster_labels)):

        speaker_embeds.append(embeds[cluster_labels == speaker_idx].mean(axis=0))

    return speaker_embeds
