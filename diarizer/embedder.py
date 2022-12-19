import numpy as np
from pathlib import Path 
from sklearn.cluster import AgglomerativeClustering
from typing import List
import torch
import torchaudio
from sklearn.metrics import pairwise_distances
from speechbrain.pretrained import EncoderClassifier

def similarity_matrix(embeds, metric="cosine"):
    return pairwise_distances(embeds, metric=metric)

def read_speaker_config(speaker_config):
    speakers = []
    with open(speaker_config, "r") as f:
        for line in f:
            speakers.append(line.strip())
    return speakers

def embed(audio_path: Path, embedder: EncoderClassifier):
    '''
    Embed the audio file using the pretrained model
    '''
    audio, sr = torchaudio.load(audio_path)
    # with torch.no_grad():
    #     embedding = embedder.encode_batch(audio)
    #     embedding = embedding.flatten().cpu().numpy()
    embedding = np.random.rand(192)

    return embedding

def label_embeddings(sim_matrix: np.array, num_speakers: int) -> List[str]:
    '''
    Label the embeddings using Agglomerative Clustering
    '''

    clusterer = AgglomerativeClustering(
        n_clusters=num_speakers,
        affinity="precomputed",
        linkage="average"
        ).fit(sim_matrix)

    return clusterer.labels_


def auto_label(num_speakers: int, audio_files: List[Path]) -> None:
    embedder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts="cuda",
    )

    embeds = []
    for audio_path in audio_files:
        embeds.append(embed(audio_path, embedder))

    sim_matrix = similarity_matrix(embeds)

    labels = label_embeddings(sim_matrix, num_speakers)

    print()