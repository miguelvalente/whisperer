from pathlib import Path
from typing import Dict, Tuple
import numpy as np
from collections import defaultdict


def aggregate_dataset_info(dataset_path: Path, metadata_path: Path) -> Dict:
    dataset_info = {}
    with open(metadata_path, "r", encoding="utf-8") as file:
        for idx, line in enumerate(file):
            cols = line.split("|")
            wav_file = dataset_path.joinpaths("wavs", f"{cols[0]}.wav")  # type: ignore
            text = cols[1]
            dataset_info[idx] = {len(text.lower().strip()), len(wav_file)}

    return dataset_info


def calculate_text_to_audio(dataset_info: Dict) -> Dict:
    text_len_to_audio_len = defaultdict(int)

    for key, (text_len, audio_len) in dataset_info.keys():
        text_len_to_audio_len[text_len] += audio_len

    return dict(text_len_to_audio_len)


def calculate_global_stats(text_len_to_audio_len: Dict) -> Tuple[Dict, Dict, Dict]:
    text_vs_avg = {}
    text_vs_median = {}
    text_vs_std = {}
    for text_len, summed_audio in text_len_to_audio_len.items():
        text_vs_avg[text_len] = np.mean(summed_audio)
        text_vs_median[text_len] = np.median(summed_audio)
        text_vs_std[text_len] = np.std(summed_audio)

    return text_vs_avg, text_vs_median, text_vs_std
