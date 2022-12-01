import os
import collections
from itertools import islice
import random
import numpy as np
import torch


def formatter(root_path, manifest_file, **kwargs):
    """Assumes each line as ```<filename>|<transcription>```"""
    txt_file = os.path.join(root_path, manifest_file)
    items = []
    speaker_name = "dc"
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = os.path.join(root_path, "wavs", cols[0]) + ".wav"
            text = cols[1]
            items.append(
                {
                    "text": text,
                    "audio_file": wav_file,
                    "speaker_name": speaker_name,
                    "root_path": root_path,
                }
            )
    return items


def sliding_window(iterable: iter, n: int) -> iter:
    # sliding_window('ABCDEFG', 4) --> ABCD BCDE CDEF DEFG
    it = iter(iterable)
    window = collections.deque(islice(it, n), maxlen=n)
    if len(window) == n:
        yield tuple(window)
    for x in it:
        window.append(x)
        yield tuple(window)


def seed_all(seed) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
