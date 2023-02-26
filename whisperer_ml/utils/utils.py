import collections
import os
import random
from itertools import islice, zip_longest

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


def grouper(n, iterable, padvalue=None):
    # grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')
    return list(zip_longest(*[iter(iterable)] * n, fillvalue=padvalue))


def get_available_gpus():
    from os import environ
    from subprocess import check_output

    if "CUDA_VISIBLE_DEVICES" in environ:
        return len(environ["CUDA_VISIBLE_DEVICES"].split(","))
    else:
        command = ["nvidia-smi", "-L"]
        output = check_output(command)
        return len(output.splitlines())


def seed_all(seed) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
