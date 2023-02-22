import logging
import math
import subprocess
from collections import deque
from multiprocessing import Process
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torchaudio
import whisper
from librosa import effects

import whisperer_ml.config.config as CONF
from whisperer_ml.utils.utils import get_available_gpus, grouper


def initialize_whisperer(
    device: str, fp16: bool
) -> Tuple[whisper.Whisper, whisper.DecodingOptions]:
    print("\tInitializing whisper")
    options = whisper.DecodingOptions(language="en", without_timestamps=True, fp16=fp16)
    model = whisper.load_model(CONF.whisper_model, device=device)

    return model, options


def sampling_seconds(loc: int, scale: int) -> float:
    from scipy.stats import truncnorm

    trunc = truncnorm(
        (CONF.lower_bound - loc) / scale,
        (CONF.upper_bound - loc) / scale,
        loc=loc,
        scale=scale,
    )
    seconds = trunc.rvs(1)[0]
    return seconds


def get_silence_pairs(splits) -> List[Tuple[int, int]]:
    pairs = []
    splits_iter = iter(splits)

    for _ in range(len(splits) - 1):
        start = next(splits_iter, None)
        end = next(splits_iter, None)

        if start is not None and end is not None:
            pairs.append((start[1], end[0]))
        else:
            break

    return pairs


def find_nearest_value(array: List, value: float) -> Tuple[int, int]:
    array = np.asarray(array)
    idx, _ = np.abs(array - value).argmin(axis=0)
    audio_segment_frame = array[idx, 0]
    to_cut_frame = array[idx, 1]
    return audio_segment_frame, to_cut_frame


def find_silent_frame(
    audio: torch.Tensor, frame_rate: int
) -> Tuple[List[Tuple[int, int]], int]:
    seconds = sampling_seconds(CONF.loc, CONF.scale)
    frame = int(seconds * frame_rate)

    silences = []
    counter = 0
    while not silences:
        if frame * counter > audio.shape[0]:
            return None, 1

        # lower = frame
        upper = frame + (16000 * counter)
        voices = effects.split(
            y=audio[:upper],
            frame_length=CONF.frame_lenght,
            top_db=CONF.top_db,
            hop_length=CONF.hop_length,
        )

        silences = get_silence_pairs(voices.tolist())
        counter += 1

    return silences, frame


def transcribe(audio_files: List[Path], wavs_path, transcription_path) -> None:
    try:
        subprocess.check_output("nvidia-smi")
        split_audio_files_into_gpus(audio_files, wavs_path, transcription_path)
    except (
        Exception
    ):  # this command not being found can raise quite a few different errors depending on the configuration
        whisperer(audio_files, wavs_path, transcription_path, "cpu", False)


def split_audio_files_into_gpus(
    audio_files: List[Path], wavs_path: Path, transcriptions_path: Path
) -> None:
    number_of_gpus = get_available_gpus()
    print(f"## Detected {number_of_gpus} GPU")

    groups_audio = grouper(math.ceil(len(audio_files) / number_of_gpus), audio_files)
    processes = []
    for idx, group_audio in enumerate(groups_audio):
        device = f"cuda:{idx}"
        p = Process(
            target=whisperer,
            args=(group_audio, wavs_path, transcriptions_path, device, True),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


def whisperer(
    audio_files_wav: List[Path],
    wavs_path: Path,
    transcription_path: Path,
    device: str,
    fp16: bool,
) -> None:
    logging.debug({f"Device: {device}"})
    model, options = initialize_whisperer(device, fp16)

    if audio_files_wav is None:
        print("\tSkipping appended None. Expected behavior")
        return

    for audio_file in audio_files_wav:
        batch = deque(maxlen=CONF.batch_size)
        print(f"\tTranscribing: {audio_file.name}")

        audio, frame_rate = torchaudio.load(audio_file)
        audio = audio.squeeze()

        seg_idx = 0
        while audio.shape.numel() > frame_rate * 2:
            mels = []
            while len(batch) < batch.maxlen and audio.shape.numel() > frame_rate * 2:
                silences, frame = find_silent_frame(audio, frame_rate)
                if silences is None:
                    audio = audio[:frame]
                    continue
                else:
                    audio_segment_frame, to_cut_frame = find_nearest_value(
                        silences, frame
                    )

                audio_segment = audio[:audio_segment_frame]
                audio = audio[to_cut_frame:]

                if audio_segment.shape.numel() > frame_rate * 10:
                    continue

                mel = whisper.log_mel_spectrogram(audio_segment)
                padded_mel = whisper.pad_or_trim(mel, 3000)

                mels.append(padded_mel)
                batch.append(audio_segment)

            if mels:
                padded_mels = torch.stack(mels).to(device)

                results = model.decode(padded_mels, options)

                for idx, result in enumerate(results):
                    if (
                        len(result.text) < CONF.min_len
                        or len(result.text) > CONF.max_len
                    ):
                        continue

                    export_wav_path = wavs_path.joinpath(
                        audio_file.stem + f"_{seg_idx}.wav"
                    )
                    torchaudio.save(
                        export_wav_path, batch[idx].unsqueeze(dim=0), frame_rate
                    )

                    audio_file.stem
                    with open(
                        transcription_path.joinpath(f"{audio_file.stem}.txt"), "a"
                    ) as f:
                        f.write(f"{export_wav_path.name}|{result.text}\n")

                    seg_idx += 1

                del padded_mels
                del results
                torch.cuda.empty_cache()

            batch.clear()
