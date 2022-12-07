import math
from multiprocessing import Process

from utils.utils import grouper, seed_all, get_available_gpus
from utils.paths import DatasetPaths
from whisperer.audio_manipulate import convert
from whisperer.whisperer import whisperer
import config.config as CONF


def main():
    number_of_gpus = get_available_gpus()

    seed_all(CONF.seed)
    dataset_name = f"{CONF.dataset_name}_{CONF.seed}"

    print(f"#### Starting pipeline to create dataset {dataset_name} ####")
    dataset_paths = DatasetPaths(__file__, dataset_name)
    dataset_paths.prepare_for_dataset()

    print(
        f"## Converting files in {dataset_paths.AUDIO_FILES} to .wav with frame_rate=16000"
    )
    convert(dataset_paths)
    print("\t--- Done converting to .wav\n")

    print(f"## Detected {number_of_gpus} GPU")
    print(f"## Running whisper on all files in {dataset_paths.AUDIO_FILES_WAV}")
    audio_files = list(dataset_paths.get_audio_files_wav())

    groups = grouper(math.ceil(len(audio_files) / number_of_gpus), audio_files)
    ## Start a new instance of whisperer per GPU
    processes = []
    for idx, group in enumerate(groups):
        device = f"cuda:{idx}"
        p = Process(
            target=whisperer,
            args=(group, dataset_paths.WAVS, dataset_paths.TRANSCRIPTIONS, device),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    dataset_paths.write_to_metadata()

    print(f"## Done creating dataset {dataset_name} ##")


if __name__ == "__main__":
    main()
