import math
from multiprocessing import Process
from torch.cuda import device_count

from utils.utils import grouper, seed_all
from utils.paths import DatasetPaths
from whisperer.audio_manipulate import convert
from whisperer.whisperer import whisperer
import config.config as CONF


def main():
    number_of_gpus = device_count()

    seed_all(CONF.seed)
    dataset_name = f"{CONF.dataset_name}_{CONF.seed}"

    print(f"#### Starting pipeline to create dataset {dataset_name} ####")
    paths = DatasetPaths(__file__, dataset_name)
    paths.prepare_for_dataset()

    print(f"## Converting files in {paths.AUDIO_FILES} to .wav with frame_rate=16000")
    convert(paths)
    print("\t--- Done converting to .wav\n")

    print(f"## Detected {number_of_gpus} GPU")
    print(f"## Running whisper on all files in {paths.AUDIO_FILES_WAV}")
    audio_files = list(paths.get_audio_files_wav())

    groups = grouper(math.ceil(len(audio_files) / number_of_gpus), audio_files)
    ## Start a instance of whisperer per GPU
    whisperer(audio_files, paths.AUDIO_FILES_WAV, paths.TRANSCRIPTIONS)
    processes = []
    for group in groups:
        p = Process(target=whisperer, args=(group, paths.AUDIO_FILES_WAV, paths.TRANSCRIPTIONS))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print(f"## Done creating dataset {dataset_name} ##")


if __name__ == "__main__":
    main()
