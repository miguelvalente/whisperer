from utils.utils import seed_all

from utils.paths import DatasetPaths
from whisperer.audio_manipulate import convert
from whisperer.whisperer import whisperer
import config.config as CONF


def main():
    seed_all(CONF.seed)
    dataset_name = f"{CONF.dataset_name}_{CONF.seed}"

    print(f"#### Starting pipeline to create dataset {dataset_name} ####")
    paths = DatasetPaths(__file__)
    paths.prepare_for_dataset(dataset_name)

    print(f"## Converting files in {paths.AUDIO_FILES} to .wav with frame_rate=16000")
    convert(paths)
    print("\t--- Done converting to .wav\n")

    print(f"## Running whisper on all files in {paths.AUDIO_FILES_WAV}")
    whisperer(paths)

    print(f"## Done creating dataset {dataset_name} ##")


if __name__ == "__main__":
    main()
