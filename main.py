from utils.utils import seed_all
from utils.paths import DatasetPaths
from whisperer.audio_manipulate import convert
from whisperer.whisperer import transcribe
import config.config as CONF


def main():

    seed_all(CONF.seed)
    dataset_name = f"{CONF.dataset_name}_{CONF.seed}"

    print(f"#### Starting pipeline to create dataset {dataset_name} ####")
    dataset_paths = DatasetPaths(__file__,dataset_name)
    

    print(f"## Converting files in {dataset_paths.AUDIO_FILES} to .wav with frame_rate=16000")
    convert(dataset_paths)
    print("\t--- Done converting to .wav\n")

    print(f"## Running whisper on all files in {dataset_paths.AUDIO_FILES_WAV}")
    transcribe(dataset_paths.get_audio_files_wav(),
               dataset_paths.WAVS,
               dataset_paths.TRANSCRIPTIONS)

    dataset_paths.write_to_metadata()

    print(f"## Done creating dataset {dataset_name} ##")


if __name__ == "__main__":
    main()
