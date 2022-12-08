from utils.utils import seed_all
from utils.paths import DatasetPaths
from whisperer.audio_manipulate import convert
from whisperer.whisperer import transcribe
import config.config as CONF
import click


def create_dataset_path(dataset_name):
    seed_all(CONF.seed)

    dataset_name = f"{dataset_name}_{CONF.seed}"
    dataset_paths = DatasetPaths(__file__,dataset_name)
    return dataset_paths


@click.group()
def pipeline(dataset_name):
    pass

@click.command(help="converts all audio files in data/audio_files to .wav with frame_rate=16000")
@click.argument("--dataset_name", type=str)
def convert(dataset_name):
    dataset_paths = create_dataset_path(dataset_name)

    print(f"## Converting files in {dataset_paths.AUDIO_FILES} to .wav with frame_rate=16000")
    convert(dataset_paths)
    print("\t--- Done converting to .wav\n")


@click.command(help="runs whisperer on all files in data/audio_files_wav")
@click.argument("--dataset_name", type=str)
def transcribe(dataset_name):
    dataset_paths = create_dataset_path(dataset_name)

    convert(dataset_name)

    print(f"## Running whisper on all files in {dataset_paths.AUDIO_FILES_WAV}")
    transcribe(dataset_paths.get_audio_files_wav(),
               dataset_paths.WAVS,
               dataset_paths.TRANSCRIPTIONS)

    dataset_paths.write_to_metadata()

    print(f"## Done creating dataset {dataset_name} ##")    


pipeline.add_command(convert)
pipeline.add_command(transcribe)

if __name__ == "__main__":
    pipeline()

# def main():
#     print(f"#### Starting pipeline to create dataset {dataset_name} ####")
#     dataset_paths = DatasetPaths(__file__,dataset_name)
    

#     print(f"## Converting files in {dataset_paths.AUDIO_FILES} to .wav with frame_rate=16000")
#     convert(dataset_paths)
#     print("\t--- Done converting to .wav\n")

#     print(f"## Running whisper on all files in {dataset_paths.AUDIO_FILES_WAV}")
#     transcribe(dataset_paths.get_audio_files_wav(),
#                dataset_paths.WAVS,
#                dataset_paths.TRANSCRIPTIONS)

#     dataset_paths.write_to_metadata()

#     print(f"## Done creating dataset {dataset_name} ##")


# if __name__ == "__main__":
#     main()
