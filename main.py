from utils.utils import seed_all
from paths import DefaultPaths, SpeakerPaths, DatasetPaths
from diarizer.diarizer import diarize as _diarize
from diarizer.embedder import auto_label as _auto_label
from whisperer.audio_manipulate import convert as _convert
from whisperer.whisperer import transcribe as _transcribe
from pathlib import Path
import config.config as CONF
import typer


seed_all(CONF.seed)

app = typer.Typer()


@app.command()
def convert(data_directory: Path) -> None:
    """
    Convert all audio files in data/audio_files to .wav.

    The converted audio files will be saved in data/audio_files_wav.
    """
    default_paths = DefaultPaths(data_directory)

    print(
        f"## Converting files in {default_paths.RAW_FILES} to .wav with frame_rate=16000"
    )
    _convert(default_paths)
    print("\t--- Done converting to .wav\n")


@app.command()
def diarize(data_directory: Path, join: bool) -> None:
    convert(data_directory)
    """
    Diarize all audio files in data/audio_files_wav.
    Diarized audio files will be saved in data/speakers.

    \b
    ARGUMENTS
        num_speakers: Number of speakers in data/audio_files_wav.

    OPTION
        --join: Join speakers with the same name.
                default: True
    """

    speaker_paths = SpeakerPaths(data_directory)

    print(f"## Diarizing all files in {speaker_paths.WAV_FILES}")
    _diarize(
        speaker_paths.get_wav_files(),
        speaker_paths.SPEAKERS,
        join_speaker=join,
    )
    print("\t--- Done diarizing\n")


@app.command()
def auto_label(data_directory: Path, num_speakers: int) -> None:
    diarize(data_directory, join=True)
    """
    Auto label all audio files in data/audio_files_wav/speakers

    \b
    ARGUMENTS
        num_speakers: Number of speakers to label

    """

    speaker_paths = SpeakerPaths(data_directory)

    print(f"## Auto labeling all wav files in {speaker_paths.SPEAKERS}")
    _auto_label(
        num_speakers, speaker_paths.get_speakers_wavs(), speaker_paths.SPEAKERS_METADATA
    )
    print("\t--- Done auto labeling\n")


@app.command()
def transcribe(data_directory: Path, dataset_name: str) -> None:
    convert(data_directory)
    """
    Transcribe all audio files. data/speakers must
    has priority over data/audio_files_wav.

    \b
    OPTION
        dataset_name: Name of the dataset.
    """

    dataset_name = f"{dataset_name}_{CONF.seed}"
    dataset_paths = DatasetPaths(data_directory, dataset_name)

    print(f"## Running whisper on all files in {dataset_paths.WAV_FILES}")

    # Check if speakers audio exists
    # Since speakers audio has priority over audio_files_wav
    if dataset_paths.number_of_speakers() > 0:
        audio_files = dataset_paths.get_speakers_wavs()
    else:
        audio_files = dataset_paths.get_wav_files()

    _transcribe(
        audio_files,
        dataset_paths.WAVS_DIR,
        dataset_paths.TRANSCRIPTIONS,
    )

    dataset_paths.write_to_metadata()

    print(f"## Done creating dataset {dataset_name} ##")


@app.callback()
def main(context: typer.Context,):
    """
    Main function of Whisperer.
    Deals with the order of execution of the functions.
    """
    # match context.invoked_subcommand:
    #     case "convert":
    #         pass
    #     case "diarize", "transcribe":
    #         convert()
    #         pass
    #     case "auto_label":
    #         diarize(join=True)
    #     case other:
    #         typer.echo("Please specify a subcommand")


if __name__ == "__main__":
    app()
