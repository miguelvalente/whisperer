from utils.utils import seed_all
from utils.paths import DefaultPaths, SpeakerPaths, DatasetPaths
from diarizer.diarizer import diarize as _diarize
from diarizer.embedder import auto_label as _auto_label
from whisperer.audio_manipulate import convert as _convert
from whisperer.whisperer import transcribe as _transcribe
import config.config as CONF
import click


seed_all(CONF.seed)


@click.group(chain=True)
def cli():
    """
        Whisperer: A tool for creating audio-text datasets.
    """
    pass


@cli.command()
def convert():
    """
    Convert all audio files in data/audio_files to .wav.

    The converted audio files will be saved in data/audio_files_wav.
    """

    default_paths = DefaultPaths(__file__)

    print(
        f"## Converting files in {default_paths.AUDIO_FILES} to .wav with frame_rate=16000"
    )
    _convert(default_paths)
    print("\t--- Done converting to .wav\n")


@cli.command()
@click.option("--join", is_flag=True, default=True)
def diarize(join):
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

    speaker_paths = SpeakerPaths(__file__)

    print(f"## Diarizing all files in {speaker_paths.AUDIO_FILES_WAV}")
    _diarize(
        speaker_paths.get_audio_files_wav(),
        speaker_paths.SPEAKERS,
        join_speaker=join,
    )
    print(f"\t--- Done diarizing\n")

@cli.command()
@click.argument("num_speakers", type=int, required=True)
def auto_label(num_speakers):
    """
    Auto label all audio files in data/audio_files_wav/speakers

    \b
    ARGUMENTS
        num_speakers: Number of speakers to label

    """   

    speaker_paths = SpeakerPaths(__file__)

    print(f"## Auto labeling all wav files in {speaker_paths.SPEAKERS}")
    _auto_label(
        num_speakers,
        speaker_paths.get_speakers_wavs(),
        speaker_paths.SPEAKERS_METADATA)
    print(f"\t--- Done auto labeling\n")

@cli.command()
@click.argument("dataset_name")
def transcribe(dataset_name):
    """
    Transcribe all audio files. data/speakers must
    has priority over data/audio_files_wav.
    
    \b
    OPTION
        dataset_name: Name of the dataset.
    """   

    dataset_name = f"{dataset_name}_{CONF.seed}"
    dataset_paths = DatasetPaths(__file__, dataset_name)

    print(f"## Running whisper on all files in {dataset_paths.AUDIO_FILES_WAV}")
    _transcribe(
        dataset_paths.get_audio_files_wav(),
        dataset_paths.WAVS,
        dataset_paths.TRANSCRIPTIONS,
    )

    dataset_paths.write_to_metadata()

    print(f"## Done creating dataset {dataset_name} ##")


cli()
