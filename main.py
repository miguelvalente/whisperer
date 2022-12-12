from utils.utils import seed_all
from utils.paths import DefaultPaths, SpeakerPaths, DatasetPaths
from diarizer.diaraizer import diarize as _diarize
from whisperer.audio_manipulate import convert as _convert
from whisperer.whisperer import transcribe as _transcribe
import config.config as CONF
import click


seed_all(CONF.seed)

@click.group(chain=True)
def cli():
    pass

@cli.command(help="converts all audio files in data/audio_files to .wav with frame_rate=16000")
def convert():
    default_paths =  DefaultPaths(__file__)

    print(f"## Converting files in {default_paths.AUDIO_FILES} to .wav with frame_rate=16000")
    _convert(default_paths)
    print("\t--- Done converting to .wav\n")


@cli.command(help="Finds the speakers in all files in data/audio_files_wav")
def diarize():
    speaker_paths = SpeakerPaths(__file__)
    
    print(f"## Running diarizer on all files in {speaker_paths.AUDIO_FILES_WAV}")
    _diarize(speaker_paths.get_audio_files_wav())
    print(f"\t--- Done diarizing\n")

    

@cli.command(help="runs whisperer on all files in data/audio_files_wav")
@click.argument("dataset_name")
def transcribe(dataset_name):
    dataset_name = f"{dataset_name}_{CONF.seed}"
    dataset_paths = DatasetPaths(__file__,dataset_name)

    print(f"## Running whisper on all files in {dataset_paths.AUDIO_FILES_WAV}")
    _transcribe(dataset_paths.get_audio_files_wav(),
               dataset_paths.WAVS,
               dataset_paths.TRANSCRIPTIONS)

    dataset_paths.write_to_metadata()

    print(f"## Done creating dataset {dataset_name} ##")    


cli()