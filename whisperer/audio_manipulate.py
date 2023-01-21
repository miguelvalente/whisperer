from pathlib import Path
from utils.paths import DefaultPaths
from typing import Optional, List
import torchaudio
import subprocess


def convert(paths: DefaultPaths, sample_rate: Optional[int] = 22050) -> None:
    audio_files = paths.get_audio_files()

    for audio_file in audio_files:
        export_path = paths.AUDIO_FILES_WAV.joinpath(audio_file.stem + ".wav")

        if export_path.exists():
            if check_wav_khz_mono(export_path):
                print(
                    f"\tAudio File already in correct .wav format: {export_path.name} "
                )
            else:
                convert_to_wav(audio_file, export_path, sample_rate=sample_rate)
        else:
            convert_to_wav(audio_file, export_path, sample_rate=sample_rate)


def check_wav_khz_mono(audio_file: Path, sample_rate: int) -> bool:
    """
    Returns True if a wav file is 16khz and single channel
    """
    try:
        signal, fs = torchaudio.load(audio_file)

        mono = signal.shape[0] == 1
        freq = fs == sample_rate
        if mono and freq:
            return True
        else:
            return False
    except:
        return False


def check_ffmpeg():
    """
    Returns True if ffmpeg is installed
    """
    try:
        subprocess.check_output("ffmpeg", stderr=subprocess.STDOUT)
        return True
    except OSError as e:
        return False


def convert_to_wav(
    audio_file: Path, wav_audio_file: Path, sample_rate: Optional[int] = 22050
) -> None:
    """
    Converts file to 22khz single channel mono wav with ffpemg
    """
    command = 

    subprocess.Popen(command, shell=True).wait()
