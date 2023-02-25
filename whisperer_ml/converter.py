import subprocess
from pathlib import Path
from typing import List, Optional

import torchaudio


def convert(raw_files: List[Path], WAV_FILES: Path) -> None:
    for audio_file in raw_files:
        export_path = WAV_FILES.joinpath(audio_file.stem + ".wav")

        if export_path.exists():
            if check_wav_16khz_mono(export_path):
                print(
                    f"\tAudio File already in correct .wav format: {export_path.name} "
                )
            else:
                convert_to_wav(audio_file, export_path, frame_rate=16000)
        else:
            convert_to_wav(audio_file, export_path, frame_rate=16000)


def check_wav_16khz_mono(audio_file: Path) -> bool:
    """
    Returns True if a wav file is 16khz and single channel
    """
    try:
        signal, fs = torchaudio.load(audio_file)

        mono = signal.shape[0] == 1
        freq = fs == 16000
        if mono and freq:
            return True
        else:
            return False
    except Exception:
        return False


def check_ffmpeg():
    """
    Returns True if ffmpeg is installed
    """
    try:
        subprocess.check_output("ffmpeg", stderr=subprocess.STDOUT)
        return True
    except OSError as e:
        print("ffmpeg not found. Please install ffmpeg and try again.")
        return False


def convert_to_wav(
    audio_file: Path, wav_audio_file: Path, frame_rate: Optional[int] = 16000
) -> None:
    """
    Converts file to 16khz single channel mono wav
    """
    command = f'ffmpeg -y -i "{audio_file}" -acodec pcm_s16le -ar {frame_rate} -ac 1 "{wav_audio_file}"'

    subprocess.Popen(command, shell=True).wait()
