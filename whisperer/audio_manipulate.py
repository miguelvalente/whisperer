from pydub import AudioSegment
from pathlib import Path
import shutil
from utils.paths import DefaultPaths
from typing import Optional, List


def convert(paths: DefaultPaths) -> None:
    audio_files = paths.get_audio_files()

    for audio_file in audio_files:
        export_path = paths.AUDIO_FILES_WAV.joinpath(audio_file.stem + ".wav")

        if export_path.exists():
            print(f"\tAudio File already in .wav format: {export_path.name} ")
        else:
            convert_to_wav(audio_file, export_path, frame_rate=16000)


def convert_to_wav(src: Path, dest: Path, frame_rate: Optional[int] = 16000) -> None:
    if src.suffix == ".mp3":
        mp3_to_wav(src, dest, frame_rate)
    elif src.suffix == ".opus":
        opus_to_wav(src, dest, frame_rate)
    elif src.suffix == ".wav":
        print(f"\tAudio File already in .wav format: {dest.name}")
        print(f"\tFramerate: {frame_rate}")
        audio = AudioSegment.from_wav(file=src)
        audio = audio.set_frame_rate(frame_rate)
        audio.export(str(dest), format="wav")
    else:
        print(f"\tUnknown extension: {src.suffix}")

def opus_to_wav(opus: Path, wav: Path, frame_rate: Optional[int] = 16000) -> None:
    print("Converting .opus to .wav:")
    print(f"\tSrc: {str(opus)} Dest: {str(wav)}")
    audio = AudioSegment.from_file(opus)
    if frame_rate:
        print(f"\tFrameRate: {frame_rate}")
        audio = audio.set_frame_rate(frame_rate)

    audio.export(str(wav), format="wav")


def mp3_to_wav(mp3: Path, wav: Path, frame_rate: Optional[int] = 16000) -> None:
    print("Converting: .mp3 to .wav:")
    print(f"\tSrc: {str(mp3)} Dest: {str(wav)}")
    audio = AudioSegment.from_mp3(mp3)
    if frame_rate:
        print(f"\tFrameRate: {frame_rate}")
        audio = audio.set_frame_rate(frame_rate)

    audio.export(str(wav), format="mp3")
