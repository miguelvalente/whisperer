from pydub import AudioSegment
from pydub.silence import detect_silence
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
            convert_to_wav(audio_file, export_path)


def convert_to_wav(src: Path, dest: Path, frame_rate: Optional[int] = 16000) -> None:
    if src.suffix == ".mp3":
        convert_mp3_to_wav(src, dest)
    elif src.suffix == ".opus":
        convert_opus_to_wav(src, dest)
    elif src.suffix == ".wav":
        print(f"\tAudio File already in .wav format: {dest.name}")
        print(f"\t\tCopying {src.name} to {dest}")
        shutil.copy(src, dest)
    else:
        print(f"\tUnknown extension: {src.suffix}")


def convert_opus_to_wav(opus: Path, wav: Path, frame_rate: Optional[int] = 16000) -> None:
    print("Converting .opus to .wav:")
    print(f"\tSrc: {str(opus)} Dest: {str(wav)}")
    audio = AudioSegment.from_file(opus)
    if frame_rate:
        print(f"\tFrameRate: {frame_rate}")
        audio = audio.set_frame_rate(frame_rate)

    audio.export(str(wav), format="wav")


def convert_mp3_to_wav(mp3: Path, wav: Path, frame_rate: Optional[int] = 16000) -> None:
    print("Converting: .mp3 to .wav:")
    print(f"\tSrc: {str(mp3)} Dest: {str(wav)}")
    audio = AudioSegment.from_mp3(mp3)
    if frame_rate:
        print(f"\tFrameRate: {frame_rate}")
        audio = audio.set_frame_rate(frame_rate)

    audio.export(str(wav), format="mp3")


def cut_random(audio_path: Path) -> List:
    audio = AudioSegment.from_wav(audio_path)
    interval = len(audio) // 4
    selected_silences = []
    for i in range(1, 5):
        start = (i - 1) * interval
        end = i * interval
        if i == 4:
            end = len(audio)

        selected_silences.append([start, end])

    return selected_silences


def cut_on_silences(audio: AudioSegment) -> List:
    silences = detect_silence(
        audio, silence_thresh=int(audio.dBFS - 20), min_silence_len=500
    )
    interval = len(silences) // 5
    selected_silences = [silences[interval * i] for i in range(1, 5)]

    return selected_silences
