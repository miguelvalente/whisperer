from pathlib import Path
from typing import List
from paths.default import DefaultPaths
import logging


class SpeakerPaths(DefaultPaths):
    def __init__(self, main_path):
        super().__init__(main_path)
        self.SPEAKERS = self.AUDIO_FILES_WAV.joinpath("speakers")
        self.SPEAKERS_METADATA = self.SPEAKERS.joinpath("spekers_metadata.txt")

        self.paths = [self.SPEAKERS]
        self._check_audio_files_wav_presence()

        self._make_paths()

    def _check_audio_files_wav_presence(self) -> None:
        if not len(self.get_audio_files_wav()):
            logging.error(
                f"No audio_files_wav found in {self.AUDIO_FILES_WAV}."
                " Please place audio files in 'data/audio_files' and run:\n"
                "\tpython main.py convert"
            )
            exit(1)

    def number_of_speakers(self) -> int:
        return len(self.get_speakers_wavs())

    def get_speakers_wavs(self) -> List[Path]:
        return [
            wav
            for wav in self.SPEAKERS.iterdir()
            if wav.is_file() and wav.suffix == ".wav"
        ]
