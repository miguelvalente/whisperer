from pathlib import Path
from typing import List

from whisperer_ml.paths.default import DefaultPaths


class SpeakerPaths(DefaultPaths):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.SPEAKERS = self.WAV_FILES.joinpath("speakers")
        self.SPEAKERS_METADATA = self.SPEAKERS.joinpath("speakers_metadata.txt")

        self.paths = [self.SPEAKERS]

        self._make_paths()

    def _make_paths(self) -> None:
        self._assert_mandatory_paths()
        self._are_wav_files_present()

        for path in self.paths:
            path.mkdir(exist_ok=True)

    def number_of_speakers(self) -> int:
        return len(self.get_speakers_wavs())

    def get_speakers_wavs(self) -> List[Path]:
        return [
            wav
            for wav in self.SPEAKERS.iterdir()
            if wav.is_file() and wav.suffix == ".wav"
        ]
