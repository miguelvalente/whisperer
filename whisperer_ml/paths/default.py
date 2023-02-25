from pathlib import Path
from typing import List

FFMEPG_FORMATS = [
    ".mp3",
    ".wav",
    ".ogg",
    ".flac",
    ".aac",
    ".m4a",
    ".mp4",
    ".avi",
    ".mkv",
]


class DefaultPaths:
    def __init__(self, data_path):
        self.DATA_PATH = Path(data_path)
        self.RAW_FILES = self.DATA_PATH.joinpath("raw_files")
        self.WAV_FILES = self.DATA_PATH.joinpath("wav_files")

        self.mandatory_paths = [self.DATA_PATH, self.RAW_FILES]
        self.paths = [self.WAV_FILES]

        self._make_paths()

    def _make_paths(self) -> None:
        self._assert_mandatory_paths()
        self._are_raw_files_present()

        for path in self.paths:
            path.mkdir(exist_ok=True)

    def _assert_mandatory_paths(self) -> None:
        for path in self.mandatory_paths:
            assert (
                path.exists()
            ), f"Path {path} does not exist\n The following paths are mandatory: {self.mandatory_paths}"

    def _are_raw_files_present(self) -> None:
        if not len(self.get_raw_files()):
            raise FileNotFoundError(f"No files found in {self.RAW_FILES}")

    def _are_wav_files_present(self) -> None:
        if not len(self.get_wav_files()):
            raise FileNotFoundError(f"No files found in {self.WAV_FILES}")

    def get_raw_files(self) -> List[Path]:
        raw_files = []
        for raw_file in self.RAW_FILES.iterdir():
            if raw_file.is_file():
                assert (
                    raw_file.suffix in FFMEPG_FORMATS
                ), f"File {raw_file} is not a valid audio file\n Allowed formats: {FFMEPG_FORMATS}"
                raw_files.append(raw_file)

        return raw_files

    def get_wav_files(self) -> List[Path]:
        return [
            wav
            for wav in self.WAV_FILES.iterdir()
            if wav.is_file() and wav.suffix == ".wav"
        ]
