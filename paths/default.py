from pathlib import Path
from typing import List
import logging

# logging.basicConfig(
#     format="[%(filename)s:%(lineno)d] %(message)s",
#     datefmt="%Y-%m-%d:%H:%M:%S",
#     level=logging.DEBUG,
# )


class DefaultPaths:
    def __init__(self, main_path):
        main_path = Path(main_path)
        self.BASE_PATH = main_path.parent if main_path.is_file() else main_path
        self.DATA_PATH = self.BASE_PATH.joinpath("data")
        self.AUDIO_FILES = self.DATA_PATH.joinpath("audio_files")
        self.AUDIO_FILES_WAV = self.DATA_PATH.joinpath("audio_files_wav")
        self.DATASET_DIR = self.DATA_PATH.joinpath("datasets")

        self.mandatory_paths = [self.DATA_PATH, self.AUDIO_FILES]
        self.paths = [self.AUDIO_FILES_WAV, self.DATASET_DIR]

        self._make_paths()

    def _make_paths(self) -> None:
        self._assert_mandatory_paths()
        self._ensure_audio_files_are_present()

        for path in self.paths:
            path.mkdir(exist_ok=True)

    def _assert_mandatory_paths(self) -> None:
        for path in self.mandatory_paths:
            assert path.exists(), f"Path {path} does not exist"

    def _ensure_audio_files_are_present(self):
        if not len(self.get_audio_files()):
            raise FileNotFoundError(f"No audio_files found in {self.AUDIO_FILES}")

    def _assert_wav_files(self, directory: Path) -> None:
        for audio_file in directory.iterdir():
            if audio_file.is_file():
                assert (
                    audio_file.suffix == ".wav"
                ), f"File {audio_file} is not a .wav file"

    def get_audio_files(self) -> List[Path]:
        return [
            audio_file
            for audio_file in self.AUDIO_FILES.iterdir()
            if audio_file.is_file()
        ]

    def get_audio_files_wav(self) -> List[Path]:
        self._assert_wav_files(self.AUDIO_FILES_WAV)
        return [
            audio_file
            for audio_file in self.AUDIO_FILES_WAV.iterdir()
            if audio_file.is_file()
        ]

    def get_datasets(self) -> List[Path]:
        return [dataset for dataset in self.DATASET_DIR.iterdir() if dataset.is_dir()]
