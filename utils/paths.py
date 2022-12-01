from pathlib import Path
from typing import List


class DefaultPaths:
    def __init__(self, main_path):
        main_path = Path(main_path)
        self.BASE_PATH = main_path.parent if main_path.is_file() else main_path
        self.DATA_PATH = self.BASE_PATH.joinpath("data")
        self.AUDIO_FILES = self.DATA_PATH.joinpath("audio_files")
        self.AUDIO_FILES_WAV = self.DATA_PATH.joinpath("audio_files_wav")
        self.DATASET_DIR = self.DATA_PATH.joinpath("datasets")

        self.mandatory_paths = [self.DATA_PATH, self.AUDIO_FILES]
        self.paths = [
            self.AUDIO_FILES_WAV,
        ]


    def _make_paths(self):
        self._assert_mandatory_paths()
        self._ensure_audio_files_are_present()

        for path in self.paths:
            path.mkdir(exist_ok=True)

    def _assert_mandatory_paths(self):
        for path in self.mandatory_paths:
            assert path.exists(), f"Path {path} does not exist"

    def _ensure_audio_files_are_present(self):
        if not len(self.get_audio_files()):
            raise FileNotFoundError(f"No audio_files found in {self.AUDIO_FILES}")

    def _assert_wav_files(self, directory: Path):
        for episode in directory.iterdir():
            assert episode.suffix == ".wav", f"File {episode} is not a .wav file"

    def _touch_metadata(self):
        self.METADATA.touch(exist_ok=True)

    def prepare_for_dataset(self, dataset_name: str):
        self.DATASET = self.DATASET_DIR.joinpath(dataset_name)
        self.METADATA = self.DATASET.joinpath("metadata.txt")
        self.WAVS = self.DATASET.joinpath("wavs")

        if self.DATASET.exists():
            raise FileExistsError(f"Dataset {self.DATASET} already exists")
        else:
            self.DATASET.mkdir()
            self.WAVS.mkdir()
            self._touch_metadata()


    def get_audio_files(self) -> List[Path]:
        return [episode for episode in self.AUDIO_FILES.iterdir() if episode.is_file()]

    def get_audio_files_wav(self) -> List[Path]:
        self._assert_wav_files(self.AUDIO_FILES_WAV)
        return [episode for episode in self.AUDIO_FILES_WAV.iterdir()]

    def get_datasets(self) -> List[Path]:
        return [dataset for dataset in self.DATASET_DIR.iterdir() if dataset.is_dir()]
