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
        return [wav for wav in self.SPEAKERS.iterdir() if wav.is_file() and wav.suffix == ".wav"]


class DatasetPaths(SpeakerPaths):
    def __init__(self, main_path, dataset_name):
        super().__init__(main_path)
        self.DATASET = self.DATASET_DIR.joinpath(dataset_name)
        self.TRANSCRIPTIONS = self.DATASET.joinpath("transcriptions")
        self.WAVS_DIR = self.DATASET.joinpath("wavs")
        self.METADATA = self.DATASET.joinpath("metadata.txt")

        self.paths = [self.DATASET, self.TRANSCRIPTIONS, self.WAVS_DIR]

        self._check_audio_files_wav_presence()
        self._prepare_for_dataset()

    def _check_audio_files_wav_presence(self) -> None:
        if not len(self.get_audio_files_wav()):
            logging.error(
                f"No audio_files_wav found in {self.AUDIO_FILES_WAV}."
                " Please place audio files in 'data/audio_files' and run:\n"
                "\tpython main.py convert"
            )
            exit(1)

    def _touch_metadata(self) -> None:
        self.METADATA.touch(exist_ok=True)

    def _prepare_for_dataset(self) -> None:
        if self.DATASET.exists():
            logging.error(
                f"Dataset {self.DATASET} already exists. Delete folder or choose a different dataset name"
            )
        else:
            self._make_paths()
            self._touch_metadata()

    def get_transcriptions(self) -> List[Path]:
        return [
            transcription
            for transcription in self.TRANSCRIPTIONS.iterdir()
            if transcription.is_file()
        ]

    def reads_transcriptions(self) -> List[str]:
        transcriptions = []
        for transcription in self.get_transcriptions():
            with open(transcription, "r") as f:
                transcriptions.append(f.read())
        return transcriptions

    def write_to_metadata(self) -> None:
        transcriptions = self.reads_transcriptions()
        with open(self.METADATA, "a") as f:
            for transcription in transcriptions:
                f.write(transcription)
