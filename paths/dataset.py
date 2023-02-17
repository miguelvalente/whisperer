from pathlib import Path
from typing import List
from paths.default import SpeakerPaths
import logging


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
