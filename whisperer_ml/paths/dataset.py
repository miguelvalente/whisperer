import logging
from pathlib import Path
from typing import List

from whisperer_ml.paths.speaker import SpeakerPaths


class DatasetPaths(SpeakerPaths):
    def __init__(self, data_path, dataset_name):
        super().__init__(data_path)
        self.DATASET_DIR = self.DATA_PATH.joinpath("datasets")
        self.DATASET = self.DATASET_DIR.joinpath(dataset_name)
        self.TRANSCRIPTIONS = self.DATASET.joinpath("transcriptions")
        self.WAVS_DIR = self.DATASET.joinpath("wavs")
        self.METADATA = self.DATASET.joinpath("metadata.txt")

        self.paths = [
            self.DATASET_DIR,
            self.DATASET,
            self.TRANSCRIPTIONS,
            self.WAVS_DIR,
        ]

        self._make_dataset_strucutre()

    def _make_dataset_strucutre(self) -> None:
        if self.DATASET.exists():
            logging.error(
                f"Dataset {self.DATASET} already exists.\n Delete folder or choose a different dataset name"
            )
        else:
            self._make_paths()
            self.METADATA.touch(exist_ok=True)

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
