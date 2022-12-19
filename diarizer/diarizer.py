from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
import torch
import torchaudio
from collections import defaultdict

from typing import List, Tuple
from pathlib import Path


def get_int_input(audio_file):
    prompt = f"Please enter the number of speakers in: {audio_file.name}"
    while True:
        try:
            user_input = int(input(prompt))
        except ValueError:
            print("Please enter a valid integer")
            continue
        else:
            return user_input



def clusterize_speakers(total_embeds, num_speakers):
    model = AgglomerativeClustering(n_clusters=num_speakers, linkage="ward").fit(
        total_embeds
    )

    return model.labels_


def diarize_audio(pipeline, wav_file, num_speakers=None):
    diarization = pipeline(str(wav_file))

    fresh_cuts = diarization.extrude(diarization.get_overlap(), "intersection")

    concated_speakers = []
    for turn, _, speaker in fresh_cuts.itertracks(yield_label=True):
        if not concated_speakers:
            concated_speakers.append([turn.start, turn.end, speaker])
        else:
            if concated_speakers[-1][2] == speaker:
                concated_speakers[-1][1] = turn.end
            else:
                concated_speakers.append([turn.start, turn.end, speaker])

    return concated_speakers


def get_embeds(model_embeder, audio_path, concated_speaker):
    embeds = defaultdict(list)
    audio, _ = torchaudio.load(audio_path)
    with torch.no_grad():
        for start, end, speaker in concated_speaker:
            start_ = int(start * 16000)
            end_ = int(end * 16000)
            if (end_ - start_) / 16000 < 0.35:
                continue

            results = model_embeder(audio[:, start_:end_]).detach().numpy()
            embeds[f"{audio_path.name}_{speaker}"].append(results)

    return embeds


def export_joined_speaker_segment(
    speakers_path: Path, audio_path: Path, speakers_segments: List[Tuple[int, int, str]]
) -> defaultdict:
    '''
    Join the diariazed segments of each speaker and
    export the joined audio segment of each speaker in a wav file
    '''
    audio, sampling_rate = torchaudio.load(audio_path)

    speakers = defaultdict(list)
    for start_, end_, speaker in speakers_segments:
        start = int(start_ * sampling_rate)
        end = int(end_ * sampling_rate)

        speakers[speaker].append(audio[:, start:end])
        speakers[speaker].append(torch.zeros((1, 16000)))

    for speaker, speakers_audio in speakers.items():
        speaker_path = speakers_path.joinpath(f"{audio_path.stem}_{speaker}.wav")
        torchaudio.save(speaker_path, torch.cat(speakers_audio, axis=1), sampling_rate)


def export_speaker_segments(
    speakers_path: Path, audio_path: Path, speakers_segments: List[Tuple[int, int, str]]
):
    audio, sampling_rate = torchaudio.load(audio_path)

    for idx, (start_, end_, speaker) in enumerate(speakers_segments):
        start = int(start_ * sampling_rate)
        end = int(end_ * sampling_rate)
        audio_segment = audio[:, start:end]

        speaker_path = speakers_path.joinpath(f"{audio_path.stem}_{speaker}_{idx}.wav")
        torchaudio.save(speaker_path, audio_segment, sampling_rate)



def diarize(audio_files: List[Path], speakers_path: Path, join_speaker: bool) -> None:
    from pyannote.audio import Pipeline

    diarizing_pipeling = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token="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",
    )

    for audio_file in tqdm(audio_files, desc="Diarizing"):
        speakers_segments = diarize_audio(diarizing_pipeling, audio_file)
        if join_speaker:
            export_joined_speaker_segment(speakers_path, audio_file, speakers_segments)
        else:
            export_speaker_segments(speakers_path, audio_file, speakers_segments)