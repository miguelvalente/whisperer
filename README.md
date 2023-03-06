
# whisperer

Go from raw audio files to a speaker separated text-audio datasets automatically.

![plot](https://github.com/miguelvalente/whisperer/blob/master/logo.png?raw=true)


## Table of Contents

- [Summary](#summary)
- [Key Features](#key-features)
- [Instalation](#instalation)
- [How to use:](#how-to-use)
   - [Using Multiple-GPUS](#using-multiple-gpus)
   - [Configuration](#configuration)
- [To Do](#to-do)
- [Acknowledgements](#acknowledgements)

## Summary

This repo takes a directory of audio files and converts them to a text-audio dataset with normalized distribution of audio lengths. *See ```AnalyzeDataset.ipynb``` for examples of the dataset distributions across audio and text length*

The output is a text-audio dataset that can be used for training a speech-to-text model or text-to-speech.
The dataset structure is as follows:
```
│── /dataset
│   ├── metadata.txt
│   └── wavs/
│      ├── audio1.wav
│      └── audio2.wav
```

metadata.txt
```
peters_0.wav|Beautiful is better than ugly.
peters_1.wav|Explicit is better than implicit.

```

## Key Features

* Audio files are automatically split by speakers
* Speakers are auto-labeled across the files
* Audio splits on silences
* Audio splitting is configurable
* The dataset creation is done so that it follows Gaussian-like distributions on clip length. Which, in turn, can lead to Gaussian-like distributions on the rest of the dataset statistics. Of course, this is highly dependent on your audio sources.
* Leverages the GPUs available on your machine. GPUs also be set explicitly if you only want to use some.


## Instalation
You have two options

1. Install from PyPi with pip

```
pip install whisperer-ml
```

2. User Friendly WebApp
[Whisperer Web](https://github.com/miguelvalente/whisperer_ml_app)

Note: _Under Development but ready to be used_

## How to use:


1. Create data folder and move audio files to it
```
mkdir data data/raw_files
```
2. There are four commands
   1. Convert
      ```
      whisperer_ml convert path/to/data/raw_files
      ```
   2. Diarize 
      ```
      whisperer_ml diarize path/to/data/raw_files
      ```
   3. Auto-Label 
      ```
      whisperer_ml auto-label path/to/data/raw_files number_speakers
      ```
   4. Transcribe 
      ```
      whisperer_ml transcribe path/to/data/raw_files your_dataset_name
      ```
   5. Help lists all commands 
      ```
      whisperer_ml --help 
      ```
   6. You can run help on a specific command
   ```
      whisperer_ml convert --help
   ```


3. Use the ```AnalyseDataset.ipynb``` notebook to visualize the distribution of the dataset
4. Use the ```AnalyseSilence.ipynb``` notebook to experiment with silence detection configuration

### Using Multiple-GPUS

The code automatically detects how many GPU's are available and distributes the audio files in ```data/wav_files``` evenly across the GPUs.
The automatic detection is done through ```nvidia-smi```.

You can to make the available GPU's explicit by setting the environment variable ```CUDA_AVAILABLE_DEVICES```.

### Configuration

Modify `config.py` file to change the parameters of the dataset creation. Including silence detection.
## To Do

- [x] Speech Diarization
- [x] Replace click with typer


## Acknowledgements


 - [AnalyseDataset.ipynb adapted from coqui-ai example](https://github.com/coqui-ai)
 - [OpenAI Whisper](https://github.com/openai/whisper)
 - [PyAnnote](https://github.com/pyannote/pyannote-audio)
 - [SpeechBrain](https://github.com/speechbrain/speechbrain)
