
# whisperer

Go from raw audio files to a speaker separated text-audio datases automatically.

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



## How to use:

1. Clone the repo
``` 
git clone https://github.com/miguelvalente/whisperer.git
```
2. Install the dependencies
      - Install [Poetry](https://python-poetry.org/docs/)
```
cd whisperer
poetry install
poetry shell
```

3. Create data folder and move audio files to it
```
mkdir data
mkdir data/audio_files 
```
4. Commands can be called individually or sequentially
   1. Convert
      ```
      python -m main convert
      ```
   2. Diarize *(requires converted to be called first)*
      ```
      python -m diarize
      ```
   3. Auto-Label *(requires diarize to be called first)*
      ```
      python -m auto-label number_of_speakers_present_in_your_audio_file_
      ```    
   4. Transcribe *(requires converted to be called first)*
      ```
      python -m transcribe your_dataset_name
      ```
   5. Convert & Diarize & Auto-Label & Transcribe
      ```
      python main.py convert diarize auto-label 6 transcribe your_dataset_name
      ```

5. Use the ```AnalyseDataset.ipynb``` notebook to visualize the distribution of the dataset
6. Use the ```AnalyseSilence.ipynb``` notebook to experiment with silence detection configuration

### Using Multiple-GPUS

The code automatically detects how many GPU's are available and distributes the audio files in ```data/audio_files_wav``` evenly across the GPUs.
The automatic detection is done through ```nvidia-smi```. 

You can to make the available GPU's explicit by setting the environment variable ```CUDA_AVAILABLE_DEVICES```.  

### Configuration

Modify `config.py` file to change the parameters of the dataset creation.

## To Do

- [x] Speech Diarization


## Acknowledgements

 - [AnalyseDataset.ipynb adapted from coqui-ai example](https://github.com/coqui-ai)
 - [OpenAI Whisper](https://github.com/openai/whisper)
 - [PyAnnote](https://github.com/pyannote/pyannote-audio)
 - [SpeechBrain[(https://github.com/speechbrain/speechbrain)
