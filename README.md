
# whisperer

Go from raw audio files to a text-audio dataset automatically with OpenAI's Whisper.

## Summary

This repo takes a directory of audio files and converts them to a text-audio dataset with normalized distribution of audio lengths. *See ```AnalyzeDataset.ipynb``` for examples of the dataset distributions across audio and text length*

The output is a text-audio dataset that can be used for training a speech-to-text model or text-to-speech. Currently the code only supports single speaker audio files.

The dataset structure is as follows:

```
/dataset
      |
      | -> metadata.txt
      | -> /wavs
              | -> audio1.wav
              | -> audio2.wav
              | ...
```

metadata.txt 
```
peters_0.wav|Beautiful is better than ugly.
peters_1.wav|Explicit is better than implicit.

```



## Key Features

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
```
conda create -n whisperer python=3.11
conda activate whisperer
pip install -r requirements.txt
```
3. Create data folder and move audio files to it
```
mkdir data
mkdir data/audio_files 
```
4. Run the main file
```
python main.py
```

5. Use the ```AnalyseDataset.ipynb``` notebook to visualize the distribution of the dataset

### Using Multiple-GPUS

The code automatically detects how many GPU's are available and distributes the audio files in ```data/audio_files_wav``` evenly across the GPUs.
The automatic detection is done through ```nvidia-smi```. 

You can to make the available GPU's explicit by setting the environment variable ```CUDA_AVAILABLE_DEVICES```.  

### Configuration

Modify `config.py` file to change the parameters of the dataset creation.

## To Do

- [ ] Speech Diarization


## Acknowledgements

 - [AnalyseDataset.ipynb adapted from coqui-ai example](https://github.com/coqui-ai)
 - [OpenAI Whisper](https://github.com/openai/whisper)
