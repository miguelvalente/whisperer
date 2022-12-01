
# whisperer

Go from raw audio files to a text-audio dataset automatically with OpenAI's Whisper.

## Summary

This repo takes a directory of audio files and converts them to a text-audio dataset. The dataset is split on silences and the text is transcribed using OpenAI's Whisper tool. The output is a text-audio dataset that can be used for training a speech-to-text model. Currently the code only supports single speaker audio files.

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
## Configuration


### How to use:

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

5. Use the Analyse.ipynb notebook to visualize the distribution of the dataset




### Configuration

Modify `config.py` file to change the parameters of the dataset creation.

## To Do

- [ ] Speech Diarization

- [ ] Deal with audio segments longer than 30 seconds


## Acknowledgements

 - [AnalyseDataset.ipynb adapted from coqui-ai example](https://github.com/coqui-ai)
 - [OpenAI Whisper](https://github.com/openai/whisper)