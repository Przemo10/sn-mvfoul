
# VARS model

##  Installation

conda create -n vars python=3.9

conda activate vars

Install Pytorch with CUDA : https://pytorch.org/get-started/locally/

pip install SoccerNet

pip install -r requirements.txt

python main.py --path "path/to/dataset" 


## Downloading dataset

Custom script `downolad_raw_dastaset.py`

It requires python variable:
```python
TOTAL_DATASETS = ["train", "valid", "test", ]
```
and `.env` file within this `vars-model` directory

```bash
DATASET_PASS=***
DATASET_VERSION=224p ## alternatively 720p for high resolution
```

After downloading the data we have `mvfolus` directory with specific zip files

Command for extracting zip files `dataset224p` (low resolution) or `dataset720p` (high resolution):

```bash
7z x mvfouls/test.zip -o./dataset224p/test/
7z x mvfouls/valid.zip -o./dataset224p/valid/
7z x mvfouls/train.zip 7z x mvfouls/test.zip -o./dataset224p/train/
7z x mvfouls/chalenge.zip 7z x -o./dataset224p/chalenge/
```
