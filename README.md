# An Investigation into Neural Question Group Generation augmented by Genetic Algorithm and Negative Label Learning

## Components
There are some practical things or components that are independent, welcome to use

- [Dataset](https://github.com/p208p2002/EQG-RACE-PLUS/tree/qgg-dataset)
- [qgg-utils](https://github.com/p208p2002/qgg-utils)
> Negative Label Loss, Genetic algorithm and Evaluation scorer

## Environment
### Require
- OS:ubuntu 16.04+
- RAM:24GB
- GPU: CUDA device with 12GB VRAM
### Setup
Please install `pytorch>=1.7.1` manually first
> PyTorch install: https://pytorch.org/get-started/locally/

```
sudo apt install unzip
pip install -Ur requirements.txt
python -c "import stanza;stanza.download('en')"
python init_dataset.py
python setup_scorer.py
```
## Training
```
python train_xxx.py -m'message to log for this training'
```
