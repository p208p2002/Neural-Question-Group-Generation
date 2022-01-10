# EQGG: Educational Question Group Generation
<span>
<a target="_blank" href="https://github.com/p208p2002/Neural-Question-Group-Generation">
<img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white">
</a>

<a target="_blank" href="https://huggingface.co/p208p2002/qmst-qgg">
<img src="https://img.shields.io/badge/🤗 HF Model Hub-ffea00?style=for-the-badge&logoColor=white">
</a>
</span>

<a target="_blank" href="https://huggingface.co/p208p2002/qmst-qgg">
<img src="https://img.shields.io/badge/💻 Live Demo-78ab78?style=for-the-badge&logoColor=white">
</a>
</span>

## Components
There are some practical things or components that are independent and spread across multiple repos

- [QGG-RACE dataset](https://github.com/p208p2002/QGG-RACE-dataset)
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
