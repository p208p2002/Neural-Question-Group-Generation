# Educational Question Group Generation for Reading Assessment Preparation
## Minimum System Requirements
- OS: Ubuntu 16.04
- RAM: 32GB
- GPU: 1080ti*1
- Python: 3.6
- Pytorch: 1.7

## Envermient Setup
The following scripts will automatic setup the envermient
```bash
sudo apt install -y unzip
pip install -Ur requirements.txt
python init_dataset.py # download dataset
python setup_scorer.py
```

## Training
```
python train_xxx.py -m'record message'
```
> Use `-h` to check other available `args`

> To reproduce the experiments, you do not need to specify other parameters
> We have set it to the default value, check `utils/argparser.py` for details
