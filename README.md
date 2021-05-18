# Educational Question Group Generation for Reading Assessment Preparation
## Minimum System Requirements
- OS: Ubuntu 16.04
- RAM: 32GB
- GPU: 1080ti*1
- Python: 3.6
- Pytorch: 1.7

## Envermient Setup
```bash
sudo apt install -y unzip
pip install -U requirements.txt
python init_dataset.py
python setup_scorer.py
```
> This process may take 20 minutes

## Training
```
python train_xxx.py -m'record message'
```