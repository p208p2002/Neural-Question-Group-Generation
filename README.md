# EQGG: Automatic Question Group Generation
<span>
<a target="_blank" href="https://github.com/p208p2002/Neural-Question-Group-Generation">
<img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white">
</a>

<a target="_blank" href="https://huggingface.co/p208p2002/qmst-qgg">
<img src="https://img.shields.io/badge/🤗 HF Model Hub-ffea00?style=for-the-badge&logoColor=white">
</a>

<a target="_blank" href="https://huggingface.co/spaces/p208p2002/Question-Group-Generator">
<img src="https://img.shields.io/badge/💻 Demo (HF)-78ab78?style=for-the-badge&logoColor=white">
</a>

<a target="_blank" href="https://qgg-demo.nlpnchu.org/">
<img src="https://img.shields.io/badge/💻 Demo (NCHU)-5b62a4?style=for-the-badge&logoColor=white">
</a>
</span>

## Abstract
Question generation (QG) task plays a crucial role in adaptive learning. While significant QG performance advancements are reported, the existing QG studies are still far from practical usage. One point that needs strengthening is to consider the generation of question group , which remains untouched. For forming a question group, intra-factors among generated questions should be considered. This paper proposes a two-stage framework by combining neural language models and genetic algorithms for addressing the issue of question group generation.

## Components
There are some components spread across multiple repositories:

- [QGG-RACE dataset](https://github.com/p208p2002/QGG-RACE-dataset)
- [qgg-demo](https://github.com/p208p2002/qgg-demo)
- [qgg-utils](https://github.com/p208p2002/qgg-utils)
> Negative Label Loss, Genetic algorithm and Evaluation scorer

## Environment
### Require
- OS:ubuntu 16.04+
- RAM:24GB
- GPU: CUDA device with 12GB VRAM
### Setup
Please install `pytorch>=1.7.1,<=1.9.0` manually first
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
python train_xxx.py -m'message to note for this training'
```

## Citation
If your research references the relevant content, please cite:
```bibtex
@ARTICLE{10609322,
  author={Huang, Po-Chun and Chan, Ying-Hong and Yang, Ching-Yu and Chen, Hung-Yuan and Fan, Yao-Chung},
  journal={IEEE Transactions on Learning Technologies}, 
  title={EQGG: Automatic Question Group Generation}, 
  year={2024},
  volume={},
  number={},
  pages={1-14},
  keywords={Task analysis;Context modeling;Question generation;Training;Redundancy;Fans;Employment;Neural Question Generation;Natural Language Generation;Reading Comprehension Testing},
  doi={10.1109/TLT.2024.3430225}
}
```
