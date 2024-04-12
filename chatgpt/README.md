## Requirements
```
pip install -r requirements.txt
pip install git+https://github.com/voidful/nlg-eval.git@master
export LC_ALL=C.UTF-8&&export LANG=C.UTF-8&&nlg-eval --setup
pip install stanza
python -c "import stanza;stanza.download('en')"
python init_dataset.py
python setup_scorer.py
```