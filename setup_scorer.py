import os

if __name__ == "__main__":
    os.system('pip install git+https://github.com/Maluuba/nlg-eval.git@master')
    os.system('export LC_ALL=C.UTF-8&&export LANG=C.UTF-8&&nlg-eval --setup')
    os.system('pip install git+https://github.com/Tiiiger/bert_score')