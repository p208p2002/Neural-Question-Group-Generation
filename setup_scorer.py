import os
if __name__ == "__main__":
    os.system('sudo apt-get update && sudo apt-get install default-jre && sudo apt-get install default-jdk')
    os.system('pip install git+git+https://github.com/voidful/nlg-eval.git@master')
    os.system('export LC_ALL=C.UTF-8&&export LANG=C.UTF-8&&nlg-eval --setup')