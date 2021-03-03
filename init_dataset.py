import gdown
import os

def download_from_gd(gid,save_name,save_dir = 'datasets'):
    os.makedirs(save_dir,exist_ok=True)
    gdown.download('https://drive.google.com/uc?id=%s'%(gid), os.path.join(save_dir,save_name), quiet=False)

if __name__ == "__main__":
    os.system('rm -rf datasets/')
    # EQG-RACE-PLUS
    download_from_gd('1wXGyEjzwDpvG1TCv6C8JfUwDJmOiBwOr','EQG-RACE-PLUS.zip')
    os.system('unzip datasets/EQG-RACE-PLUS.zip -d datasets')
    

    