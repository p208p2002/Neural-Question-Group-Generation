import os

CMD ="""
python train_base.py -ds m_race --batch_size 8 &&
python train_beam_search.py -ds m_race --batch_size 8 &&
python train_unlimited.py -ds m_race --batch_size 8 &&
python train_feedback.py -ds m_race --batch_size 4 --epoch 10 
"""

os.system(CMD)