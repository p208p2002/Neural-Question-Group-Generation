import os

CMD ="""
python train_base.py -ds m_race --batch_size 8 &&
python train_beam_search.py -ds m_race --batch_size 8 &&
python train_unlimited.py -ds m_race --batch_size 8 &&
python train_feedback.py -ds m_race --batch_size 4 --epoch 8 &&
python train_feedback.py -ds m_race --batch_size 4 --disable_negative_loss --epoch 8
"""

os.system(CMD)