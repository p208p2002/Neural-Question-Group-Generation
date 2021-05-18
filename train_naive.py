import pytorch_lightning as pl
from models.Naive import argparser
from models.Naive.model import Model
from models.Naive.data_module import DataModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from models.Naive.config import GPUS,ACCELERATOR
from copy import deepcopy
args = argparser.get_args()

if __name__ == "__main__":
    
    trainer = pl.Trainer(
        gpus=GPUS,
        accelerator=ACCELERATOR,
        fast_dev_run=args.dev,
        precision=32,
        default_root_dir='.log_Naive',
        max_epochs=args.epoch,
        callbacks=[
            # EarlyStopping(monitor='dev_loss',patience=3),
            ModelCheckpoint(monitor='dev_loss',filename='{epoch}-{dev_loss:.2f}',save_last=True),
        ]
    )

    dm = DataModule()

    if args.from_checkpoint is None:
        model = Model()
    else:
        print('load from checkpoint')
        model = Model.load_from_checkpoint(args.from_checkpoint)

    # train
    if args.run_test == False:
        tuner = pl.tuner.tuning.Tuner(deepcopy(trainer))
        new_batch_size = tuner.scale_batch_size(model, datamodule=dm)
        del tuner
        model.hparams.batch_size = new_batch_size
        trainer.fit(model,datamodule=dm)

    trainer.test(model if args.run_test else None,datamodule=dm,ckpt_path=None if args.dev else 'best')