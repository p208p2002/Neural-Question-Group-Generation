import pytorch_lightning as pl
from modeling.base import argparser
from modeling.base.model import Model
from modeling.base.data_module import DataModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from modeling.base.config import GPUS,ACCELERATOR
args = argparser.get_args()

if __name__ == "__main__":
    
    trainer = pl.Trainer(
        gpus=GPUS,
        accelerator=ACCELERATOR,
        fast_dev_run=args.dev,
        precision=32,
        default_root_dir='.log_base',
        max_epochs=args.epoch,
        callbacks=[
            EarlyStopping(monitor='dev_loss',patience=2),
            ModelCheckpoint(monitor='dev_loss',filename='{epoch}-{dev_loss:.2f}',save_last=True),
        ]
    )

    dm = DataModule()

    if args.from_checkpoint is None:
        model = Model()
    else:
        print('load from checkpoint')
        model = Model.load_from_checkpoint(args.from_checkpoint)

    if args.run_test == False:
        trainer.fit(model,datamodule=dm)
    trainer.test(model if args.run_test else None,datamodule=dm,ckpt_path=None if args.dev else 'best')