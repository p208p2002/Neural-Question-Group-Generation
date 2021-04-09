import pytorch_lightning as pl
from models.feedback import argparser
from models.feedback.model import Model
from models.feedback.data_module import DataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from models.feedback.config import GPUS,ACCELERATOR
from copy import deepcopy

args = argparser.get_args()

if __name__ == "__main__":
    
    trainer = pl.Trainer(
        gpus=GPUS,
        accelerator=ACCELERATOR,
        fast_dev_run=args.dev,
        precision=32,
        default_root_dir='.log_feedback',
        max_epochs=args.epoch,
        callbacks=[
            ModelCheckpoint(monitor='dev_loss',filename='{epoch}-{total_dev_loss:.2f}',save_last=True),
        ]
    )

    # DataModule
    dm = DataModule()

    # from_checkpoint
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

    # run_test
    trainer.test(model,datamodule=dm)