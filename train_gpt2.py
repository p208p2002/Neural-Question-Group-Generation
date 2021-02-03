import pytorch_lightning as pl
from modeling.gpt2 import argparser
from modeling.gpt2.model import Model
from modeling.gpt2.data_module import DataModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
args = argparser.get_args()

if __name__ == "__main__":
    
    trainer = pl.Trainer(\
        gpus=-1,\
        accelerator='dp',\
        fast_dev_run=1 if args.dev else False,\
        precision=32,\
        log_every_n_steps=1 if args.dev else 50,
        default_root_dir='.log_gpt2',
        max_epochs=args.epoch,
        checkpoint_callback= not args.run_test,
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
    trainer.test(datamodule=dm,ckpt_path=None if args.dev else 'best')