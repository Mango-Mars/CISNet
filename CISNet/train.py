import os
from argparse import ArgumentParser
from model.cisnet import CISNet
from dataset.glacier_data import GlacierDataModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
import sys
from torchsummary import summary
from pytorch_lightning.callbacks import ModelCheckpoint
import time
import torch
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def main(hparams, run_number, running_mode):
    checkpoint_dir = os.path.join('checkpoints', hparams.model_name, 'run_' + str(run_number))
    tb_logs_dir = os.path.join('tb_logs', hparams.model_name, 'run_' + str(run_number))
    checkpoint_callback = ModelCheckpoint(monitor='avg_metric_validation',
                                          dirpath=checkpoint_dir,
                                          filename='-{epoch:02d}-{avg_metric_validation:.2f}',
                                          mode='max',  # Here: the higher the IoU the better
                                          save_top_k=1)
    early_stop_callback = EarlyStopping(monitor="avg_metric_validation", patience=30,
                                        verbose=False, mode="max", check_finite=True)
    clip_norm = 1.0


    logger = TensorBoardLogger(tb_logs_dir, name="log", default_hp_metric=False)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # if we have already trained this model take up the training at the last checkpoint
    if os.path.isfile(os.path.join(checkpoint_dir, "temporary.ckpt")):
        print("Taking up the training where it was left (temporary checkpoint)")
        trainer = Trainer(resume_from_checkpoint=os.path.join(checkpoint_dir, "temporary.ckpt"),
                          callbacks=[early_stop_callback, checkpoint_callback, lr_monitor],
                          deterministic=True,
                          gpus=1,  # Train on gpu
                          gradient_clip_val=clip_norm,
                          logger=logger,
                          max_epochs=hparams.epochs)
    else:
        if running_mode == "batch_overfit":
            # Try to overfit on some batches
            trainer = Trainer.from_argparse_args(hparams,
                                                 callbacks=[checkpoint_callback, lr_monitor],
                                                 deterministic=True,
                                                 fast_dev_run=False,
                                                 flush_logs_every_n_steps=100,
                                                 gpus=1,
                                                 log_every_n_steps=1,
                                                 logger=logger,
                                                 max_epochs=1000,
                                                 overfit_batches=1)

        elif running_mode == "debugging":
            # Debugging mode
            trainer = Trainer.from_argparse_args(hparams,
                                                 callbacks=[checkpoint_callback, lr_monitor],
                                                 deterministic=True,
                                                 fast_dev_run=3,
                                                 flush_logs_every_n_steps=100,
                                                 gpus=1,
                                                 log_every_n_steps=1,
                                                 logger=logger)

        elif running_mode == "training":
            # Training mode
            trainer = Trainer.from_argparse_args(hparams,
                                                 callbacks=[early_stop_callback, checkpoint_callback, lr_monitor],
                                                 deterministic=True,
                                                 gpus=1,  # Train on gpu
                                                 gradient_clip_val=clip_norm,
                                                 logger=logger,
                                                 max_epochs=hparams.epochs,
                                                 num_sanity_val_steps=0)
        else:
            print("Running mode not recognized")
            sys.exit()

    datamodule = GlacierDataModule(model_name=hparams.model_name,
                                   batch_size=hparams.batch_size,
                                   augmentation=running_mode != "batch_overfit",
                                   parent_dir=hparams.parent_dir,
                                   bright=hparams.bright,
                                   wrap=hparams.wrap,
                                   noise=hparams.noise,
                                   rotate=hparams.rotate,
                                   flip=hparams.flip)

    model = CISNet(vars(hparams))
    summary(model.cuda(), (2, 256, 256))
    print(model.eval())

    trainer.fit(model, datamodule=datamodule)

    # create a checkpoint if we are training (and delete the old one if it exists)
    if running_mode == "training":
        if os.path.isfile(os.path.join(checkpoint_dir, "temporary.ckpt")):
            os.remove(os.path.join(checkpoint_dir, "temporary.ckpt"))
        trainer.save_checkpoint(filepath=os.path.join(checkpoint_dir, "temporary.ckpt"))


if __name__ == '__main__':
    seed_everything(42)
    torch.multiprocessing.set_start_method('spawn')  # needed if ddp mode in Trainer

    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument('--parent_dir', default=".",
                               help="The directory in which the data directory lies. "
                                    "Default is '.' - this is where the data_preprocessing script has produced it.")
    parent_parser.add_argument('--training_mode', default="debugging",
                               help="Either 'training', 'debugging' or 'batch_overfit'.")
    parent_parser.add_argument('--epochs', type=int, default=150, help="The number of epochs the model shall be trained."
                                                                       "The weights after the last training epoch will "
                                                                       "be stored in temporary.ckpt."
                                                                       "Train.py will resume training from a "
                                                               
                                                                       "temporary.ckpt if one is available for this "
                                                                       "run.")
    parent_parser.add_argument('--run_number', type=int, default=0,
                               help="The number how often this model was already trained. "
                                    "If you run train.py twice with the same run_number, "
                                    "the second run will pick up training the first model from the temporary.ckpt.")

    parent_parser.add_argument('--model_name', type=str, default='CISNet',
                               help="Model used for segmentation. ")

    tmp = parent_parser.parse_args()
    parser = CISNet.add_model_specific_args(parent_parser)
    parser = Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    print(vars(hparams))

    assert hparams.training_mode == "training" or hparams.training_mode == "debugging" or hparams.training_mode == "batch_overfit", \
        "Please set --training_mode correctly. Either 'training' or 'debugging' or 'batch_overfit'."
    start_save = time.time()
    main(hparams, hparams.run_number, hparams.training_mode)
    end_save = time.time()
    print(f"Total time for training: {end_save - start_save}")