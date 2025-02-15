import os
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from data import ImageNetDataModule


def main(args):
    # Setting
    if args.size == "small":
        from model import LTBC
    else:
        from large_model import LTBC
    pl.seed_everything(42)

    # Handle the data
    dm = ImageNetDataModule(args.data_folder, batch_size=args.batch_size)

    # Define model
    model = LTBC()

    # Exp logger
    logger = TensorBoardLogger("logs/tensorboard_logs")

    # Define training
    checkpointer = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=3,
        mode="min",
        save_last=True,
        filename="{epoch}-{val_loss:.2f}-{train_loss:.2f}",
    )
    trainer = pl.Trainer(
        devices=6,
        strategy=DDPStrategy(process_group_backend="nccl"),
        num_nodes=args.n_nodes,
        accelerator="gpu",
        max_epochs=args.epochs,
        callbacks=[checkpointer],
        logger=logger,
        limit_train_batches=0.2,
        limit_val_batches=0.2,
        limit_test_batches=0.2,
        val_check_interval=1.0,
    )

    # Train
    trainer.fit(model, dm)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n_nodes", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--data_folder", type=str, default="../imagenet/train/")
    parser.add_argument("--size", type=str, default="large")
    args = parser.parse_args()
    main(args)
