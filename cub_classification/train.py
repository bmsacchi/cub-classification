import argparse
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
from cub_classification.model import CUBModel
from cub_classification.dataset import CUBDataModule
import optuna

import numpy as np
import random
# check if this is the main execution file
# so you can import without running the whole script
pl.seed_everything(42)

# e.g. def something(): wihtout runnign the whole thnig...
def objective(trial):
    lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
    classification_weight = trial.suggest_float("classification_weight", 0.1, 1.0)
    regression_weight = trial.suggest_float("regression_weight", 0.1, 1.0)
    
    wandb_logger = WandbLogger(
        project = "cub_classification_hyper",
        name = f'trial-{trial.number}',
        log_model = True
    )

    wandb_logger.experiment.config.update(
        {
            "lr": lr,
            "classification_weight": classification_weight,
            "regression_weight": regression_weight
        }
    )   

    data_module = CUBDataModule(
        data_dir = Path(args.data_dir),
        batch_size = 4
    )

    data_module.setup()

    model = CUBModel(
        num_classes = 200,
        train_classification = True,
        train_regression = True,
        classification_weight = classification_weight,
        regression_weight = regression_weight
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_combined_metric',
        patience=2,
        mode='min'
    )

    trainer = pl.Trainer(max_epochs=5,
                        logger=wandb_logger,
                        callbacks=[early_stopping_callback],
                        precision = '16-mixed')

    trainer.fit(model, datamodule = data_module)

    wandb_logger.experiment.finish()

    return trainer.callback_metrics["val_combined_metric"].item()


if __name__ =="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, help="Path to the data directory.")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of trials for hyperparameter optimization.")
    # parser.add_argument("--train_classification", type = bool, default = True, help = "Train classification")
    # parser.add_argument("--train-regression", type = bool, default=True, help = "Train regression")
    # parser.add_argument("--classification_weight", type = float, default = 1.0, help = "Weight for classification loss")
    # parser.add_argument("--regression_weight", type = float, default = 1.0, help = "Weight for regression loss")
    # parser.add_argument("--lr", type = float, default = 1e-3, help = "Learning rate")

    args = parser.parse_args()

    study = optuna.create_study(
        direction="maximize",
        study_name="cub_classification_hyper",
        storage="sqlite:///cub_classification.db",
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=args.n_trials)


    # wandb_logger = WandbLogger(
    #     project = "cub_classification",
    #     name = f'{args.classification_weight}_{args.regression_weight}_{args.lr}',
    #     log_model = True,
    #     save_dir = 'reports'
    # )

    # wandb_logger.experiment.config.update(
    #     {
    #         'classification_weight': args.classification_weight,
    #         'regression_weight': args.regression_weight,
    #         'learning_rate': args.lr,
    #     }   
    # )


    # data_module = CUBDataModule(
    #     data_dir = Path(args.data_dir),
    #     batch_size = 4
    # )

    # data_module.setup()

    # model = CUBModel(
    #     num_classes = 200,
    #     train_classification = args.train_classification,
    #     train_regression = args.train_regression,
    #     classification_weight = args.classification_weight,
    #     regression_weight = args.regression_weight
    # )

    # trainer = pl.Trainer(max_epochs=5, logger=wandb_logger)

    # trainer.fit(model, datamodule = data_module)

    # to run: pip install -e . 
    # so the directory is treated as a python module
    
