import argparse
import pytorch_lightning as pl
from pathlib import Path
from cub_classification.model import CUBModel
from cub_classification.dataset import CUBDataModule

# check if this is the main execution file
# so you can import without running the whole script

# e.g. def something(): wihtout runnign the whole thnig...

if __name__ =="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, help="Path to the data directory.")
    parser.add_argument("--train_classification", type = bool, default = True, help = "Train classification")
    parser.add_argument("--train-regression", type = bool, default=True, help = "Train regression")
    parser.add_argument("--classification_weight", type = float, default = 1.0, help = "Weight for classification loss")
    parser.add_argument("--regression_weight", type = float, default = 1.0, help = "Weight for regression loss")

    args = parser.parse_args()

    data_module = CUBDataModule(
        data_dir = Path(args.data_dir),
        batch_size = 4
    )

    data_module.setup()

    model = CUBModel(
        num_classes = 200,
        train_classification = args.train_classification,
        train_regression = args.train_regression,
        classification_weight = args.classification_weight,
        regression_weight = args.regression_weight
    )

    trainer = pl.Trainer(max_epochs=5)

    trainer.fit(model, datamodule = data_module)

    # to run: pip install -e . 
    # so the directory is treated as a python module
    