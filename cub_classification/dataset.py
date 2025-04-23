import csv
from pathlib import Path
import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
from PIL import Image
from torchvision import transforms as T

class CUBDataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=None):
        super().__init__()
        self.csv_file = csv_file
        self.data_dir = data_dir
        self.transform = transform
        
        self.samples = []
        with open(self.csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append(row)
    # return correct n samples            
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx] # row is a dict with keys corresponding to entry in csv

        image_filename = row['filename']
        img_path = Path(self.data_dir) / "images" / image_filename

        # Load image
        with Image.open(img_path) as img:
            img = img.convert("RGB")
        
        # convert row number to zero-indexed class id
        class_id = int(row['class_id']) - 1

        # Load the bounding box

        x_min = float(row['x_min'])
        y_min = float(row['y_min'])
        x_max = float(row['x_max'])
        y_max = float(row['y_max'])

        bounding_box = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)

        return img, (class_id, bounding_box)


class CUBDataModule(pl.LightningDataModule):

    def __init__(self, data_dir, batch_size, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        
        if transform is None:
            transform = T.ToTensor()
        self.transform = transform

    # setup
    def setup(self, stage=None):
        # Load the train and test datasets
        self.train_dataset = CUBDataset(
            csv_file = self.data_dir / "train.csv", 
            data_dir = self.data_dir, 
            transform = self.transform
        )

        self.val_dataset = CUBDataset(
            csv_file = self.data_dir / "val.csv",
            data_dir= self.data_dir, 
            transform= self.transform
        )

    # training step
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size = self.batch_size,
            shuffle = True # prevent NN from memorizing order of images
        )


    # validation step to track performance
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size = self.batch_size,
            shuffle = False
        )

