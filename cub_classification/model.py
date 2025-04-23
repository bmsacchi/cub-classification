import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD


class CUBModel(pl.LightningDataModule):
    def __init__(:
        self, 
        num_classes=200,
        train_classification = True,
        train_regression = True, 
        # something about weights i didn't understand
        classification_weight = 1.0,
        regression_weight = 1.0,
        lr = 1e-3
    ):

        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.train_classification = train_classification
        self.train_regression = train_regression
        self.classification_weight = classification_weight
        self.regression_weight = regression_weight
        self.lr = lr
        # convolutional layers
        # 2d outputs
        # feature map has same dimesnions as image, flatten into 1d
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding =1)
        self.pool = nn.MaxPool2d(2,2)

        self.gap = nn.AdaptiveAvgPool2d((56, 56))

        self.classifier = nn.Sequential(
            nn.Linear(8 * 56 * 56, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_classes)
        )

        self.regressor = nn.Sequential(
            nn.Linear(8 * 56 * 56, 512),
            nn.ReLU(),
            nn.Linear(512, 4)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x) # non parametric - nothing needs to be learned during training
        x = self.pool(x) # non parametric
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1) # convert to 1d

        # Classifier
        x_classif = self.classifier(x)

        # Regressor
        x_regr = self.regressor(x)

        return x_classif, x_regr



    def training_step( self, batch, batch_idx):
        images, (labels, bounding_boxe) = batch
        # make prediction
        labels_pred, bounding_boxes_pred = self(images)

        # calculate loss
        loss = 0.0
        # logging loss values in dict
        log_dict = {} 

        # logging classification loss, adding to total loss
        if self.train_classification:
            classification_loss = F.cross_entropy(labels_pred, labels) # what is cross entropy?
            loss += self.classification_weight * classification_loss 
            log_dict["classification_loss"] = classification_loss
        # logging regresssion loss
        if self.train_regression:
            regression_loss = F.mse_loss(bounding_boxes_pred, bounding_boxes)
            loss += self.regression_weight * regression_loss
            log_dict["regression_loss"] = regression_loss
        # log the values
        self.log_dict(log_dict, prog_bar=True)
        
        return loss


    def validation_step(self, batch, batch_idx):
        images, (labels, bounding_boxe) = batch
        # make prediction
        labels_pred, bounding_boxes_pred = self(images)

        # calculate loss
        loss = 0.0
        # logging loss values in dict
        log_dict = {} 

        # logging classification loss, adding to total loss
        if self.train_classification:
            classification_loss = F.cross_entropy(labels_pred, labels) # what is cross entropy?
            loss += self.classification_weight * classification_loss 
            log_dict["val_classification_loss"] = classification_loss
        # logging regresssion loss
        if self.train_regression:
            regression_loss = F.mse_loss(bounding_boxes_pred, bounding_boxes)
            loss += self.regression_weight * regression_loss
            log_dict["val_regression_loss"] = regression_loss
        # log the values
        self.log_dict(log_dict, prog_bar=True, on_epoch=True)
        
        return loss

    # stochastic gradient descent optimizer
    def configure_optimizers(self):
        return SGD(self.parameters(), lr = self.lr)