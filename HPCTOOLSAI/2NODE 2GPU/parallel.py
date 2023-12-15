import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from pathlib import Path
from pytorch_lightning.callbacks import EarlyStopping, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from time import time
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split




# Training hyperparameters
INPUT_SIZE = 784
NUM_CLASSES = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 20
NUM_Neurons = 128

# Dataset
DATA_DIR = "dataset/"
NUM_WORKERS = 4

# Compute related
ACCELERATOR = "gpu"
DEVICES = -1 # -1 for all available GPUs
PRECISION = 16

Path("tb_logs/profiler0").mkdir(parents=True, exist_ok=True)
Path("tb_logs/mnist_model_v1").mkdir(parents=True, exist_ok=True)

torch.set_float32_matmul_precision("medium") # to make lightning happy

class MnistDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage):
        entire_dataset = datasets.MNIST(
            root=self.data_dir,
            train=True,
            transform=transforms.Compose([
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]),
            download=False,
        )
        self.train_ds, self.val_ds = random_split(entire_dataset, [50000, 10000])
        self.test_ds = datasets.MNIST(
            root=self.data_dir,
            train=False,
            transform=transforms.ToTensor(),
            download=False,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

class MyPrintingCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer, pl_module):
        print("Starting to train!")

    def on_train_end(self, trainer, pl_module):
        print("Training is done.")
        

class NN(pl.LightningModule):
    def __init__(self, input_size, learning_rate, num_classes):
        super().__init__()
        self.lr = learning_rate
        self.fc1 = nn.Linear(input_size, NUM_Neurons)
        # Add 10 hidden layers
        for i in range(10):
            setattr(self, f"fc{i + 2}", nn.Linear(NUM_Neurons, NUM_Neurons))
        self.fc12 = nn.Linear(NUM_Neurons, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
        self.training_step_outputs = None

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # Pass through 10 hidden layers
        for i in range(10):
            x = F.relu(getattr(self, f"fc{i + 2}")(x))

        x = self.fc12(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, scores, y = self._common_step(batch, batch_idx)

        self.log_dict(
            {
                "train_loss": loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        acc = self.accuracy(scores, y)
        f1 = self.f1_score(scores, y)
        self.training_step_outputs = {"acc": acc, "f1": f1}
        
        if batch_idx % 100 == 0:
            x = x[:8]
            grid = torchvision.utils.make_grid(x.view(-1, 1, 28, 28))
            self.logger.experiment.add_image("mnist_images", grid, self.global_step)

        return {"loss": loss, "scores": scores, "y": y}
    
    def on_train_epoch_end(self):
        self.log_dict(
            {
                "train_acc": self.training_step_outputs["acc"],
                "train_f1": self.training_step_outputs["f1"],
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )


    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log("val_loss", loss,sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log("test_loss", loss,sync_dist=True)
        return loss

    def _common_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
        
if __name__ == "__main__":
    start = time()
    logger = TensorBoardLogger("tb_logs", name="mnist_model_v1")
    profiler = PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler("tb_logs/profiler0"),
        schedule=torch.profiler.schedule(skip_first=10, wait=1, warmup=1, active=20),
    )
    model = NN(input_size=INPUT_SIZE,learning_rate=LEARNING_RATE,num_classes=NUM_CLASSES)
    dm = MnistDataModule(data_dir=DATA_DIR,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS)
    trainer = pl.Trainer(
        strategy="ddp",
        num_nodes=2,
        profiler=profiler,
        logger=logger,
        accelerator=ACCELERATOR,
        devices=DEVICES,
        min_epochs=1,
        max_epochs=NUM_EPOCHS,
        precision=PRECISION,
        callbacks=[MyPrintingCallback(), EarlyStopping(monitor="val_loss")]
    )
    trainer.fit(model, dm)
    metrics_dict_val = trainer.validate(model, dm)
    metrics_dict_test = trainer.test(model, dm)
    end = time()
    print(f"total time {end-start:.2f}")
    print("Validation Stats",metrics_dict_val)
    print('test metrics',metrics_dict_test)

