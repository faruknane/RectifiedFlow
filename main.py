import numpy as np
import torch
import torch.backends
import torch.backends.cuda
import torch.backends.cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import cv2
import util
import json
import modules
import modules.attention
import modules.diffusion
import modules.diffusion.util
import modules.diffusion.openaimodel
import time
import gc
import data
import pytorch_lightning as pl
import shutil
import os
import yaml


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    
    # with open("configs/config.json", "r") as f:
    #     config = json.load(f)

    # with open("configs/config.yaml", "w") as f:
    #     yaml.dump(config, f, default_flow_style=False)

    config_path = "configs/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    config = util.instantiate_object(config)

    device = config["device"]
    model = config["model"]
    trainset = config["trainset"]
    valset = config["valset"]
    testset = config["testset"]
    train_dataloader = config["train_dataloader"]
    val_dataloader = config["val_dataloader"]
    test_dataloader = config["test_dataloader"]

    print("Trainset:", len(trainset))
    print("Valset:", len(valset))

    model.to(device)

    # define a csv logger
    csv_logger = pl.loggers.CSVLogger("logs", name="diffusion")

    # trainer
    trainer = pl.Trainer(max_epochs=1000, logger=csv_logger, accumulate_grad_batches=1)
    
    experiment_folder = trainer.logger.log_dir
    os.makedirs(experiment_folder, exist_ok=True)
    
    print("Experiment folder:", experiment_folder)
    shutil.copy(config_path, os.path.join(experiment_folder, "config.yaml"))
        

    if True:
        trainer.fit(model, train_dataloader, val_dataloader)
    else:
        trainer.test(model, dataloaders=test_dataloader, verbose=True)























    # x = torch.randn(1, 12, 64, 64).to(device)
    # timesteps = torch.full((1, ), 2, dtype=torch.long).to(device)
    # y = model(x, timesteps=timesteps)

    # print(x.shape)
    # print(y.shape)

    # print memory usage
    # print(torch.cuda.memory_summary(device=device, abbreviated=True))


    # timesteps = timesteps = torch.arange(0, 1000, 1).to(torch.long)
    
    # res = timestep_embedding(timesteps, 128, max_period=1000)

    # # convert res to numpy 
    # res = res.cpu().detach().numpy()

    # # normalize res
    # res = (res - res.min()) / (res.max() - res.min()) 
    # res = (res * 255).astype(np.uint8)

    # # transpose res
    # res = np.transpose(res, (1, 0))

    # # save res as image
    # cv2.imwrite("timestep_embedding.png", res)




