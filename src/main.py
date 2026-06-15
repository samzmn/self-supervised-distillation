import os
from datetime import datetime
import copy

import torch
from torch.utils.tensorboard import SummaryWriter

from data import load_cifar
from modules import DINOModel
from utils import DINOLoss
from train import train_dino


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 100
LR = 0.0005
MOMENTUM_TEACHER = 0.996  # EMA momentum for teacher
TEMPERATURE = 0.1
OUT_DIM = 1024  # Projection head output dimension
N_LOCAL_CROPS = 6
NUM_CLASSES = 10  # Change to 100 for CIFAR-100
DATASET = 'cifar10'  # or 'cifar100'

def main():
    save_path = "checkpoints"
    os.makedirs(save_path, exist_ok=True)
    writer = SummaryWriter(f'runs/dino_{DATASET}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')

    data_loader = load_cifar(BATCH_SIZE, shuffle = True, normal=True, 
               return_original=False, num_local_crops=N_LOCAL_CROPS)
    
    student = DINOModel(embed_dim=384, img_size=32, patch_size=4, num_heads=6, depth=12, dropout=0.,
                        head_hidden_dim=1024, head_out_dim=OUT_DIM).to(device)
    teacher = copy.deepcopy(student).to(device)

    dino_loss = DINOLoss(OUT_DIM, ncrops=2+N_LOCAL_CROPS, nepochs=EPOCHS)

    train_dino(student, teacher, dino_loss, data_loader, EPOCHS, save_path,
               base_lr=LR, teacher_momentum=MOMENTUM_TEACHER, writer=writer, device=device)
    
    torch.save(student, "dino_vit_cifar.pt")

if __name__ == "__main__":
    main()
