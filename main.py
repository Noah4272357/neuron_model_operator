import argparse
import yaml
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import get_model
from data.dataset import get_dataloader
from trainer import Trainer
from utils import get_loss_func  # Define your custom losses here

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

def main(args):
        # 1. Load YAML Hyperparameters
    config_path = os.path.join("configs",f"{args.model_config}.yaml")
    with open(config_path, 'r') as f:
        model_param = yaml.safe_load(f)
    # 2. Setup Tensorboard & Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. Initialize Model, Loss, and Optimizer
    model = get_model(args.model_name, **model_param).to(device)
    
    train_loader,val_loader = get_dataloader(args.dataset_name,batch_size = args.batch_size)
    criterion = get_loss_func(args.loss_func_name) 
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs,
                                                    steps_per_epoch=len(train_loader))
    writer = SummaryWriter(log_dir=f"logs/{args.model_config}_{args.dataset_name}_{args.loss_func_name}")
    
    trainer = Trainer(model,optimizer,scheduler,criterion,writer,device)

    start_epoch = 0

    min_val_loss = float('inf')
    # 4. Resume from Checkpoint
    if args.resume_path and os.path.isfile(args.resume_path):
        print(f"--- Loading Checkpoint: {args.resume_path} ---")
        checkpoint = torch.load(args.resume_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        min_val_loss = checkpoint.get('min_val_loss',float('inf'))
        print(f"Resuming from epoch {start_epoch}")

    # 5. Training Loop
    pbar=tqdm(range(start_epoch,start_epoch+args.epochs), dynamic_ncols=True, smoothing=0.05)
    for epoch in pbar:
        # --- Training Logic ---
        trainer.train_one_epoch(train_loader,epoch) 
        checkpoint_name = f"./checkpoints/{args.model_config}_{args.loss_func_name}_{args.dataset_name}_last.pth.tar"
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, filename=checkpoint_name)


        if (epoch+1)%20 == 0:
            val_loss = trainer.validate(val_loader,epoch) 
            if val_loss<min_val_loss:
                min_val_loss = val_loss
                checkpoint_name = f"./checkpoints/{args.model_config}_{args.loss_func_name}_{args.dataset_name}_best.pth.tar"
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'min_val_loss': min_val_loss
                }, filename=checkpoint_name)
    writer.close()
    print("Experiment Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Deep Learning Experiment")

    # --- CLI Arguments ---
    parser.add_argument("--model_name", type=str, default='WNO')
    parser.add_argument("--dataset_name", type=str, default = 'multi_izhikevich')
    parser.add_argument("--model_config", type=str, default = 'WNO_config1',help="Path to YAML config")
    parser.add_argument("--loss_func_name", type=str, default="relative_l2")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--resume_path", type=str, default=None, help="Path to .pth.tar checkpoint")

    args = parser.parse_args()


    main(args)