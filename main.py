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
from trainer import EarlyStopping, Trainer
from utils import get_loss_func  # Define your custom losses here

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    checkpoint_dir = os.path.dirname(filename)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(state, filename)

def print_model_parameter_count(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

def checkpoint_state(epoch, model, optimizer, scheduler, min_val_loss, early_stopping=None):
    state = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'min_val_loss': min_val_loss,
    }
    if early_stopping is not None:
        state['early_stopping_counter'] = early_stopping.counter
        state['early_stopping_best_loss'] = early_stopping.best_loss
    return state

def main(args):
        # 1. Load YAML Hyperparameters
    config_path = os.path.join("configs",f"{args.model_config}.yaml")
    with open(config_path, 'r') as f:
        model_param = yaml.safe_load(f)
    # 2. Setup Tensorboard & Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. Initialize Model, Loss, and Optimizer
    model = get_model(args.model_name, **model_param).to(device)
    print_model_parameter_count(model)
    
    train_loader,val_loader = get_dataloader(
        args.dataset_name,
        batch_size=args.batch_size,
        normalize_labels=args.normalize_labels,
    )
    criterion = get_loss_func(args.loss_func_name) 
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs,
                                                    steps_per_epoch=len(train_loader))
    
    
    run_name = f"{args.model_config}_{args.dataset_name}_{args.loss_func_name}"
    if args.normalize_labels:
        run_name = f"{run_name}_normalized_labels"

    log_dir = f"logs/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    trainer = Trainer(model,optimizer,scheduler,criterion,writer,device)

    start_epoch = 0

    min_val_loss = float('inf')
    early_stopping = EarlyStopping(args.patience) if args.early_stopping else None
    # 4. Resume from Checkpoint
    if args.resume_path and os.path.isfile(args.resume_path):
        print(f"--- Loading Checkpoint: {args.resume_path} ---")
        checkpoint = torch.load(args.resume_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        min_val_loss = checkpoint.get('min_val_loss',float('inf'))
        if early_stopping is not None:
            early_stopping.best_loss = checkpoint.get('early_stopping_best_loss', min_val_loss)
            early_stopping.counter = checkpoint.get('early_stopping_counter', 0)
        print(f"Resuming from epoch {start_epoch}")

    # 5. Training Loop
    pbar=tqdm(range(start_epoch,start_epoch+args.epochs), dynamic_ncols=True, smoothing=0.05)
    for epoch in pbar:
        # --- Training Logic ---
        trainer.train_one_epoch(train_loader,epoch) 

        should_validate = args.early_stopping or (epoch+1)%20 == 0
        if should_validate:
            val_loss = trainer.validate(val_loader,epoch) 
            improved = early_stopping.step(val_loss) if early_stopping is not None else val_loss < min_val_loss
            if improved:
                min_val_loss = val_loss
                checkpoint_name = f"./checkpoints/{run_name}_best.pth.tar"
                save_checkpoint(
                    checkpoint_state(epoch, model, optimizer, scheduler, min_val_loss, early_stopping),
                    filename=checkpoint_name
                )

        checkpoint_name = f"./checkpoints/{run_name}_last.pth.tar"
        save_checkpoint(
            checkpoint_state(epoch, model, optimizer, scheduler, min_val_loss, early_stopping),
            filename=checkpoint_name
        )

        if early_stopping is not None and early_stopping.should_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            break
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
    parser.add_argument("--early_stopping", action="store_true", help="Enable early stopping based on validation loss")
    parser.add_argument("--patience", type=int, default=10, help="Epochs without validation improvement before early stopping")
    parser.add_argument("--normalize_labels", action="store_true", help="Min-max normalize labels using train-set statistics")

    args = parser.parse_args()


    main(args)
