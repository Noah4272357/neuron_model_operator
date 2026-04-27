import torch

class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion, writer, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.writer = writer
        self.device = device
        self.global_step = 0

    def train_one_epoch(self, dataloader, epoch):
        self.model.train()
        train_loss = 0.0
        for data, target, grid in dataloader:
            data, target, grid = data.to(self.device), target.to(self.device), grid.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data, grid)
            loss = self.criterion(output, target)
            loss.backward()
            
            train_loss += loss.item()
            # Log Grad Norm and LR
            #grad_norm = self._get_grad_norm()
            #self.writer.add_scalar('Batch/Loss', loss.item(), self.global_step)
            #self.writer.add_scalar('Batch/GradNorm', grad_norm, self.global_step)
            
            self.optimizer.step()
            self.global_step += 1
            
        # Log LR once per epoch
        self.writer.add_scalar('Loss/train', train_loss/len(dataloader), epoch)
        self.writer.add_scalar('Epoch/LR', self.optimizer.param_groups[0]['lr'], epoch)
        
        self.scheduler.step()

    def _get_grad_norm(self):
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5
    
    def validate(self,dataloader,epoch):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target, grid in dataloader:
                data, target, grid = data.to(self.device), target.to(self.device), grid.to(self.device)
                outputs = self.model(data, grid)
                loss = self.criterion(outputs, target)
                val_loss += loss.item()
                
        self.writer.add_scalar('Loss/valid', val_loss / len(dataloader), epoch)
        
        return val_loss/len(dataloader)