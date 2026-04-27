import torch
import torch.nn as nn
import torch.nn.functional as F

class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)
    
# calculate Lp loss with weight function    
class wLpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(wLpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]
        y_plus=y-y.min(dim=1,keepdim=True).values

        weight=F.normalize(y_plus,p=2,dim=1)

        diff_norms = torch.norm(weight*(x.reshape(num_examples,-1) - y.reshape(num_examples,-1)), 
                                self.p, 1)
        y_norms = torch.norm(weight*y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


class SoftDTW(nn.Module):
    def __init__(self, gamma=1.0, normalize=False):
        super(SoftDTW, self).__init__()
        self.gamma = gamma
        self.normalize = normalize

    def forward(self, x, y):
        # x: [batch_size, seq_len_x, feature_dim]
        # y: [batch_size, seq_len_y, feature_dim]
        batch_size, n, d = x.shape
        m = y.size(1)

        # Compute squared Euclidean distance matrix
        dist_mat = torch.cdist(x, y, p=2)**2

        # Initialize DP table
        # We use a large value (infinity) for boundaries
        D = torch.zeros((batch_size, n + 1, m + 1), device=x.device) + 1e8
        D[:, 0, 0] = 0

        # Soft-DTW Forward Pass (Recurrence)
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = dist_mat[:, i-1, j-1]
                
                # Soft-min of (up, left, diagonal)
                prev_costs = torch.stack([
                    D[:, i-1, j],    # Insertion
                    D[:, i, j-1],    # Deletion
                    D[:, i-1, j-1]   # Match
                ], dim=1)
                
                # Log-Sum-Exp trick for stability
                softmin = -self.gamma * torch.logsumexp(-prev_costs / self.gamma, dim=1)
                D[:, i, j] = cost + softmin

        result = D[:, n, m]
        
        if self.normalize:
            # Optionally subtract self-similarity to ensure loss >= 0
            # (Soft-DTW(x,y) - 0.5 * (Soft-DTW(x,x) + Soft-DTW(y,y)))
            pass 
            
        return result.mean()

# Example Usage
#batch_size, seq_len, dims = 2, 50, 1
#predicted = torch.randn(batch_size, seq_len, dims, requires_grad=True)
#target = torch.randn(batch_size, seq_len, dims)
#
#criterion = SoftDTW(gamma=0.1)
#loss = criterion(predicted, target)
#
#print(f"Loss: {loss.item():.4f}")
#print(f"Gradient shape: {predicted.grad.shape}")