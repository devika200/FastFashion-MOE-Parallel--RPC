import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef
from torch.distributed.optim import DistributedOptimizer
import torch.distributed.autograd as dist_autograd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# -----------------------------
# 1. Config
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
num_experts = 3
num_classes = 10
epochs = 5
lr = 1e-3

# -----------------------------
# 2. Dataset
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# -----------------------------
# 3. Expert Network
# -----------------------------
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


# ================================================================
# Remote Expert Wrapper
# ================================================================
class RemoteExpert:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.model = Expert(input_dim, hidden_dim, output_dim).to("cpu")

    def forward(self, x):
        return self.model(x)

    def parameter_rrefs(self):
        return [RRef(p) for p in self.model.parameters()]


# ================================================================
# Mixture of Experts (Main Router)
# ================================================================
class RPCMoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, expert_rrefs, top_k=2):
        super().__init__()
        self.router = nn.Linear(input_dim, len(expert_rrefs))
        self.expert_rrefs = expert_rrefs
        self.top_k = top_k

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        logits = self.router(x_flat)
        probs = F.softmax(logits, dim=1)
        top_vals, top_idx = torch.topk(probs, self.top_k, dim=1)

        results = []
        for i in range(x_flat.size(0)):
            combined = 0
            for j, expert_idx in enumerate(top_idx[i]):
                expert_rref = self.expert_rrefs[expert_idx.item()]
                y = rpc.rpc_sync(to=expert_rref.owner(), func=_call_expert, args=(expert_rref, x_flat[i].unsqueeze(0)))
                combined += top_vals[i, j] * y
            results.append(combined)
        return torch.cat(results, dim=0)

    def parameter_rrefs(self):
        return [RRef(p) for p in self.router.parameters()]


# ================================================================
# Helper: remote function to call expert
# ================================================================
def _call_expert(expert_rref, x):
    return expert_rref.local_value().forward(x)

# Helper: fetch parameter RRefs from a remote expert instance
def _get_param_rrefs(expert_rref):
    return expert_rref.local_value().parameter_rrefs()


# ================================================================
# Training
# ================================================================
def run_master(world_size, input_dim=784, hidden_dim=256, output_dim=10, num_experts=4, top_k=2):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.FashionMNIST("./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # Get expert references
    expert_rrefs = [rpc.remote(f"worker{i+1}", RemoteExpert, args=(input_dim, hidden_dim, output_dim))
                    for i in range(num_experts)]

    model = RPCMoE(input_dim, hidden_dim, output_dim, expert_rrefs, top_k)

    # Gather parameters (router + all experts)
    param_rrefs = model.parameter_rrefs()
    for e in expert_rrefs:
        remote_param_rrefs = rpc.rpc_sync(to=e.owner(), func=_get_param_rrefs, args=(e,))
        param_rrefs.extend(remote_param_rrefs)

    optimizer = DistributedOptimizer(torch.optim.Adam, param_rrefs, lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(2):
        for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1}", ncols=100):
            data = data.to("cpu")
            target = target.to("cpu")

            with dist_autograd.context() as context_id:
                output = model(data)
                loss = criterion(output, target)
                dist_autograd.backward(context_id, [loss])
                optimizer.step(context_id)
            print(f"Epoch {epoch+1}, loss = {loss.item():.4f}")


# ================================================================
# Entry point
# ================================================================
if __name__ == "__main__":
    # Read rank/world size from environment (when launched with torchrun),
    # fallback to a 5-process setup (1 master + 4 experts)
    world_size = int(os.environ.get("WORLD_SIZE", "5"))
    rank = int(os.environ.get("RANK", "0"))

    rpc.init_rpc(name=f"worker{rank}", rank=rank, world_size=world_size)

    if rank == 0:
        run_master(world_size)

    rpc.shutdown()
