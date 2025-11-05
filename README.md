# 🧠 Mixture-of-Experts (MoE) using PyTorch RPC (experimental)

This is my small experiment with **PyTorch’s RPC framework** — I tried to make a **Mixture-of-Experts model** where each expert runs on a separate process and still gets trained properly using **distributed autograd** and **DistributedOptimizer**.

The experiment is tested on the **FashionMNIST** dataset for simplicity.

⚠️ RPC backend (TensorPipe) doesn’t fully work on **Windows**, so this won’t run there.  
It should work fine on **Linux / WSL2 / Colab** though.

Reference: https://github.com/pytorch/pytorch/issues/50805? — TensorPipe backend not supported on Windows.

---

## 🌱 What this project is about
I wanted to see how we can train a model where different parts (experts) live in different processes.  
With normal `torch.multiprocessing`, this is tough because gradients don’t travel across processes — only the master part learns.

So I tried RPC, which lets us call remote models like normal Python functions, *and still get gradients back* through PyTorch’s distributed autograd system.  
This is more like how real distributed model-parallel systems (like large MoE models) actually work.

---

## 🧩 Main idea

### 1. What happens in this code
- The **master process** has the router network.  
- It also creates a few **remote experts** on different worker processes using `rpc.remote()`.  
- When we pass input, the router decides which experts to use (top-k).  
- The master then calls them using `rpc.rpc_sync()` and mixes their outputs.  
- During training, everything runs inside a `dist_autograd.context()` which records the computation across processes.  
- When we call `dist_autograd.backward()`, gradients automatically flow to remote experts.  
- Finally, `DistributedOptimizer` applies those gradients directly on each worker where the expert lives.

✅ This means both router and experts get trained together, even though they’re on different processes!

---

## ⚙️ Why RPC and not torch.multiprocessing
| Feature | `torch.multiprocessing` | `torch.distributed.rpc` |
|----------|--------------------------|--------------------------|
| Data transfer | Sends pickled tensors | Uses TensorPipe backend (efficient) |
| Gradients | ❌ Lost when crossing processes | ✅ Works with distributed autograd |
| Optimizer | Local only | `DistributedOptimizer` updates remote params |
| Training remote parts | Not possible | Supported |
| Ideal for | Simple parallel CPU work | Model-parallel training setups |

Basically, in multiprocessing, when you send tensors or models, they’re **pickled** and lose gradient connections.  
So only the master’s model updates, experts stay frozen.  

RPC fixes this by keeping a global autograd graph that connects all workers.  

---

## ⚗️ How Distributed Autograd Works (in simple words)
Normally, `autograd` works only inside one process.  
When you send data to another process, it forgets where it came from.  

`torch.distributed.autograd` fixes that by:
- creating a distributed graph across processes,
- remembering which worker did what,
- and sending gradients back automatically during `.backward()`.

So when you do:
```python
with dist_autograd.context() as ctx:
    output = model(data)
    loss = criterion(output, target)
    dist_autograd.backward(ctx, [loss])
