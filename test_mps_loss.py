import torch
import torch.nn as nn
import time
import sys

print("=" * 60)
print("MPS CROSSENTROPY LOSS TEST")
print("=" * 60)

device = torch.device("mps")
torch.manual_seed(0)

B, C, H, W = 4, 3, 1578, 751
print(f"Tensor shape: [{B}, {C}, {H}, {W}]")

# Create tensors
print("Creating tensors...", flush=True)
x = torch.randn(B, C, H, W, device=device, requires_grad=True)
y = torch.randint(0, C, (B, H, W), device=device)
w = torch.tensor([0.2, 0.2, 0.6], device=device)

criterion = nn.CrossEntropyLoss(weight=w)

print("\n--- Testing Forward Pass ---", flush=True)
t0 = time.time()
loss = criterion(x, y)
torch.mps.synchronize()
forward_time = time.time() - t0
print(f"Forward done: loss={loss.item():.4f}, time={forward_time:.1f}s", flush=True)

print("\n--- Testing Backward Pass ---", flush=True)
t0 = time.time()
loss.backward()
torch.mps.synchronize()
backward_time = time.time() - t0
print(f"Backward done: time={backward_time:.1f}s", flush=True)

print("\n" + "=" * 60)
print("✅ TEST COMPLETE")
print(f"Forward: {forward_time:.1f}s")
print(f"Backward: {backward_time:.1f}s")
print("=" * 60)