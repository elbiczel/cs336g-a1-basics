import torch

import cs336_nn as nn

steps = 10
for lr in [1, 10, 100, 1000]:
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = nn.optimizer.SGD([weights], lr=lr)
    for t in range(steps):
        opt.zero_grad()
        loss = (weights**2).mean()
        #print(loss.cpu().item())
        loss.backward()
        opt.step()
    print(f"Final loss for lr {lr}: {loss.cpu().item()}")

# Final loss for lr 1: 22.876144409179688
# Final loss for lr 10: 2.973475456237793
# best --> Final loss for lr 100: 2.4665386498748656e-23
# clearly diverges --> Final loss for lr 1000: 2.1448573270381036e+18
