import matplotlib.pyplot as plt
import torch
import os

checkpoint = torch.load(os.path.join("Checkpoints", "Checkpoint24000.pkl"))
plt.plot(checkpoint["train_losses"])
plt.plot(checkpoint["test_losses"])
plt.show()
