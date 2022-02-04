from Transformer.transfomer import TransformerTranslator
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

seq_size = 100
token_size = 10
num_blocks = 2
num_heads = 2  # Must be factor of token size
vocab_size = 5
max_context_length = 1000

x_nums = np.random.rand(seq_size) * (vocab_size - 2)  # Not Used
y_nums = np.random.rand(seq_size) * (vocab_size - 2)
z_nums = np.random.rand(seq_size) * (vocab_size - 2)  # Not Used

x = np.zeros((seq_size))
y = np.zeros((seq_size))
z = np.zeros((seq_size))
for idx, item in enumerate(x_nums):
    x[idx] = abs(int(item)) + 1
for idx, item in enumerate(y_nums):
    y[idx] = abs(int(item)) + 1
for idx, item in enumerate(z_nums):
    z[idx] = abs(int(item)) + 1

x = torch.from_numpy(x).type(torch.LongTensor)
y = torch.from_numpy(y).type(torch.LongTensor)
z = torch.from_numpy(z).type(torch.LongTensor)


import matplotlib.pyplot as plt


def plot_data(x, y, epoch):
    plt.cla()
    plt.clf()
    fig, ax = plt.subplots()
    ax.set_title("Epoch:" + str(epoch))
    fig.dpi = 100.0
    plt.xlim([0, seq_size])
    plt.ylim([0, vocab_size])
    plt.plot(x)
    plt.plot(y)
    plt.savefig(str(epoch) + ".png")


epoch_num = 100
test = TransformerTranslator(token_size, num_blocks, num_heads, vocab_size,vocab_size)
optimizer = torch.optim.Adam(test.parameters(), lr=1e-2)
criterion = nn.MSELoss()
import random

for i in range(epoch_num):
    outs = []
    avg_loss = []
    ix = y.unsqueeze(0)
    test.zero_grad()
    optimizer.zero_grad()
    torch_outs = None
    iy_label_outs = None
    iy_label_outs_args = torch.tensor([], requires_grad=False, dtype=torch.int64)

    _ = test.encode(ix)  # Encode
    for interval_jdx in range(1, 101):
        if interval_jdx != 1:
            iy = iy_label_outs_args  # y[:interval_jdx]
        else:
            iy = torch.tensor([0], dtype=torch.int64)
        # Predict next token
        iy = iy.unsqueeze(0)
        out = test.forward(iy)

        iy_label_raw = np.zeros((vocab_size), dtype=np.float32)
        iy_label_raw[y[interval_jdx - 1]] = 1
        iy_label = torch.from_numpy(iy_label_raw).unsqueeze(0)

        if interval_jdx != 1:
            torch_outs = torch.cat((torch_outs, out), dim=1)
            iy_label_outs = torch.cat((iy_label_outs, iy_label), dim=0)
        else:
            torch_outs = out
            iy_label_outs = iy_label
        iy_label_outs_args = torch.cat(
            (iy_label_outs_args, torch.tensor([torch.argmax(out)], dtype=torch.int64)),
            dim=0,
        )
        outs.append(torch.argmax(out.detach()))
    # plot_data(y,outs,i)
    loss = criterion(iy_label_outs, torch_outs[0])
    loss.backward()
    optimizer.step()
    print("MEAN LOSS: {}".format(i), loss.item())


import random

for i in range(1):
    outs = []
    avg_loss = []
    ix = y.unsqueeze(0)
    torch_outs = None
    iy_label_outs = None
    iy_label_outs_args = torch.tensor([], requires_grad=False, dtype=torch.int64)
    _ = test.encode(ix)
    for interval_jdx in range(1, 101):
        if interval_jdx != 1:
            iy = iy_label_outs_args  # y[:interval_jdx]
        else:
            iy = torch.tensor([0], dtype=torch.int64)
        # Predict next token
        iy = iy.unsqueeze(0)
        out = test.forward(iy)

        iy_label_raw = np.zeros((vocab_size))
        iy_label_raw[y[interval_jdx - 1]] = 1
        iy_label = torch.tensor((iy_label_raw), requires_grad=True).unsqueeze(0)

        if interval_jdx != 1:
            torch_outs = torch.cat((torch_outs, out), dim=1)
            iy_label_outs = torch.cat((iy_label_outs, iy_label), dim=0)
        else:
            torch_outs = out
            iy_label_outs = iy_label

        iy_label_outs_args = torch.cat(
            (iy_label_outs_args, torch.tensor([torch.argmax(out)], dtype=torch.int64)),
            dim=0,
        )
        outs.append(torch.argmax(out.detach()))

    loss = criterion(iy_label_outs, torch_outs)
    print("MEAN LOSS TEST: {}".format(i), loss)
plot_data(y, outs, "Test")
