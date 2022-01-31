from Transformer.transfomer import TransformerTranslator
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

batch_size = 1
seq_size = 10
token_size = 50
num_blocks = 3
num_heads = 3 #Must be factor of token size
vocab_size = 100
max_context_length = 5000

x_nums = torch.rand(100)*(vocab_size-2)
y_nums = torch.rand(100)*(vocab_size-2)
z_nums = torch.rand(100)*(vocab_size-2)
x = torch.zeros((vocab_size),requires_grad=True)
y = torch.zeros((vocab_size),requires_grad=True)
z = torch.zeros((vocab_size),requires_grad=True)
for idx,item in enumerate(x_nums):
    x[idx] = abs(int(item))+1
for idx,item in enumerate(y_nums):
    y[idx] = abs(int(item))+1
for idx,item in enumerate(z_nums):
    z[idx] = abs(int(item))+1

x = x.type(torch.LongTensor)
y = y.type(torch.LongTensor)
z = z.type(torch.LongTensor)

import matplotlib.pyplot as plt
def plot_data(x,y):
    plt.plot(x)
    plt.plot(y)
    plt.show()
plot_data(x,y)
epoch_num = 100
test = TransformerTranslator(token_size,num_blocks,num_heads,vocab_size)
optimizer = torch.optim.Adam(test.parameters(), lr=1e-2)
criterion = nn.MSELoss()
import random
for i in range(epoch_num):
    outs = []   
    avg_loss = []
    ix = x.unsqueeze(0)
    test.zero_grad()
    optimizer.zero_grad()
    torch_outs = torch.tensor([],requires_grad=True,dtype=torch.float32)
    iy_label_outs = torch.tensor([],requires_grad=True,dtype=torch.double)
    iy_label_outs_args = torch.tensor([0],requires_grad=False,dtype=torch.int64)
    for interval_jdx in range(1,101):
        if(interval_jdx!=1):
            if(random.random()>0.5):
                iy = iy_label_outs_args#y[:interval_jdx]
            else:
                iy = y[:interval_jdx]
        else:
            iy = torch.tensor([0],dtype=torch.int64)
        #Predict next token
        _ = test.encode(ix[:interval_jdx])
        iy = iy.unsqueeze(0)
        out = test.forward(iy) 

        iy_label_raw = np.zeros((vocab_size))
        iy_label_raw[iy[:,-1]] = 1
        iy_label = torch.tensor((iy_label_raw),requires_grad=True)

        torch_outs = torch.cat((torch_outs,out),dim=-1)
        iy_label_outs = torch.cat((iy_label_outs,iy_label),dim=-1)
        
        iy_label_outs_args = torch.cat((iy_label_outs_args,torch.tensor([torch.argmax(out)],dtype=torch.int64)),dim=-1)
        outs.append(torch.argmax(out.detach()))
    loss = criterion(iy_label_outs,torch_outs)
    loss.backward()
    optimizer.step()
    print("MEAN LOSS: {}".format(i),loss)
plot_data(y,outs)   


import random
for i in range(1):
    outs = []   
    avg_loss = []
    ix = x.unsqueeze(0)
    torch_outs = torch.tensor([],requires_grad=True,dtype=torch.float32)
    iy_label_outs = torch.tensor([],requires_grad=True,dtype=torch.double)
    iy_label_outs_args = torch.tensor([0],requires_grad=False,dtype=torch.int64)
    for interval_jdx in range(1,101):
        if(interval_jdx!=1):
            iy = iy_label_outs_args#y[:interval_jdx]
        else:
            iy = torch.tensor([0],dtype=torch.int64)
        #Predict next token
        _ = test.encode(ix[:interval_jdx])
        iy = iy.unsqueeze(0)
        out = test.forward(iy) 

        iy_label_raw = np.zeros((vocab_size))
        iy_label_raw[iy[:,-1]] = 1
        iy_label = torch.tensor((iy_label_raw),requires_grad=True)

        torch_outs = torch.cat((torch_outs,out),dim=-1)
        iy_label_outs = torch.cat((iy_label_outs,iy_label),dim=-1)
        
        iy_label_outs_args = torch.cat((iy_label_outs_args,torch.tensor([torch.argmax(out)],dtype=torch.int64)),dim=-1)
        outs.append(torch.argmax(out.detach()))

    loss = criterion(iy_label_outs,torch_outs)
    print("MEAN LOSS TEST: {}".format(i),loss)
plot_data(y,outs)