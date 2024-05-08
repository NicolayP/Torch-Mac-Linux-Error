import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

torch.manual_seed(50)

LEN_STATE=6

class NNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(12, 256)
        self.act1 = nn.LeakyReLU()
        self.hidden2 = nn.Linear(256, 256)
        self.act2 = nn.LeakyReLU()
        self.output = nn.Linear(256, 6)
        self.act_output = nn.Sigmoid()
 
    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.output(x)
        return x

def train():
    model = NNModel()
    model.to(torch.float64)


    df = pd.read_csv('odom-12-01-2024-run1.csv', sep=',', header=0)

    torch_data = torch.tensor(df.values)
    v_t0 = torch_data[:-1, 8:14]
    F_t0 = torch_data[:-1, 20:26]
    # Input
    X = torch.cat((v_t0,F_t0),1)

    # Target
    v_t1 = torch_data[1:, 8:14]
    y = v_t1 - v_t0

    loss_fn = nn.MSELoss()  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    n_epochs = 100
    batch_size = 100
    
    for epoch in range(n_epochs):
        for i in range(0, len(X), batch_size):
            Xbatch = X[i:i+batch_size]
            y_pred = model(Xbatch)
            ybatch = y[i:i+batch_size]
            loss = loss_fn(y_pred, ybatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Finished epoch {epoch}, latest loss {loss}')

    StartPoint = 500
    LEN = 200

    v0 = v_t0[StartPoint]

    full_state = np.zeros((LEN+1,LEN_STATE))
    full_state[0] = v0
    for i in range(StartPoint,StartPoint+LEN):
        
        dv  = model(torch.cat((v0,F_t0[i]),0))
        v0 = v0 + dv 
        full_state[i-StartPoint+1] = v0.detach().numpy()

    # INFERENCE
    StartPoint = 500
    LEN = 200

    v0 = v_t0[StartPoint]

    full_state = np.zeros((LEN+1,LEN_STATE))
    full_state[0] = v0
    for i in range(StartPoint,StartPoint+LEN):
        
        dv  = model(torch.cat((v0,F_t0[i]),0))
        v0 = v0 + dv 
        full_state[i-StartPoint+1] = v0.detach().numpy()


    vel_loss = loss_fn(torch.from_numpy(full_state), v_t0[StartPoint:StartPoint+1+LEN])

    dt = 0.1
    time = np.linspace(0, LEN*dt, LEN+1) 
    v_t0 = v_t0.detach().numpy()

    plt.plot(time[0:LEN], v_t0[StartPoint:StartPoint+LEN,0], linewidth=2)
    plt.plot(time[0:LEN], full_state[0:LEN,0], linewidth=2)
    plt.legend(['GT', 'NN'])
    plt.ylabel('Velocity [m/s]')
    plt.xlabel('Time [s]')
    plt.savefig('vx_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    # plt.show()

    plt.plot(time[0:LEN], v_t0[StartPoint:StartPoint+LEN,1], linewidth=2)
    plt.plot(time[0:LEN], full_state[0:LEN,1], linewidth=2)
    plt.legend(['GT', 'NN'])
    plt.ylabel('Velocity [m/s]')
    plt.xlabel('Time [s]')
    plt.savefig('vy_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    # plt.show()

    plt.plot(time[0:LEN], v_t0[StartPoint:StartPoint+LEN,2], linewidth=2)
    plt.plot(time[0:LEN], full_state[0:LEN,2], linewidth=2)
    plt.legend(['GT', 'NN'])
    plt.ylabel('Velocity [m/s]')
    plt.xlabel('Time [s]')
    plt.savefig('superiority_vz_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    print("Inference Loss:", vel_loss)

train()