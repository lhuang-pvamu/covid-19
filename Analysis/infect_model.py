__author__ = 'Lei Huang'

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import math


# Standard Susceptible-Exposable-Infectious-Recovered (SEIR) model
# N: population
# Force of infection: lambda(I) = beta * I/N
# Rate of change of the number of susceptiable people: dS/dt = -lambda(I) * S
# Rate of change of the number of infectious people: dI/dt  = lambda(I)*S - gamma*I
# Rate of change of the number of recovered people: dR/dt = gamma * I
def SEIR_model(S, I, R, N, beta, gamma, expo, start=0, end=0):
# Numpy array: S, I, R; Int: N; Tensor: beta, gamma
    #beta = gamma / (N/I)

    if end == 0:
        end = S.size

    Result_S = [] #torch.zeros(end-start).double()
    Result_I = [] #torch.zeros(end-start).double()
    Result_R = [] #torch.zeros(end-start).double()

    for i in range(start.int()):
        Result_S.append(S)
        Result_I.append(I)
        Result_R.append(R)

    for i in range(start.int(), end):
        lam = beta * (I / N)
        dSdt = - (lam + expo) * S
        dIdt = lam*S - gamma*I
        dRdt = gamma*I
        S1 = S + dSdt
        I1 = I + dIdt
        R1 = R + dRdt
        Result_S.append(S1)
        Result_I.append(I1)
        Result_R.append(R1)
        S = S1
        I = I1
        R = R1

    return torch.stack(Result_S), torch.stack(Result_I), torch.stack(Result_R)

def fit_SEIR(data_c, data_d, data_r, population, config):

    data_r *= 10.0

    beta = Variable(torch.tensor(0.80).double(), requires_grad=True)
    gamma = Variable(torch.tensor(0.12).double(), requires_grad=True)
    expo = Variable(torch.tensor(0.09).double(), requires_grad=True)
    start = Variable(torch.tensor(53.0), requires_grad=True)

    _EPOCHS = 5000
    learning = 0.01
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam([beta, gamma, expo, start], lr=learning,
                                     weight_decay=1e-5)
    # lambda2 = lambda epoch: learning * (0.99 ** epoch)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda2])

    end = data_c.size(0)
    R = torch.tensor(0.0).double()
    N = torch.tensor(population).double()
    S = torch.tensor(population).double()
    I = data_c[start.int()]

    #for i in range(end):
    #    if data_c[i] > 0:
    #        start = i
    #        I = data_c[i]
    #        break

    print("I====", I, data_c, data_c.size(0))
    for i in range(_EPOCHS):
        #print(S, I, R, N, start, end)
        s0, c0, r0 = SEIR_model(S, I, R, N, beta, gamma, expo, start, config['nx'])
        #print(c0.size())
        optimizer.zero_grad()
        # print(C0[0:len(C)])
        loss = criterion(c0[0:data_c.size(0)], data_c-data_d-data_r) + criterion(r0[0:data_d.size(0)], data_d+data_r)
        loss.backward()
        optimizer.step()

        if i%50 == 0:
            print(i, loss.data.cpu().numpy(), beta.data.cpu().numpy(), gamma.data.cpu().numpy(), expo.data.cpu().numpy(), start.data.cpu().numpy())

            xs = np.arange(config['nx']) * config['dx']
            plt.figure()
            #plt.subplot(2, 1, 1)
            #plt.plot(xs, C.detach().numpy(), label=r'$C$')
            #plt.legend()
            #plt.subplot(2, 1, 2)
            plt.plot(xs, c0.detach().numpy(), label=r'$Prediction$')
            plt.plot(data_c-data_d-data_r, label=r'$Confirmed$')
            #plt.plot(xs, d0.detach().numpy(), label=r'$Pred. Deaths$')
            plt.plot(data_d, label=r'$Deaths$')
            plt.plot(xs, r0.detach().numpy(), label=r'$Pred. Recovered$')
            plt.plot(data_r, label=r'Recovered')
            plt.legend()
            plt.savefig('Figures/model'+str(i)+'.png')
            plt.close()

    return s0, c0, r0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    S, I, R = SEIR_model(329466283, 1.0, 0.0, 329466283, 1.2, 0.10, 0.05, 0, 120)
    print(S)
    print(I)
    print(R)
    #fit(C, config)

    plt.plot(S.detach().numpy(), label='Susceptiable')
    plt.plot(I.detach().numpy(), label='Infectious')
    plt.plot(R.detach().numpy(), label='Recovered')
    plt.legend()
    plt.grid()
    plt.yscale("log")
    plt.show()


