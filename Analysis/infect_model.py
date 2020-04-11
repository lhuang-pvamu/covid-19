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


# Standard Susceptible-Infectious-Recovered (SIR) model
# N: population
# Force of infection: lambda(I) = beta * I/N
# Rate of change of the number of susceptiable people: dS/dt = -lambda(I) * S
# Rate of change of the number of infectious people: dI/dt  = lambda(I)*S - gamma*I
# Rate of change of the number of recovered people: dR/dt = gamma * I
def SIR_model(S, I, R, N, beta, gamma, expo, start=0, end=0):
# Numpy array: S, I, R; Int: N; Tensor: beta, gamma
    #beta = gamma / (N/I)


    if end == 0:
        end = S.size

    Result_S = torch.zeros(end-start)
    Result_I = torch.zeros(end-start)
    Result_R = torch.zeros(end-start)

    for i in range(start, end):
        lam = beta * (I / N)
        dSdt = - (lam + expo) * S
        dIdt = lam*S - gamma*I
        dRdt = gamma*I
        S += dSdt
        I += dIdt
        R += dRdt
        Result_S[i-start] = S
        Result_I[i-start] = I
        Result_R[i-start] = R

    return Result_S, Result_I, Result_R




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    S, I, R = SIR_model(329466283, 1.0, 0.0, 329466283, 1.2, 0.10, 0.05, 0, 120)
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


