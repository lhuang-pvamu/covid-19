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


def gauss_model(config, positions, amp, span):

    x_limits = config['x_limits']
    nx = config['nx']
    dx = config['dx']

    xs = torch.from_numpy(x_limits[0] + np.arange(nx) * dx).double()

    #C = torch.ones(nx).float()

    dC = amp.double()*torch.exp(-((xs-positions.double())**2)/span.double())
    dC[torch.where(abs(dC) < 1e-7)] = 0

    #C = C + dC

    return dC

def poisson_model(config, mu, amp):
    x_limits = config['x_limits']
    nx = config['nx']
    dx = config['dx']

    #xs = torch.from_numpy(x_limits[0] + np.arange(nx) * dx).float()
    k = torch.arange(nx).float()

    xs = np.array([np.double(math.factorial(i)) for i in range(nx)])
    xf = torch.from_numpy(xs).double()
    print(xf)

    #C = torch.ones(nx).double()

    dC = (amp* (mu**k).double() * torch.exp(-mu.double()).double()/xf.double()).double()
    dC[torch.where(abs(dC) < 1e-7)] = 0

    #C = C + dC

    return dC

def erlang_model(config, mu, k, amp):
    x_limits = config['x_limits']
    nx = config['nx']
    dx = config['dx']

    xs = torch.from_numpy(x_limits[0] + np.arange(nx) * dx)

    #C = torch.ones(nx).double()

    dC = (amp* (mu**k) * (xs**(k-1)) * torch.exp(-mu * xs).double()) / math.factorial(k-1)
    dC[torch.where(abs(dC) < 1e-7)] = 0

    #C = C + dC

    return dC

def exp_model(config, amp, a, b):
    x_limits = config['x_limits']
    nx = config['nx']
    dx = config['dx']

    xs = torch.from_numpy(x_limits[0] + np.arange(nx) * dx).double()

    dC = amp / (torch.exp(a*xs.double()).double() + torch.exp(-b*xs).double())
    dC[torch.where(abs(dC) < 1e-7)] = 0

    return dC

def distance_from_target(data_c, target_c, data_d, target_d, pos_c, pos_d, span_c, span_d):
    return torch.mean((target_c - data_c)**2) + torch.mean((target_d - data_d)**2) +  \
            torch.abs((pos_d - pos_c) -7.0) * 100.0 + torch.abs((span_d - span_c)) * 100.0 + \
            ((torch.sum(data_c) - torch.sum(target_c)) / float(len(data_c))) ** 2 + \
            ((torch.sum(data_d) - torch.sum(target_d)) / float(len(data_d))) ** 2 + \
            ((torch.sum(target_c)*0.035 - torch.sum(target_d)) ** 2)



def fit(C, D, config, dist='gaussian'):
    _EPOCHS = 50000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #US: 35, 110000, 68
    #New York: 34.32, 47593, 39.11
    #Texas: 32.66, 1080, 43.87
    #Diff: 29, 12000, 47
    #print(torch.max(C))
    position_c = Variable(torch.FloatTensor([50.0]).double(), requires_grad=True)
    #amp = Variable(torch.FloatTensor([10000.0]), requires_grad=True)
    amp_c = Variable(torch.max(C).double(), requires_grad=True)
    span_c = Variable(torch.FloatTensor([100.0]).double(), requires_grad=True)
    position_d = Variable(torch.FloatTensor([60.0]).double(), requires_grad=True)
    amp_d = Variable(torch.max(D.double()), requires_grad=True)
    span_d = Variable(torch.FloatTensor([100.0]).double(), requires_grad=True)
    mu = Variable(torch.FloatTensor([0.1]).double(), requires_grad=True)
    k = Variable(torch.FloatTensor([0.1]).double(), requires_grad=True)
    print(position_c,amp_c,position_d,amp_d)

    learning = 0.1
    #criterion = nn.MSELoss()
    if dist == "gaussian":
        optimizer = torch.optim.Adam([position_c,amp_c,span_c,position_d,amp_d,span_d], lr=learning, weight_decay=1e-5)
    elif dist == 'exp':
        optimizer = torch.optim.Adam([amp_c, mu, k], lr=learning, weight_decay=1e-5)
    else:
        optimizer = torch.optim.Adam([mu, amp_c], lr=learning, weight_decay=1e-5)
    #lambda2 = lambda epoch: learning * (0.99 ** epoch)
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda2])

    for i in range(_EPOCHS):
        if dist == "gaussian":
            C0 = gauss_model(config, position_c, amp_c, span_c)
            D0 = gauss_model(config, position_d, amp_d, span_d)
        elif dist == "poisson":
            C0 = poisson_model(config, mu, amp_c)
        elif dist == 'exp':
            C0 = exp_model(config, amp_c, mu, k)
        else:
            C0 = erlang_model(config, mu, k, amp_c)
        optimizer.zero_grad()
        #print(C0[0:len(C)])
        #loss = criterion(C0[0:C.size(0)], C)
        loss = distance_from_target(C0[0:C.size(0)], C, D0[0:D.size(0)], D, position_c, position_d, span_c, span_d)
        loss.backward()
        optimizer.step()
        #scheduler.step(i)

        if i%5000 == 0:
            if dist == 'gaussian':
                print(i, loss.data.cpu().numpy(), position_c.data.cpu().numpy(), amp_c.data.cpu().numpy(), span_c.data.cpu().numpy(), position_d.data.cpu().numpy(), amp_d.data.cpu().numpy(), span_d.data.cpu().numpy())
            else:
                print(i, mu, k)
            xs = np.arange(config['nx']) * config['dx']
            plt.figure()
            #plt.subplot(2, 1, 1)
            #plt.plot(xs, C.detach().numpy(), label=r'$C$')
            #plt.legend()
            #plt.subplot(2, 1, 2)
            plt.plot(xs, C0.detach().numpy(), label=r'$Prediction$')
            plt.plot(C, label=r'$Confirmed$')
            plt.plot(xs, D0.detach().numpy(), label=r'$Prediction$')
            plt.plot(D, label=r'$Deaths$')
            plt.legend()
            plt.savefig('Figures/model'+str(i)+'.png')
            plt.close()
    return C0.detach().numpy(), position_c.detach().numpy()[0],amp_c.detach().numpy(),span_c.detach().numpy()[0], \
    D0.detach().numpy(), position_d.detach().numpy()[0],amp_d.detach().numpy(),span_d.detach().numpy()[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    config = dict()
    length = 10000
    config['x_limits'] = [0, length]
    config['nx'] = length+1
    config['dx'] = (config['x_limits'][1] - config['x_limits'][0]) / (config['nx'] - 1)

    #C = gauss_model(config, length/2, 20000.0, 50.0)
    #C = poisson_model(config, torch.FloatTensor([21.0]), 100000)
    #C = erlang_model(config, torch.FloatTensor([0.15]).double(), 10, 100000)
    C = exp_model(config, 10000.0, 0.050, 0.10)
    print(C)
    #fit(C, config)

    plt.plot(C.detach().numpy())
    plt.show()