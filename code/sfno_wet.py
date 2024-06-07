import torch
import torch.nn as nn
import torch.nn.functional as F
import os
try:
    from neuralop.models import SFNO
except:
    os.system('pip install neuraloperator torch_harmonics wandb')
from neuralop.models import SFNO
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
import operator
from functools import reduce, partial
from timeit import default_timer
import pickle
import matplotlib.pyplot as plt
from util import UnitGaussianNormalizer, LpLoss, rfft, irfft
import argparse

def load_data(file):
    data_load_file = []
    file_1 = open(file, "rb")
    data_load_file = pickle.load(file_1)
    return data_load_file

def parse_args():
    parser = argparse.ArgumentParser(description='Cell Division Tension Training')

    parser.add_argument('--train_path', type=str, default='/wanghaixin/Cell_Division/wet_636_train.pt', help='Path to the training data')
    parser.add_argument('--test_path', type=str, default='/wanghaixin/Cell_Division/wet_160_val.pt', help='Path to the test data')
    parser.add_argument('--use_grid', type=bool, default=False, help='Use grid search')
    parser.add_argument('--ntrain', type=int, default=636, help='Number of training samples')
    parser.add_argument('--ntest', type=int, default=160, help='Number of test samples')
    parser.add_argument('--ts', type=int, default=20, help='Time steps')
    parser.add_argument('--seed', type=list, default=[42,88,2024], help='random seed')

    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')

    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--step_size', type=int, default=100, help='Step size for learning rate decay')
    parser.add_argument('--gamma', type=float, default=0.5, help='Gamma for learning rate decay')

    parser.add_argument('--k', type=int, default=3, help='Number of steps for k-step prediction')
    parser.add_argument('--N', type=int, default=1, help='Sliding window size')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print('k = ', args.k, args.train_path)

    TRAIN_PATH = args.train_path
    TEST_PATH = args.test_path
    use_grid = args.use_grid
    ntrain = args.ntrain
    ntest = args.ntest
    ts = args.ts
    seeds = args.seed

    batch_size = args.batch_size
    learning_rate = args.learning_rate

    epochs = args.epochs
    step_size = args.step_size
    gamma = args.gamma

    k = args.k #基于k步预测k步
    N = args.N #sliding window
    if use_grid:
        in_chan = k + 2
    else:
        in_chan = k
    out_chan = k
    modes = 12
    width = 32

    r = 5
    h = int(((421 - 1)/r) + 1)
    s = 32

    ################################################################
    # load data and data normalization
    ################################################################

    file_train = torch.load(TRAIN_PATH)
    file_test = torch.load(TEST_PATH)
    
    file_normalizer = UnitGaussianNormalizer(file_train)
    file_train = file_normalizer.encode(file_train)
    file_test = file_normalizer.encode(file_test)
   
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(file_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(file_test), batch_size=batch_size, shuffle=False)
    
    # print(file_test.shape,'=========')

    ###############
    # training and evaluation
    ##########################
    best_test_l2_seq = []
    best_test_mse_seq = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = SFNO(n_modes=(16, 16), hidden_channels=64,
                in_channels=args.k, out_channels=args.k).cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        best_test_mse = 100000
        best_test_l2 = 100000
        for ep in range(epochs):
            model.train()
            t1 = default_timer()
            train_mse = 0
            train_l2_full = 0
            for inputs in train_loader:
                inputs = inputs[0].cuda() # [bs, 32, 32, in_chan]
                if use_grid:
                    num_timesteps = inputs.shape[1] - 2
                else:
                    num_timesteps = inputs.shape[1]
                seq_loss = 0
                seq_l2_loss = 0
                for t in range(0, num_timesteps - 2*k + 1, N):
                    
                    x = inputs[:,t:t+k, :, :]
                    y = inputs[:, t+k:t+2*k, :, :]
                    optimizer.zero_grad()
                    
                    out = model(x)
                    # print(out.shape, y.shape)
                    out = out.reshape(x.shape[0],-1)
                    y = y.reshape(y.shape[0],-1)
                    loss = F.mse_loss(out, y, reduction='mean')
                    loss.backward()
                    optimizer.step()
                    seq_loss += loss.item()
                    l2_full = (torch.norm((out - y).flatten(start_dim=1), dim=1) / torch.norm(y.flatten(start_dim=1), dim=1)).mean(0)
                    seq_l2_loss += l2_full.item()
                seq_loss/=len(range(0, num_timesteps - 2*k + 1, N))
                seq_l2_loss/=len(range(0, num_timesteps - 2*k + 1, N))
                train_mse +=seq_loss
                train_l2_full += seq_l2_loss
            scheduler.step()
            train_mse /= len(train_loader)
            train_l2_full /= len(train_loader)
            t2 = default_timer()
            if (ep+1) % 100 ==0:
                print(ep, t2-t1, "Train MSE:", train_mse, '\t', "Train Relative L2 Norm:",train_l2_full)
            
            model.eval()
            t1 = default_timer()
            with torch.no_grad():
                test_mse = 0
                test_l2_full = 0
                for inputs in test_loader:
                    inputs = inputs[0].cuda() # [bs, 32, 32, 20+2]
                    if use_grid:
                        num_timesteps = inputs.shape[1] - 2
                    else:
                        num_timesteps = inputs.shape[1]
                    seq_loss = 0
                    seq_l2_loss = 0
                    for t in range(0, num_timesteps - 2*k +1, N):
                        x = inputs[:,t:t+k, :, :]
                        y = inputs[:, t+k:t+2*k, :, :]
                        out = model(x)
                        out = out.reshape(x.shape[0],-1)
                        y = y.reshape(y.shape[0],-1)
                        loss = F.mse_loss(out, y, reduction='mean')
                        seq_loss += loss.item()
                        l2_full = (torch.norm((out - y).flatten(start_dim=1), dim=1) / torch.norm(y.flatten(start_dim=1), dim=1)).mean(0)
                        seq_l2_loss += l2_full.item()
                    seq_loss/=len(range(0, num_timesteps - 2*k + 1, N))
                    seq_l2_loss/=len(range(0, num_timesteps - 2*k + 1, N))
                    test_mse +=seq_loss
                    test_l2_full += seq_l2_loss
                    
                test_mse /= len(test_loader)
                test_l2_full /= len(test_loader)
                
                t2 = default_timer()
                if (ep+1) % 100 ==0:
                    print(ep, t2-t1, "Test MSE:", test_mse, '\t', "Test Relative L2 Norm:",test_l2_full)
                if test_mse < best_test_mse:
                    best_test_mse = test_mse
                    torch.save(model.state_dict(), '/wanghaixin/Cell_Division/ckpt/Best_SFNO_wet_{}.pth'.format(args.k))
                if test_l2_full < best_test_l2:
                    best_test_l2 = test_l2_full
        best_test_l2_seq.append(best_test_l2)
        best_test_mse_seq.append(best_test_mse)
    print(best_test_l2_seq,'MSE:',np.mean(best_test_l2_seq),np.var(best_test_l2_seq))
    print(best_test_mse_seq,'L2:',np.mean(best_test_mse_seq),np.var(best_test_mse_seq))
