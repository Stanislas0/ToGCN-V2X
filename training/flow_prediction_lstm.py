import os
import sys
import argparse
import random
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
from tensorboardX import SummaryWriter

sys.path.append("../")

from models.MyLSTM import *

N = 81  # Number of keypoints
dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def training(input_tensor, target_tensor, model, optimizer, criterion):
    timestep = input_tensor.shape[1]  # Length of input time interval (10 min)
    # Reset gradient to zero for each batch.
    optimizer.zero_grad()

    # Encode history flow map
    hidden_state = None
    for i in range(timestep):
        output, hidden_state = model(input_tensor[:, i], hidden_state)

    output, _ = model(output, hidden_state)
    loss = criterion(output, target_tensor)
    # Back-propagation to update parameters
    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate(input_tensor, target_tensor, model, criterion):
    timestep = input_tensor.shape[1]
    with torch.no_grad():
        hidden_state = None
        for i in range(timestep):
            output, hidden_state = model(input_tensor[:, i], hidden_state)

        output, _ = model(output, hidden_state)
        loss = criterion(output, target_tensor)
    return output, loss.item()


def training_iter(args):

    hidden_size = args.hidden_size
    batch_size = args.batch_size
    epoch = args.epoch
    learning_rate = args.learning_rate
    path = args.path
    data_path = os.path.join(path, "moving_sample")
    train_x = torch.tensor(np.load(os.path.join(data_path, "train_x.npy")), device=device, dtype=dtype)
    train_y = torch.tensor(np.load(os.path.join(data_path, "train_y.npy")), device=device, dtype=dtype)
    test_x = torch.tensor(np.load(os.path.join(data_path, "test_x.npy")), device=device, dtype=dtype)
    test_y = torch.tensor(np.load(os.path.join(data_path, "test_y.npy")), device=device, dtype=dtype)

    # Adam optimizer
    model = LSTM(N, hidden_size, N)
    if torch.cuda.is_available():
        model.cuda()

    # Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    model_name = time.strftime('%Y_%m_%d_%H%M', time.localtime(time.time()))
    output_path = os.path.join(args.output_path, model_name)
    writer = SummaryWriter()
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(os.path.join(output_path, "checkpoints")):
        os.mkdir(os.path.join(output_path, "checkpoints"))
    if not os.path.exists(os.path.join(output_path, "outputs")):
        os.mkdir(os.path.join(output_path, "outputs"))

    with open(os.path.join(output_path, 'configuration.txt'), 'w') as f:
        json.dump(model_name, f)
        json.dump(args.__dict__, f, indent=2)

    # Mini-batch training
    for t in range(epoch):
        for i in range(train_x.shape[0] // batch_size):
            input_tensor = train_x[i * batch_size:(i + 1) * batch_size]
            target_tensor = train_y[i * batch_size:(i + 1) * batch_size]
            loss = training(input_tensor, target_tensor, model, optimizer, criterion)

        # Total evaluation
        train_output, train_loss = evaluate(train_x, train_y, model, criterion)
        test_output, test_loss = evaluate(test_x, test_y, model, criterion)
        # RMSE loss
        train_rmse_loss = np.sqrt(train_loss / N)
        test_rmse_loss = np.sqrt(test_loss / N)
        # rRMSE loss
        train_rrmse_loss = train_rmse_loss / torch.mean(train_y)
        test_rrmse_loss = test_rmse_loss / torch.mean(test_y)

        print("Epoch {}, training RMSE loss {:.4f}, testing RMSE loss {:.4f}, "
              "training rRMSE loss {:.4f}, testing rRMSE loss {:.4f}".format(
            t, train_rmse_loss, test_rmse_loss, train_rrmse_loss, test_rrmse_loss))

        # Tensorboard visualization
        writer.add_scalars('model_{}/RMSE_losses'.format(model_name), {"train_rmse_loss": train_rmse_loss,
                                                                       "test_rmse_loss": test_rmse_loss}, t)
        writer.add_scalars('model_{}/rRMSE_losses'.format(model_name), {"train_rrmse_loss": train_rrmse_loss,
                                                                        "test_rrmse_loss": test_rrmse_loss}, t)

        # Save models and results
        if t != 0 and t % 200 == 0:
            torch.save(model, os.path.join(output_path, "checkpoints/lstm_epoch_{}.pth".format(t)))
            np.save(os.path.join(output_path, "outputs/train_output_{}.npy".format(t)), train_output.cpu().detach().numpy())
            np.save(os.path.join(output_path, "outputs/test_output_{}.npy".format(t)), test_output.cpu().detach().numpy())

            print("Checkpoint saved.")

    torch.save(model, os.path.join(output_path, "checkpoints/lstm_epoch_{}.pth".format(t)))
    np.save(os.path.join(output_path, "outputs/train_output_{}.npy".format(t)), train_output.cpu().detach().numpy())
    np.save(os.path.join(output_path, "outputs/test_output_{}.npy".format(t)), test_output.cpu().detach().numpy())

    print("Training finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="LSTM", default="LSTM")
    parser.add_argument("-t", "--train", help="Boolean, True if want to train the model.", default=False)
    parser.add_argument("-p", "--path", help="Path of datasets.", default="../datasets")
    parser.add_argument("-o", "--output_path", help="Path of checkpoints and results.", default="../results")
    parser.add_argument("-l", "--learning_rate", help="Initial learning rate.", default=0.001)
    parser.add_argument("-e", "--epoch", help="Number of epoch.", default=10000)
    parser.add_argument("-b", "--batch_size", help="Batch size.", default=64)
    parser.add_argument("-hs", "--hidden_size", help="Hidden_size of LSTM units.", default=128)
    parser.add_argument("-dt", "--decoder_t", help="Parameter of decoder.", default=3)

    args = parser.parse_args()

    if args.train:
        training_iter(args)