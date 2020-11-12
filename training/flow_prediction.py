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

from models.MyGNN import *

N = 81  # Number of keypoints
dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def training(input_tensor, target_tensor, adjacent_matrix, encoder, decoder,
             encoder_optimizer, decoder_optimizer, criterion):

    T = 3  # Decoder hyper-parameter
    teacher_forcing_ratio = 0.5
    loss = torch.zeros(1)
    timestep_1 = input_tensor.shape[1]  # Length of input time interval (10 min each)
    timestep_2 = target_tensor.shape[1]  # Length of output time interval (10 min each)

    # Reset gradient to zero for each batch.
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Encode history flow map
    encoder_hidden = None
    for ei in range(timestep_1):
        encoder_input = input_tensor[:, ei]
        encoder_output, encoder_hidden = encoder(encoder_input, adjacent_matrix, encoder_hidden)

    # Decode to predict future flow map
    decoder_hidden = encoder_hidden
    for di in range(T):
        decoder_input = input_tensor[:, timestep_1 - (T - di) - 1].clone().detach()
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

    decoder_input = input_tensor[:, timestep_1 - 1].clone().detach()

    # Teacher forcing mechanism.
    if random.random() < teacher_forcing_ratio:
        use_teacher_forcing = True
    else:
        use_teacher_forcing = False

    if use_teacher_forcing:
        for di in range(timestep_2):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[:, di])
            decoder_input = target_tensor[:, di]
    else:
        for di in range(timestep_2):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[:, di])
            decoder_input = decoder_output

    # Back-propagation to update parameters
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()


def evaluate(input_tensor, target_tensor, adjacent_matrix, encoder, decoder, criterion):

    T = 3  # Decoder hyper-parameter
    timestep_1 = input_tensor.shape[1]
    timestep_2 = target_tensor.shape[1]
    loss = torch.zeros(1)

    with torch.no_grad():
        encoder_hidden = None
        for ei in range(timestep_1):
            encoder_input = input_tensor[:, ei]
            encoder_output, encoder_hidden = encoder(encoder_input, adjacent_matrix, encoder_hidden)

        decoder_hidden = encoder_hidden
        for di in range(T):
            decoder_input = input_tensor[:, timestep_1 - (T - di) - 1].clone().detach()
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

        decoder_input = input_tensor[:, timestep_1 - 1].clone().detach()
        for di in range(timestep_2):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            target_tensor = target_tensor.view(target_tensor.size(0), target_tensor.size(2))
            loss += criterion(decoder_output, target_tensor)
            decoder_input = decoder_output

    return decoder_output, loss.item()


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
    # Add one dimension
    train_y = train_y.unsqueeze(1)
    test_y = test_y.unsqueeze(1)

    A = torch.tensor(np.load(os.path.join(path, "adjacent_matrix/adjacent_matrix_gcn.npy")), device=device, dtype=dtype)
    encoder = Encoder(N, hidden_size)
    decoder = Decoder(N, hidden_size, N)
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    # Adam optimizer
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    model_name = time.strftime('%Y_%m_%d_%H%M',time.localtime(time.time()))
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
            loss = training(input_tensor, target_tensor, A, encoder, decoder,
                            encoder_optimizer, decoder_optimizer, criterion)

        # Total evaluation
        train_output, train_loss = evaluate(train_x, train_y, A, encoder, decoder, criterion)
        test_output, test_loss = evaluate(test_x, test_y, A, encoder, decoder, criterion)
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
            torch.save(encoder, os.path.join(output_path, "checkpoints/encoder_epoch_{}.pth".format(t)))
            torch.save(decoder, os.path.join(output_path, "checkpoints/decoder_epoch_{}.pth".format(t)))
            np.save(os.path.join(output_path, "outputs/train_output_{}.npy".format(t)), train_output.cpu().detach().numpy())
            np.save(os.path.join(output_path, "outputs/test_output_{}.npy".format(t)), test_output.cpu().detach().numpy())

            print("Checkpoint saved.")

    torch.save(encoder, os.path.join(output_path, "checkpoints/encoder_epoch_{}.pth".format(t)))
    torch.save(decoder, os.path.join(output_path, "checkpoints/decoder_epoch_{}.pth".format(t)))
    np.save(os.path.join(output_path, "outputs/train_output_{}.npy".format(t)), train_output.cpu().detach().numpy())
    np.save(os.path.join(output_path, "outputs/test_output_{}.npy".format(t)), test_output.cpu().detach().numpy())

    print("Training finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="GNN", default="GNN")
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