import torch
import torch.nn as nn
from torch.autograd import variable
from myModel import EncodingForcasting
from myModel import MyCrossEntropyLoss
from movingmnist_iterator import MovingMNISTIterator
import myNN
import numpy as np
from PIL import Image
import gc

def main():
    # set parameter for net
    layer_numbers = 2
    encoding_input_channels = 1
    encoding_hidden_channels = (8, 8)
    forcasting_input_channels = 8
    forcasting_hidden_channels = (8, 8)
    shape = (64, 64)
    kernel_size = (5, 5)
    input_seq_number = 10
    output_seq_number = 10

    # creat and load ConvLSTM model
    convLSTM = EncodingForcasting(layer_numbers, encoding_input_channels, encoding_hidden_channels, forcasting_input_channels, forcasting_hidden_channels, shape, kernel_size)
    # criterion = nn.MSELoss()
    criterion = MyCrossEntropyLoss()
    optimizer = torch.optim.SGD(convLSTM.parameters(), lr = 1e-2)

    # set optimization parameter
    maxEpoch = 1000
    batch_size = 8
    train_batchsize_number = 313*4
    valid_batchsize_number = 63*4
    test_batchsize_number = 94*4
    for it in range(maxEpoch):
        losses_train = []
        for batch_i in range(0, 2):
            # load train data based on batch

            data_path_input = 'E:/data/moving_mnist_data/train_input_' + str(batch_i) + '.npy'
            data_path_output = 'E:/data/moving_mnist_data/train_output_' + str(batch_i) + '.npy'
            xtrain = np.load(data_path_input)
            ytrain = np.load(data_path_output)
            xtrain = variable(torch.tensor(xtrain))
            ytrain = variable(torch.tensor(ytrain))
            #forward
            out = convLSTM(xtrain, output_seq_number)
            for i in range(input_seq_number):
                pic = out[i, 0, 0, :, :] * 255.0
                img = Image.fromarray( pic.data.numpy())  # numpy 转 image类
                img = img.convert('RGB')
                data_path = 'E:/data/moving_mnist_data_predit/output_' + str(batch_i)+ '_' +str(i) + '.png'
                img.save(data_path)
            loss = criterion(out, ytrain)
            print('batch [{}/{}], loss:{:.6f}'.format(batch_i + 1, train_batchsize_number, loss.data.item()))
            losses_train.append(loss.data.item())
            # backward

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del data_path_input
            del data_path_output
            del xtrain
            del ytrain
            del out
            del pic
            del img
            gc.collect()

        mean_losses_train = np.mean(losses_train)

        if it%1 == 0:
            print('epoch[{}/{}], mean loss:{:.6f}' .format(it+1, 100, mean_losses_train))
        if it%30 == 0:
            losses_valid = []
            for batch_j in range(0, 2):
                # load train data based on batch
                data_path_input = 'E:/data/moving_mnist_data/valid_input_' + str(batch_j) + '.npy'
                data_path_output = 'E:/data/moving_mnist_data/valid_output_' + str(batch_j) + '.npy'
                xvalid = np.load(data_path_input)
                yvalid = np.load(data_path_output)
                xvalid = variable(torch.tensor(xvalid))
                yvalid = variable(torch.tensor(yvalid))
                # forward
                out = convLSTM(xvalid, output_seq_number)

                loss = criterion(out, yvalid)
                losses_valid.append(loss.data.item())
                del data_path_input
                del data_path_output
                del xvalid
                del yvalid
                del out
                gc.collect()
            mean_losses_valid = np.mean(losses_valid)
            print('Valid step, mean loss:{:.6f}'.format(mean_losses_valid))

if __name__ == '__main__':
    main()