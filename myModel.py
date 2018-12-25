import torch
import torch.nn as nn
from torch.autograd import variable
from myNN import ConvLSTM

class Encoding(nn.Module):
    def __init__(self, layer_numbers, input_channels, hidden_channels, shape, kernel_size = (3, 3), stride = 1):
        super(Encoding, self).__init__()
        self.layer_numbers = layer_numbers
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.shape = shape
        self.kernel_size = kernel_size
        self.stride =stride
        assert self.layer_numbers == len(self.hidden_channels)

        self.layers = ConvLSTM(self.layer_numbers, self.input_channels, self.hidden_channels, self.shape, self.kernel_size, self.stride)

    def forward(self, input, hidden_states = None):
        output, all_hidden_states = self.layers(input, hidden_states)

        return all_hidden_states


class Forcasting(nn.Module):
    def __init__(self, layer_numbers, input_channels, hidden_channels, shape, kernel_size=(3, 3), stride=1):
        super(Forcasting, self).__init__()
        self.layer_numbers = layer_numbers
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.shape = shape
        self.kernel_size = kernel_size
        self.stride = stride
        assert self.layer_numbers == len(self.hidden_channels)

        self.layers = ConvLSTM(self.layer_numbers, self.input_channels, self.hidden_channels, self.shape,
                               self.kernel_size, self.stride)

    def forward(self, output_stepnumber, hidden_states):
        assert hidden_states != None
        input = variable(torch.tensor(torch.zeros(output_stepnumber, hidden_states[0][0].size()[0], hidden_states[0][0].size()[1], hidden_states[0][0].size()[2], hidden_states[0][0].size()[3])))
        output, all_hidden_states = self.layers(input, hidden_states)

        return output, all_hidden_states

class EncodingForcasting(nn.Module):
    def __init__(self, layer_numbers, encoding_input_channels, encoding_hidden_channels, forcasting_input_channels, forcasting_hidden_channels, shape, kernel_size=(3, 3), stride=1):
        super(EncodingForcasting, self).__init__()
        self.layer_numbers = layer_numbers
        self.encoding_input_channels = encoding_input_channels
        self.encoding_hidden_channels = encoding_hidden_channels
        self.forcasting_input_channels = forcasting_input_channels
        self.forcasting_hidden_channels = forcasting_hidden_channels
        self.shape = shape
        self.kernel_size = kernel_size
        self.stride = stride
        assert self.layer_numbers == len(self.encoding_hidden_channels)
        assert self.layer_numbers == len(self.forcasting_hidden_channels)

        self.encoding_layers = Encoding(self.layer_numbers, self.encoding_input_channels, self.encoding_hidden_channels, self.shape,
                               self.kernel_size, self.stride)
        self.forcasting_layers = Forcasting(self.layer_numbers, self.forcasting_input_channels, self.forcasting_hidden_channels, self.shape,
                               self.kernel_size, self.stride)
        self.predict_layers = nn.Conv2d(in_channels=self.forcasting_hidden_channels[-1]*self.layer_numbers,
                                        out_channels=self.encoding_input_channels,
                                        kernel_size=(1,1),
                                        stride= 1,
                                        padding= 0,
                                        bias=False)

    def forward(self, input, output_stepnumber, hidden_states=None):
        step_numbers = input.size()[0]
        batch_size = input.size()[1]
        encoding_all_hidden_states = self.encoding_layers(input, hidden_states)
        forcasting_hidden_states = encoding_all_hidden_states[-1]
        forcasting_output, forcasting_all_hidden_states = self.forcasting_layers(output_stepnumber, forcasting_hidden_states)
        a = forcasting_all_hidden_states[:][:][0]
        forcasting_all_hidden_states = list(forcasting_all_hidden_states)
        forcasting_all_hidden_states_tensor = []
        for i in forcasting_all_hidden_states:
            temp_i = []
            for j in i:
                temp_i.append(j[0])
            forcasting_all_hidden_states_tensor.append(torch.cat(temp_i, 0).view(self.layer_numbers, batch_size, self.forcasting_hidden_channels[-1], self.shape[0], self.shape[1]))
        forcasting_all_hidden_states_tensor = torch.cat(forcasting_all_hidden_states_tensor, 0).view(output_stepnumber, self.layer_numbers, batch_size, self.forcasting_hidden_channels[-1], self.shape[0], self.shape[1])
        output = []
        for i in range(forcasting_all_hidden_states_tensor.data.size()[0]):
            temp_data = forcasting_all_hidden_states_tensor[i, :, :, :, :]
            temp_data = temp_data.view(temp_data.data.size()[1], temp_data.data.size()[0]*temp_data.data.size()[2], temp_data.data.size()[3], temp_data.data.size()[4])
            output.append(self.predict_layers(temp_data))
        output = torch.sigmoid(torch.cat(output, 0).view(forcasting_all_hidden_states_tensor.data.size()[0], output[0].data.size()[0], output[0].data.size()[1], output[0].data.size()[2], output[0].data.size()[3]))
        return output

class MyCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MyCrossEntropyLoss, self).__init__()

    def forward(self, predit, truth):
        out = - torch.mean(truth * torch.log(predit) + (1 - truth) * torch.log(1 - predit))
        return out

if __name__ == '__main__':
    layer_numbers = 3
    encoding_input_channels = 3
    encoding_hidden_channels = (128, 128, 128)
    forcasting_input_channels = 128
    forcasting_hidden_channels = (128, 128, 128)
    shape = (28, 28)
    predict_number = 1


    convLSTM = EncodingForcasting(layer_numbers, encoding_input_channels, encoding_hidden_channels, forcasting_input_channels, forcasting_hidden_channels, shape)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(convLSTM.parameters(), lr = 1e-3)


    xtrain = variable(torch.randn(3, 100, 3, 28, 28))
    ytrain = variable(torch.randn(predict_number,100, 3, 28, 28))

    for it in range(1000):
        for batch_i in range(0, int(100/2)):
            #forward
            out = convLSTM(xtrain[:,batch_i*2:(batch_i+1)*2,:,:], predict_number)

            loss = criterion(out, ytrain[:,batch_i*2:(batch_i+1)*2,:,:,:])
            # backward

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if it%1 == 0:
            print('epoch[{}/{}], loss:{:.6f}' .format(it+1, 100, loss.data.item()))






