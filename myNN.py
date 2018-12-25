import torch
import torch.nn as nn
from torch.autograd import variable

# the basic core of convLSTM
class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, shape, kernel_size = (3, 3), stride=1):
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.shape = shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size[0] // 2
        # parameters of input gate
        self.Wxi = nn.Conv2d(in_channels=self.input_channels,
                             out_channels=self.hidden_channels,
                             kernel_size=self.kernel_size,
                             stride=self.stride,
                             padding=self.padding,
                             bias=False)
        self.Whi = nn.Conv2d(in_channels=self.hidden_channels,
                             out_channels=self.hidden_channels,
                             kernel_size=self.kernel_size,
                             stride=self.stride,
                             padding=self.padding,
                             bias=False)
        self.Wci = variable(torch.zeros(1, self.hidden_channels, self.shape[0], self.shape[1]))
        # parameters of forget gate
        self.Wxf = nn.Conv2d(in_channels=self.input_channels,
                             out_channels=self.hidden_channels,
                             kernel_size=self.kernel_size,
                             stride=self.stride,
                             padding=self.padding,
                             bias=False)
        self.Whf = nn.Conv2d(in_channels=self.hidden_channels,
                             out_channels=self.hidden_channels,
                             kernel_size=self.kernel_size,
                             stride=self.stride,
                             padding=self.padding,
                             bias=False)
        self.Wcf = variable(torch.zeros(1, self.hidden_channels, self.shape[0], self.shape[1]))
        # parameters of output gate
        self.Wxo = nn.Conv2d(in_channels=self.input_channels,
                             out_channels=self.hidden_channels,
                             kernel_size=self.kernel_size,
                             stride=self.stride,
                             padding=self.padding,
                             bias=False)
        self.Who = nn.Conv2d(in_channels=self.hidden_channels,
                             out_channels=self.hidden_channels,
                             kernel_size=self.kernel_size,
                             stride=self.stride,
                             padding=self.padding,
                             bias=False)
        self.Wco = variable(torch.zeros(1, self.hidden_channels, self.shape[0], self.shape[1]))

        # parameters of hidden state
        self.Wxc = nn.Conv2d(in_channels=self.input_channels,
                             out_channels=self.hidden_channels,
                             kernel_size=self.kernel_size,
                             stride=self.stride,
                             padding=self.padding,
                             bias=False)
        self.Whc = nn.Conv2d(in_channels=self.hidden_channels,
                             out_channels=self.hidden_channels,
                             kernel_size=self.kernel_size,
                             stride=self.stride,
                             padding=self.padding,
                             bias=True)

    def forward(self, input, hidden_state):

        ht_1, ct_1 = hidden_state
        it = torch.sigmoid(self.Wxi(input) + self.Whi(ht_1) + self.Wci * ct_1)
        ft = torch.sigmoid(self.Wxf(input) + self.Whf(ht_1) + self.Wcf * ct_1)
        ot = torch.sigmoid(self.Wxo(input) + self.Who(ht_1) + self.Wco * ct_1)
        ct = ft * ct_1 + it * torch.tanh(self.Wxc(input) + self.Whc(ht_1))
        ht = ot * torch.tanh(ct)

        return ht, (ht, ct)

# the extensional LSTM core besed on ConvLSTMcell
class ConvLSTM(nn.Module):
    def __init__(self, layer_numbers, input_channels, hidden_channels, shape, kernel_size = (3, 3), stride=1):
        super(ConvLSTM, self).__init__()
        self.layer_numbers = layer_numbers
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.shape = shape
        self.kernel_size = kernel_size
        self.stride = stride
        self._layers = []

        assert self.layer_numbers == len(hidden_channels)
        cell = ConvLSTMCell(self.input_channels, self.hidden_channels[0], self.shape, kernel_size=self.kernel_size,
                            stride=self.stride)
        name = 'cell{}'.format(0)
        setattr(self, name, cell)
        self._layers.append(cell)
        for i in range(1, self.layer_numbers):
            hidden_channel = self.hidden_channels[i]
            cell = ConvLSTMCell(self.hidden_channels[i-1], self.hidden_channels[i], self.shape, kernel_size = self.kernel_size, stride = self.stride)
            name = 'cell{}'.format(i)
            setattr(self, name, cell)
            self._layers.append(cell)


    def forward(self, input, hidden_states = None):

        output = []
        state = []
        step_numbers = input.size()[0]
        batch_size = input.size()[1]
        if hidden_states == None:
            hidden_states = []
            for i in range(self.layer_numbers):
                hidden_state_i = (variable(torch.zeros(batch_size, self.hidden_channels[i], self.shape[0], self.shape[1])),
                                  variable(torch.zeros(batch_size, self.hidden_channels[i], self.shape[0], self.shape[1])))
                hidden_states.append(hidden_state_i)
        hidden_states = list(hidden_states)
        for t in range(step_numbers):
            x = input[t,:,:,:,:]
            for i in range(self.layer_numbers):
                name = 'cell{}'.format(i)
                x, hidden_states[i] = getattr(self, name)(x, hidden_states[i])
            output.append(x)
            state.append(tuple(hidden_states))
        output = torch.cat(output, 0).view(step_numbers, batch_size, self.hidden_channels[i], self.shape[0], self.shape[1])
        state = tuple(state)
        return output, state

if __name__ == '__main__':

    input_channels = 3
    hidden_channels = (8,5)
    layer_numbers = 2
    shape = (28, 28)

    convLSTM = ConvLSTM(layer_numbers, input_channels, hidden_channels, shape)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(convLSTM.parameters(), lr = 1e-3)


    xtrain = variable(torch.randn(3, 100, 3, 28, 28))
    ytrain = variable(torch.randn(100, 5, 28, 28))

    for it in range(1000):
        for batch_i in range(0, int(100/2)):
            #forward
            out, _ = convLSTM(xtrain[:,batch_i*2:(batch_i+1)*2,:,:])

            loss = criterion(out[-1,:,:,:,:], ytrain[batch_i*2:(batch_i+1)*2,:,:,:])
            # backward

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if it%2 == 0:
            print('epoch[{}/{}], loss:{:.6f}' .format(it+1, 100, loss.data.item()))


    # input_channels = 3
    # hidden_channels = 5
    # shape = (28, 28)
    #
    # convLSTM = ConvLSTMCell(input_channels, hidden_channels, shape)
    # criterion = nn.MSELoss()
    # optimizer = torch.optim.SGD(convLSTM.parameters(), lr = 1e-3)
    #
    #
    # xtrain = variable(torch.randn(100, 3, 28, 28))
    # ytrain = variable(torch.randn(100, 5, 28, 28))
    #
    # for it in range(1000):
    #     for batch_i in range(0, int(100/2)):
    #         #forward
    #         hidden_state = [variable(torch.zeros(2, 5, 28, 28)), variable(torch.zeros(2, 5, 28, 28))]
    #         out, _ = convLSTM(xtrain[batch_i*2:(batch_i+1)*2,:,:,:], hidden_state)
    #         loss = criterion(out, ytrain[batch_i*2:(batch_i+1)*2,:,:,:])
    #         # backward
    #
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #     if it%10 == 0:
    #         print('epoch[{}/{}], loss:{:.6f}' .format(it+1, 100, loss.data.item()))
