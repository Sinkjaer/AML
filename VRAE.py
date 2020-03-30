import numpy as np
import torch
from torch import nn, optim
from torch import distributions
from .base import BaseEstimator
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os

class Encoder(nn.Module):
    """
    Encoder network containing enrolled LSTM/GRU
    :param number_of_features: number of input features = 1 (n timesteps)
    :param hidden_size: hidden size of the RNN = 128 
    :param hidden_layer_depth: number of layers in RNN = 1 (1 hidden layer)
    :param latent_length: latent vector length = 5 
    """
    def __init__(self, number_of_features, hidden_size, hidden_layer_depth, batch_size, latent_length, dropout):

        super(Encoder, self).__init__()

        self.number_of_features = number_of_features
        self.hidden_size = hidden_size
        self.latent_length = latent_length
        self.batch_size = batch_size


        self.model = nn.LSTM(input_size=self.number_of_features, 
                             hidden_size=self.hidden_size,
                             num_layers=1, 
                             bidirectional=True)

    def forward(self, x):
        """Forward propagation of encoder. Given input, outputs the last hidden state of encoder
        :param x: input to the encoder, of shape (sequence_length, batch_size, number_of_features)
        :return: last hidden state of encoder, of shape (batch_size, hidden_size)
        """

        _, h_end, _ = self.model(x)

        return h_end.reshape((self.batch_size, 2*self.hidden_size))


class Lambda(nn.Module): 
    """Lambda module converts output of encoder to latent vector
    :param hidden_size: hidden size of the encoder
    :param latent_length: latent vector length
    """
    def __init__(self, hidden_size, latent_length):
        super(Lambda, self).__init__()

        self.hidden_size = hidden_size
        self.latent_length = latent_length

        self.hidden_to_mean = nn.Linear(2*self.hidden_size, self.latent_length)
        self.hidden_to_var_int = nn.Linear(2*self.hidden_size, self.latent_length)
        self.hidden_to_var = nn.Softplus()


    def forward(self, cell_output):
        """Given last hidden state of encoder, passes through a linear layer, and finds the mean and variance
        :param cell_output: last hidden state of encoder
        :return: latent vector
        """

        self.latent_mean = self.hidden_to_mean(cell_output)
        self.latent_var_int = self.hidden_to_logvar_int(cell_output)
        self.latent_var = self.hidden_to_var(self.latent_var_int)

        if self.training:
            std = torch.sqrt(self.latent_var)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(self.latent_mean), self.latent_mean, std
        else:
            return self.latent_mean, std


class Decoder(nn.Module):
    """Converts latent vector into output
    :param sequence_length: length of the input sequence = 140
    :param batch_size: batch size of the input sequence
    :param hidden_size: hidden size of the RNN = 128
    :param hidden_layer_depth: number of layers in RNN = 1 
    :param latent_length: latent vector length = 5 
    :param output_size: 2, one representing the mean, other log std dev of the output
    :param dtype: Depending on cuda enabled/disabled, create the tensor
    """
    def __init__(self, sequence_length, batch_size, hidden_size, hidden_layer_depth, latent_length, output_size, dtype, block='LSTM'):

        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.output_size = output_size
        self.dtype = dtype

        self.model = nn.LSTM(latent_length, self.hidden_size, self.hidden_layer_depth)

        self.output_to_mean = nn.Linear(self.hidden_size*latent_length, 1)
        self.output_to_var_int = nn.Linear(self.hidden_size*latent_length, 1)
        self.output_to_var = nn.Softplus()

    def forward(self, latent):
        """Converts latent to hidden to output
        :param latent: latent vector
        :return: outputs consisting of mean and std dev of vector
        """
        latent_reshape = latent.unsqueeze(0)
        decoder_inputs =  latent.repeat((sequence_length,1,1))

        decoder_output, _ , _ = self.model(self.decoder_inputs)

        decoder_output = decoder_output.view((1,0,2))
        
        output_means = self.output_to_mean(decoder_output)
        output_var_int = self.output_to_var_int(decoder_output)
        output_var = self.output_to_var(output_var_int)

        return output_means, output_var
