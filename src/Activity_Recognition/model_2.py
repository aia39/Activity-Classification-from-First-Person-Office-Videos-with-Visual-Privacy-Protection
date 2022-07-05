import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable
from torchvision.models import  densenet161,resnet152,resnext101_32x8d

import torch.utils.model_zoo as model_zoo
import os
import sys

#The name of the model is
#ConvLSTM2_0for_densenet161_with_image_size_512_batch_size_5_best_74.87(full_data).pth

##############################
#         Encoder
##############################


class Encoder2(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder2, self).__init__()
        resnet = densenet161(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.final = nn.Sequential(
            nn.Linear(565248, latent_dim), nn.BatchNorm1d(latent_dim, momentum=0.01)
        )
    

    def forward(self, x):
        
        with torch.no_grad():
            x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        #print(self.final(x))
        
        return self.final(x)


##############################
#           LSTM
##############################


class LSTM2(nn.Module):
    def __init__(self, latent_dim, num_layers, hidden_dim, bidirectional):
        super(LSTM2, self).__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        self.hidden_state = None

    def reset_hidden_state(self):
        self.hidden_state = None

    def forward(self, x):
        x, self.hidden_state = self.lstm(x, self.hidden_state)
        return x


##############################
#      Attention Module
##############################


class Attention(nn.Module):
    def __init__(self, latent_dim, hidden_dim, attention_dim):
        super(Attention, self).__init__()
        self.latent_attention = nn.Linear(latent_dim, attention_dim)
        self.hidden_attention = nn.Linear(hidden_dim, attention_dim)
        self.joint_attention = nn.Linear(attention_dim, 1)

    def forward(self, latent_repr, hidden_repr):
        if hidden_repr is None:
            hidden_repr = [
                Variable(
                    torch.zeros(latent_repr.size(0), 1, self.hidden_attention.in_features), requires_grad=False
                ).float()
            ]
        h_t = hidden_repr[0]
        latent_att  = self.latent_attention(latent_att)
        hidden_att  = self.hidden_attention(h_t)
        joint_att   = self.joint_attention(F.relu(latent_att + hidden_att)).squeeze(-1)
        attention_w = F.softmax(joint_att, dim=-1)
        return attention_w


##############################
#         ConvLSTM
##############################

#dim=-1 is the right most dimension

class ConvLSTM2(nn.Module):
    def __init__(
        self, num_classes, latent_dim=512, lstm_layers=1, hidden_dim=1024, bidirectional=True, attention=True
    ):
        super(ConvLSTM2, self).__init__()
        self.encoder = Encoder2(latent_dim)
        #self.inceptionresnetV2 = InceptionResNetV2()
        self.lstm = LSTM2(latent_dim, lstm_layers, hidden_dim, bidirectional)
        self.output_layers = nn.Sequential(
            nn.Linear(2* hidden_dim if bidirectional else hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=0.01),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
            nn.Softmax(dim=-1),
        )
        self.attention = attention
        self.attention_layer = nn.Linear(2 * hidden_dim if bidirectional else hidden_dim, 1)

    def forward(self, x):
        batch_size,seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        x = self.encoder(x)
        #x = self.inceptionresnetV2(x)
        x = x.view(batch_size, seq_length, -1)
        x = self.lstm(x)
       
        if self.attention:
            attention_w = F.softmax(self.attention_layer(x).squeeze(-1) , dim=-1)
            x = torch.sum(attention_w.unsqueeze(-1) * x, dim=1)
        else:
            x = x[:, -1]
        return self.output_layers(x)