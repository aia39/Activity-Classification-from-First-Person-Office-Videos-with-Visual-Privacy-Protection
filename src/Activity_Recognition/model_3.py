import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable
from wide_resnet import  wide_resnet101_2

import torch.utils.model_zoo as model_zoo
import os
import sys

#the name of the model is
#ConvLSTM5_13for_wide_resnet101_with_added_attention_image_size_324_best_75.39(full_data).pth

##############################
#         Encoder
##############################


class Encoder3(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder3, self).__init__()
        resnet = wide_resnet101_2(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.final = nn.Sequential(
            nn.Linear(resnet.fc.in_features, latent_dim), nn.BatchNorm1d(latent_dim, momentum=0.01)
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


class LSTM3(nn.Module):
    def __init__(self, latent_dim, num_layers, hidden_dim, bidirectional):
        super(LSTM3, self).__init__()
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


class Attention3(nn.Module):
    def __init__(self,sequence_length, dimention):
        super(Attention3, self).__init__()
        self.attention = nn.Linear(dimention, 1)
        self.sequence_length = sequence_length
        

    def forward(self, x):
        attention_w = F.softmax(self.attention(x).squeeze(-1) , dim=-1)
        # print("\nthe shape of attention_w inside attention model is:\n")
        # print(attention_w.shape)
        # print("\nthe shape of x inside attention model is:\n")
        # print(x.shape)
        # print("\n\n")
        # for i in range(40):
        #         x[i] = x[i] * attention_w[i]
        attention_w = attention_w.view(self.sequence_length,1)
        x = x*attention_w
        #print("\n\n")
        return x


##############################
#         ConvLSTM
##############################

#dim=-1 is the right most dimension

class ConvLSTM3(nn.Module):
    def __init__(
        self, num_classes, latent_dim=512, lstm_layers=1, hidden_dim=1024, bidirectional=True, attention=True,batch_size=18,
    sequence_length=40):
        super(ConvLSTM3, self).__init__()
        self.encoder = Encoder3(latent_dim)
        #self.inceptionresnetV2 = InceptionResNetV2()
        self.lstm = LSTM3(latent_dim, lstm_layers, hidden_dim, bidirectional)
        self.output_layers = nn.Sequential(
            nn.Linear(2* hidden_dim if bidirectional else hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=0.01),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
            nn.Softmax(dim=-1),
        ) 
        self.attention = attention
        self.attention_layer = nn.Linear(2 * hidden_dim if bidirectional else hidden_dim, 1)
        self.attention_layer2 = Attention3(sequence_length,latent_dim)

    def forward(self, x):
        batch_size,seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        x = self.encoder(x)
        # print("\nafter encoder x shape is\n")
        # print(x.shape)
        # print("\n\n")
        
        x = x.view(batch_size, seq_length, -1)
        # print("\nafter encoder x shape is\n")
        # print(x.shape)
        # print("\n\nThe current batch_size is :%s" %(batch_size))

        for i in range(batch_size):
            if i==0:
                per_batch_attention=self.attention_layer2(x[i])
                #print(str(i) + " bar dhuksee" + str(per_batch_attention.shape))
            else:
                batch_attention = self.attention_layer2(x[i])
                #batch_attention = batch_attention.view(1, -1)
                per_batch_attention=torch.cat((per_batch_attention, batch_attention),0)
                #print(str(i) + " bar dhuksee")
        # print("\nafter first attention per_batch_attention shape is\n")
        # print(per_batch_attention.shape)
        # print("\n\n")

        x = per_batch_attention.view(batch_size,seq_length,-1)
        
            
        x = x.view(batch_size , seq_length, -1)    
        # print("\nafter first attention x shape is\n")
        # print(x.shape)
        # print("\n\n")
        x = self.lstm(x)
        # print("\nafter LSTM x shape is\n")
        # print(x.shape)
        # print("\n\n")
       
        if self.attention:
            attention_w = F.softmax(self.attention_layer(x).squeeze(-1) , dim=-1)
            x = torch.sum(attention_w.unsqueeze(-1) * x, dim=1)
        else:
            x = x[:, -1]
        return self.output_layers(x)

