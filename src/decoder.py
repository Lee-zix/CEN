import random

from torch.nn import functional as F
import torch
from torch.nn.parameter import Parameter
import math
import os
import numpy as np
path_dir = os.getcwd()



class ConvTransE(torch.nn.Module):
    def __init__(self, num_entities, embedding_dim, input_dropout=0, hidden_dropout=0, feature_map_dropout=0, channels=50, kernel_size=3, sequence_len = 10, use_bias=True):

        super(ConvTransE, self).__init__()
        # 初始化relation embeddings
        # self.emb_rel = torch.nn.Embedding(num_relations, embedding_dim, padding_idx=0)

        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.hidden_drop = torch.nn.Dropout(hidden_dropout)
        self.feature_map_drop = torch.nn.Dropout(feature_map_dropout)
        # self.loss = torch.nn.BCELoss()
        self.embedding_dim = embedding_dim
        self.conv_list = torch.nn.ModuleList()
        self.bn0_list = torch.nn.ModuleList()
        self.bn1_list = torch.nn.ModuleList()
        self.bn2_list = torch.nn.ModuleList()
        for _ in range(sequence_len):
            self.conv_list.append(torch.nn.Conv1d(2, channels, kernel_size, stride=1, padding=int(math.floor(kernel_size / 2)))  ) # kernel size is odd, then padding = math.floor(kernel_size/2))
            self.bn0_list.append(torch.nn.BatchNorm1d(2))
            self.bn1_list.append( torch.nn.BatchNorm1d(channels))
            self.bn2_list.append(torch.nn.BatchNorm1d(embedding_dim)) 

        self.fc = torch.nn.Linear(embedding_dim * channels, embedding_dim)

    def forward(self, embedding, emb_rel, triplets, nodes_id=None, mode="train", negative_rate=0, partial_embeding=None):
        score_list = []
        batch_size = len(triplets)
        for idx in range(len(embedding)):
            e1_embedded_all = F.tanh(embedding[idx])
            e1_embedded = e1_embedded_all[triplets[:, 0]].unsqueeze(1)
            rel_embedded = emb_rel[triplets[:, 1]].unsqueeze(1)
            stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)
            stacked_inputs = self.bn0_list[idx](stacked_inputs)
            x = self.inp_drop(stacked_inputs)
            x = self.conv_list[idx](x)
            x = self.bn1_list[idx](x)
            x = F.relu(x)
            x = self.feature_map_drop(x)
            x = x.view(batch_size, -1)
            x = self.fc(x)
            x = self.hidden_drop(x)
            if batch_size > 1:
                x = self.bn2_list[idx](x)
            x = F.relu(x)
            if partial_embeding is None:
                x = torch.mm(x, e1_embedded_all.transpose(1, 0))
            else:
                x = torch.mm(x, partial_embeding.transpose(1, 0))
            score_list.append(x)
        # print(score_list)
        # print("--------------------")
        return score_list

  