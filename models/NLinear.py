import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs['seq_len']
        self.pred_len = configs['pred_len']
        
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = configs['model']['parameters']['enc_in']
        self.individual = configs['individual']
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x, autoregressive=False):
        if autoregressive:
            return self.autoregressive_forward(x)
        else:
            return self.teacher_forcing_forward(x)

    def teacher_forcing_forward(self, x):
        """Normal forward pass for training - predicts all steps at once"""
        # x: [Batch, Input length, Channel]
        seq_last = x[:,-1:,:].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
        else:
            x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        x = x + seq_last
        return x # [Batch, Output length, Channel]

    def autoregressive_forward(self, x):
        """Autoregressive forward pass for testing - predicts one step at a time"""
        batch_size, seq_len, channels = x.size()
        device = x.device
        
        # Initialize with input sequence
        current_seq = x.clone()  # [Batch, seq_len, Channel]
        predictions = []
        
        for step in range(self.pred_len):
            # Use the last seq_len points to predict the next point
            input_seq = current_seq[:, -self.seq_len:, :]  # [Batch, seq_len, Channel]
            
            # Normalize
            seq_last = input_seq[:, -1:, :].detach()
            normalized_seq = input_seq - seq_last
            
            if self.individual:
                # Predict one step ahead for each channel
                next_step = torch.zeros([batch_size, 1, channels], dtype=x.dtype).to(device)
                for i in range(self.channels):
                    # For autoregressive, we need to predict just one step
                    # We'll use only the first output of the linear layer
                    full_pred = self.Linear[i](normalized_seq[:, :, i])  # [Batch, pred_len]
                    next_step[:, 0, i] = full_pred[:, 0]  # Take only the first prediction
            else:
                # Predict one step ahead
                normalized_seq_transposed = normalized_seq.permute(0, 2, 1)  # [Batch, Channel, seq_len]
                full_pred = self.Linear(normalized_seq_transposed)  # [Batch, Channel, pred_len]
                full_pred = full_pred.permute(0, 2, 1)  # [Batch, pred_len, Channel]
                next_step = full_pred[:, 0:1, :]  # Take only the first prediction [Batch, 1, Channel]
            
            # Denormalize
            next_step = next_step + seq_last
            predictions.append(next_step)
            
            # Update current sequence by appending the prediction
            current_seq = torch.cat([current_seq, next_step], dim=1)
        
        # Concatenate all predictions
        final_predictions = torch.cat(predictions, dim=1)  # [Batch, pred_len, Channel]
        return final_predictions