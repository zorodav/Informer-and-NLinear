import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.masking import TriangularCausalMask, ProbMask
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, ProbAttention, AttentionLayer
from layers.Embed import DataEmbedding,DataEmbedding_wo_pos,DataEmbedding_wo_temp,DataEmbedding_wo_pos_temp
import numpy as np


class Model(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs['pred_len']
        self.output_attention = configs['output_attention']

        # Embedding
        if configs['embed_type'] == 0:
            self.enc_embedding = DataEmbedding(configs['model']['parameters']['enc_in'], configs['model']['parameters']['d_model'], configs['embed'], configs['freq'],
                                            configs['model']['parameters']['dropout'])
            self.dec_embedding = DataEmbedding(configs['model']['parameters']['dec_in'], configs['model']['parameters']['d_model'], configs['embed'], configs['freq'],
                                           configs['model']['parameters']['dropout'])
        elif configs['embed_type'] == 1:
            self.enc_embedding = DataEmbedding(configs['model']['parameters']['enc_in'], configs['model']['parameters']['d_model'], configs['embed'], configs['freq'],
                                                    configs['model']['parameters']['dropout'])
            self.dec_embedding = DataEmbedding(configs['model']['parameters']['dec_in'], configs['model']['parameters']['d_model'], configs['embed'], configs['freq'],
                                                    configs['model']['parameters']['dropout'])
        elif configs['embed_type'] == 2:
            self.enc_embedding = DataEmbedding_wo_pos(configs['model']['parameters']['enc_in'], configs['model']['parameters']['d_model'], configs['embed'], configs['freq'],
                                                    configs['model']['parameters']['dropout'])
            self.dec_embedding = DataEmbedding_wo_pos(configs['dec_in'], configs['d_model'], configs['embed'], configs['freq'],
                                                    configs['model']['parameters']['dropout'])

        elif configs['embed_type'] == 3:
            self.enc_embedding = DataEmbedding_wo_temp(configs['model']['parameters']['enc_in'], configs['model']['parameters']['d_model'], configs['embed'], configs['freq'],
                                                    configs['model']['parameters']['dropout'])
            self.dec_embedding = DataEmbedding_wo_temp(configs['model']['parameters']['dec_in'], configs['model']['parameters']['d_model'], configs['embed'], configs['freq'],
                                                    configs['model']['parameters']['dropout'])
        elif configs['embed_type'] == 4:
            self.enc_embedding = DataEmbedding_wo_pos_temp(configs['model']['parameters']['enc_in'], configs['model']['parameters']['d_model'], configs['embed'], configs['freq'],
                                                    configs['model']['parameters']['dropout'])
            self.dec_embedding = DataEmbedding_wo_pos_temp(configs['model']['parameters']['dec_in'], configs['model']['parameters']['d_model'], configs['embed'], configs['freq'],
                                                    configs['model']['parameters']['dropout'])
        # Encoder
        self.encoder = Encoder(
            [
            EncoderLayer(
                AttentionLayer(
                ProbAttention(False, configs['model']['parameters']['factor'], attention_dropout=configs['model']['parameters']['dropout'],
                          output_attention=configs['output_attention']),
                configs['model']['parameters']['d_model'], configs['model']['parameters']['n_heads']),
                configs['model']['parameters']['d_model'],
                configs['d_ff'],
                dropout=configs['model']['parameters']['dropout'],
                activation=configs['activation']
            ) for l in range(configs['e_layers'])
            ],
            [
            ConvLayer(
                configs['model']['parameters']['d_model']
            ) for l in range(configs['e_layers'] - 1)
            ] if configs['distil'] else None,
            norm_layer=torch.nn.LayerNorm(configs['model']['parameters']['d_model'])
        )
        # Decoder
        self.decoder = Decoder(
            [
            DecoderLayer(
                AttentionLayer(
                ProbAttention(True, configs['model']['parameters']['factor'], attention_dropout=configs['model']['parameters']['dropout'], output_attention=False),
                configs['model']['parameters']['d_model'], configs['model']['parameters']['n_heads']),
                AttentionLayer(
                ProbAttention(False, configs['model']['parameters']['factor'], attention_dropout=configs['model']['parameters']['dropout'], output_attention=False),
                configs['model']['parameters']['d_model'], configs['model']['parameters']['n_heads']),
                configs['model']['parameters']['d_model'],
                configs['d_ff'],
                dropout=configs['model']['parameters']['dropout'],
                activation=configs['activation'],
            )
            for l in range(configs['model']['parameters']['d_layers'])
            ],
            norm_layer=torch.nn.LayerNorm(configs['model']['parameters']['d_model']),
            projection=nn.Linear(configs['model']['parameters']['d_model'], configs['model']['parameters']['c_out'], bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, autoregressive=False):
        
        if autoregressive:
            return self.autoregressive_forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
        else:
            return self.teacher_forcing_forward(x_enc, x_mark_enc, x_dec, x_mark_dec, 
                                               enc_self_mask, dec_self_mask, dec_enc_mask)

    def teacher_forcing_forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                               enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """Normal teacher forcing forward pass for training"""
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]

    def autoregressive_forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """Autoregressive forward pass for testing"""
        batch_size = x_enc.size(0)
        device = x_enc.device
        
        # Encode the input sequence once
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        
        # Initialize decoder input with only the label portion
        # x_dec should only contain the label portion when autoregressive=True
        dec_input = x_dec.clone()  # Shape: [batch_size, label_len, features]
        
        # We need to extend the time marks for the prediction steps
        label_len = x_dec.size(1)
        
        # Generate predictions step by step
        predictions = []
        
        for step in range(self.pred_len):
            # Current time marks (extend as needed)
            current_len = dec_input.size(1)
            if current_len <= x_mark_dec.size(1):
                current_mark = x_mark_dec[:, :current_len, :]
            else:
                # Extend time marks by repeating the last time step pattern
                # This is a simple extension - you might want more sophisticated time encoding
                last_mark = x_mark_dec[:, -1:, :].repeat(1, current_len - x_mark_dec.size(1), 1)
                current_mark = torch.cat([x_mark_dec, last_mark], dim=1)
            
            # Forward pass through decoder
            dec_out = self.dec_embedding(dec_input, current_mark)
            dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
            
            # Get the prediction for the next time step (last position)
            next_pred = dec_out[:, -1:, :]  # Shape: [batch_size, 1, features]
            predictions.append(next_pred)
            
            # Append prediction to decoder input for next iteration
            dec_input = torch.cat([dec_input, next_pred], dim=1)
        
        # Concatenate all predictions
        final_predictions = torch.cat(predictions, dim=1)  # [batch_size, pred_len, features]
        
        if self.output_attention:
            return final_predictions, None  # No attention weights in autoregressive mode
        else:
            return final_predictions