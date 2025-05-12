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
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
