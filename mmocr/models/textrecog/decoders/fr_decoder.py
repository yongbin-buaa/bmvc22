import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import mmocr.utils as utils
from mmocr.models.builder import DECODERS
from .base_decoder import BaseDecoder


def check_inf(t):
    return torch.sum(torch.isinf(t)).item() + torch.sum(torch.isnan(t)).item()


@DECODERS.register_module()
class ParallelFRDecoder(BaseDecoder):

    def __init__(self,
                 num_classes=37,
                 enc_bi_rnn=False,
                 dec_bi_rnn=False,
                 dec_do_rnn=0.0,
                 dec_gru=False,
                 d_model=512,
                 d_enc=512,
                 d_k=64,
                 pred_dropout=0.0,
                 max_seq_len=40,
                 mask=True,
                 start_idx=0,
                 padding_idx=92,
                 pred_concat=False,
                 **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.enc_bi_rnn = enc_bi_rnn
        self.d_k = d_k
        self.start_idx = start_idx
        self.max_seq_len = max_seq_len
        self.mask = mask
        self.pred_concat = pred_concat

        encoder_rnn_out_size = d_enc * (int(enc_bi_rnn) + 1)
        decoder_rnn_out_size = encoder_rnn_out_size * (int(dec_bi_rnn) + 1)
        # 2D attention layer
        self.conv1x1_1 = nn.Linear(decoder_rnn_out_size, d_k)
        self.conv3x3_1 = nn.Conv2d(
            d_model, d_k, kernel_size=3, stride=1, padding=1)
        self.conv1x1_2 = nn.Linear(d_k, 1)

        # Decoder RNN layer
        kwargs = dict(
            input_size=encoder_rnn_out_size,
            hidden_size=encoder_rnn_out_size,
            num_layers=2,
            batch_first=True,
            dropout=dec_do_rnn,
            bidirectional=dec_bi_rnn)
        if dec_gru:
            self.rnn_decoder = nn.GRU(**kwargs)
        else:
            self.rnn_decoder = nn.LSTM(**kwargs)

        # Decoder input embedding
        self.embedding = nn.Embedding(
            self.num_classes, encoder_rnn_out_size, padding_idx=padding_idx)

        # Prediction layer
        self.pred_dropout = nn.Dropout(pred_dropout)
        pred_num_classes = num_classes - 1  # ignore padding_idx in prediction
        if pred_concat:
            fc_in_channel = decoder_rnn_out_size + d_model + d_enc
        else:
            fc_in_channel = d_model
        self.prediction = nn.Linear(fc_in_channel, pred_num_classes)

        self.gaussian_params_w = nn.Parameter(torch.randn(d_model * 2, 4), requires_grad=True)
        self.gaussian_params_b = nn.Parameter(torch.randn(4), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.gaussian_params_w)
        self.gaussian_params_b.data.fill_(0.01)

    def _2d_attention(self,
                      decoder_input,
                      feat,
                      holistic_feat,
                      valid_ratios=None):
        y = self.rnn_decoder(decoder_input)[0]
        # y: bsz * (seq_len + 1) * hidden_size

        attn_query = self.conv1x1_1(y)  # bsz * (seq_len + 1) * attn_size
        bsz, seq_len, attn_size = attn_query.size()
        attn_query = attn_query.view(bsz, seq_len, attn_size, 1, 1)

        attn_key = self.conv3x3_1(feat)
        # bsz * attn_size * h * w
        attn_key = attn_key.unsqueeze(1)
        # bsz * 1 * attn_size * h * w

        attn_weight = torch.tanh(torch.add(attn_key, attn_query, alpha=1))
        # bsz * (seq_len + 1) * attn_size * h * w
        attn_weight = attn_weight.permute(0, 1, 3, 4, 2).contiguous()
        # bsz * (seq_len + 1) * h * w * attn_size
        attn_weight = self.conv1x1_2(attn_weight)
        # bsz * (seq_len + 1) * h * w * 1
        attn_logits = attn_weight
        bsz, T, h, w, c = attn_weight.size()
        assert c == 1

        if valid_ratios is not None:
            # cal mask of attention weight
            attn_mask = torch.zeros_like(attn_weight)
            for i, valid_ratio in enumerate(valid_ratios):
                valid_width = min(w, math.ceil(w * valid_ratio))
                attn_mask[i, :, :, valid_width:, :] = 1
            attn_weight = attn_weight.masked_fill(attn_mask.bool(),
                                                  float('-inf'))

        attn_weight = attn_weight.view(bsz, T, -1)
        attn_weight = F.softmax(attn_weight, dim=-1)
        attn_weight = attn_weight.view(bsz, T, h, w,
                                       c).permute(0, 1, 4, 2, 3).contiguous()

        attn_feat = torch.sum(
            torch.mul(feat.unsqueeze(1), attn_weight), (3, 4), keepdim=False)
        # bsz * (seq_len + 1) * C

        params = torch.add(
            torch.matmul(
                torch.cat((y, attn_feat), 2),
                self.gaussian_params_w
            ),
            self.gaussian_params_b
        ) # shape: bsz * T * 4
        params = torch.sigmoid(params)

        # a = (x2 - x1, y2 - y1) b = (x3 - x2, y3 - y2)

        mu_x, mu_y, sigma_x, sigma_y = torch.split(params, 1, dim=2)
        mu_x = mu_x * w # shape: bsz * T * 1
        mu_y = mu_y * h # shape: bsz * T * 1
        sigma_x = sigma_x * ((0.5 * w) ** 2) # shape: bsz * T * 1
        sigma_y = sigma_y * ((0.5 * h) ** 2) # shape: bsz * T * 1

        mu_vector_x = mu_x.squeeze(dim=-1)
        mu_vector_y = mu_y.squeeze(dim=-1)
        mu_vector_x = mu_vector_x[:, 1:] - mu_vector_x[:, :-1]
        mu_vector_y = mu_vector_y[:, 1:] - mu_vector_y[:, :-1]
        eps = 1e-5
        modulus = torch.sqrt(mu_vector_x ** 2 + mu_vector_y ** 2 + eps)
        mu_vector_x = mu_vector_x / (modulus + eps)
        mu_vector_y = mu_vector_y / (modulus + eps)

        v_sin = mu_vector_x[:, :-1] * mu_vector_y[:, 1:] - \
            mu_vector_y[:, :-1] * mu_vector_x[:, 1:]
        v_cos = mu_vector_x[:, :-1] * mu_vector_x[:, 1:] + \
            mu_vector_y[:, :-1] * mu_vector_y[:, 1:]
        angle = torch.acos(v_cos * (1 - eps)) * torch.sign(v_sin)
        # sin_var = torch.var(angle, dim=1)

        mu_x, mu_y, sigma_x, sigma_y = mu_x.tile(1, 1, w), mu_y.tile(1, 1, h), sigma_x.tile(1, 1, w), sigma_y.tile(1, 1, h)
        # shape: mu_x, sigma_x [bsz * T * w], mu_y, sigma_y [bsz * T * h]

        coord_x = torch.arange(0, w).view(1, 1, w).tile(bsz, T, 1).to(mu_x.device) # bsz * T * w
        coord_y = torch.arange(0, h).view(1, 1, h).tile(bsz, T, 1).to(mu_y.device) # bsz * T * h
        gaussian_x = torch.exp(-1.0 * torch.pow(coord_x - mu_x, 2) / (2.0 * sigma_x + eps))
        gaussian_y = torch.exp(-1.0 * torch.pow(coord_y - mu_y, 2) / (2.0 * sigma_y + eps))
        gaussian_2d = torch.matmul(
            gaussian_y.view(bsz, T, h, 1), gaussian_x.view(bsz, T, 1, w)
        ) # bsz * T * h * w
        attn_logits = attn_logits.view(bsz, T, h, w)
        mask_attn_logits = gaussian_2d * attn_logits
        mask_attn_weight = F.softmax(mask_attn_logits.view(bsz, T, h * w), dim=-1).view(bsz, T, 1, h, w)
        attn_feat_r = torch.sum(
            torch.mul(feat.unsqueeze(1), mask_attn_weight), (3, 4), keepdim=False)
        
        fused_attn_feat = attn_feat + attn_feat_r

        # linear transformation
        if self.pred_concat:
            hf_c = holistic_feat.size(-1)
            holistic_feat = holistic_feat.expand(bsz, seq_len, hf_c)
            y = self.prediction(torch.cat((y, fused_attn_feat, holistic_feat), 2))
        else:
            y = self.prediction(fused_attn_feat)
        # bsz * (seq_len + 1) * num_classes
        if self.train_mode:
            y = self.pred_dropout(y)

        return y, angle

    def forward_train(self, feat, out_enc, targets_dict, img_metas):
        if img_metas is not None:
            assert utils.is_type_list(img_metas, dict)
            assert len(img_metas) == feat.size(0)

        valid_ratios = None
        if img_metas is not None:
            valid_ratios = [
                img_meta.get('valid_ratio', 1.0) for img_meta in img_metas
            ] if self.mask else None

        targets = targets_dict['padded_targets'].to(feat.device)
        tgt_embedding = self.embedding(targets)
        # bsz * seq_len * emb_dim
        out_enc = out_enc.unsqueeze(1)
        # bsz * 1 * emb_dim
        in_dec = torch.cat((out_enc, tgt_embedding), dim=1)
        # bsz * (seq_len + 1) * C
        out_dec, angle = self._2d_attention(
            in_dec, feat, out_enc, valid_ratios=valid_ratios)
        # bsz * (seq_len + 1) * num_classes

        angle_var = 0
        cnt = 0
        for idx, img_meta in enumerate(img_metas):
            l = min(len(img_meta['text']), self.max_seq_len - 1)
            if l > 3:
                angle_var = torch.var(angle[idx, 1:l-1]) + angle_var
                cnt += 1.0

        if cnt == 0:
            angle_var = 0
        else:
            angle_var = angle_var / cnt

        return out_dec[:, 1:, :], angle_var  # bsz * seq_len * num_classes

    def forward_test(self, feat, out_enc, img_metas):
        if img_metas is not None:
            assert utils.is_type_list(img_metas, dict)
            assert len(img_metas) == feat.size(0)

        valid_ratios = None
        if img_metas is not None:
            valid_ratios = [
                img_meta.get('valid_ratio', 1.0) for img_meta in img_metas
            ] if self.mask else None

        seq_len = self.max_seq_len

        bsz = feat.size(0)
        start_token = torch.full((bsz, ),
                                 self.start_idx,
                                 device=feat.device,
                                 dtype=torch.long)
        # bsz
        start_token = self.embedding(start_token)
        # bsz * emb_dim
        start_token = start_token.unsqueeze(1).expand(-1, seq_len, -1)
        # bsz * seq_len * emb_dim
        out_enc = out_enc.unsqueeze(1)
        # bsz * 1 * emb_dim
        decoder_input = torch.cat((out_enc, start_token), dim=1)
        # bsz * (seq_len + 1) * emb_dim

        outputs = []
        for i in range(1, seq_len + 1):
            decoder_output, _ = self._2d_attention(
                decoder_input, feat, out_enc, valid_ratios=valid_ratios)
            char_output = decoder_output[:, i, :]  # bsz * num_classes
            char_output = F.softmax(char_output, -1)
            outputs.append(char_output)
            _, max_idx = torch.max(char_output, dim=1, keepdim=False)
            char_embedding = self.embedding(max_idx)  # bsz * emb_dim
            if i < seq_len:
                decoder_input[:, i + 1, :] = char_embedding

        outputs = torch.stack(outputs, 1)  # bsz * seq_len * num_classes

        return outputs
