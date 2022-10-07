import torch.nn as nn

from mmocr.models.builder import DECODERS
from .base_decoder import BaseDecoder
import torch.nn as nn
import torch.nn.functional as F
import torch


class BiLSTM_Nets(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiLSTM_Nets, self).__init__()
        self.rnn = nn.LSTM(
            input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x seq_len x input_size]
        output : contextual feature [batch_size x seq_len x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output


class AttentionCell(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_embeddings
    ):
        super(AttentionCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)  # either i2i or h2h should have bias
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.LSTMCell(input_size + num_embeddings, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, prev_hidden, prev_c, batch_H, char_onehots):
        # [batch_size x num_encoder_step x num_channel] -> [batch_size x num_encoder_step x hidden_size]
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = self.h2h(prev_hidden).unsqueeze(1)
        e = self.score(torch.tanh(batch_H_proj + prev_hidden_proj))  # batch_size x num_encoder_step x 1

        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.permute(0, 2, 1), batch_H).squeeze(1)  # batch_size x num_channel
        concat_context = torch.cat([context, char_onehots], 1)  # batch_size x (num_channel + num_embedding)
        hx, cx = self.rnn(concat_context, (prev_hidden, prev_c))
        return hx, cx, alpha


@DECODERS.register_module()
class Attention1D(BaseDecoder):
    def __init__(
        self,
        input_size=512,
        hidden_size=512,
        num_classes=90,
        max_seq_len=40,
        start_idx=0,
        rnn_flag=True,
        **kwargs
    ):
        super(Attention1D, self).__init__()
        self.attention_cell = AttentionCell(
            input_size, hidden_size, num_classes
        )
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.generator = nn.Linear(hidden_size, num_classes)
        self.seq_len = max_seq_len
        self.rnn_flag = rnn_flag
        self.start_idx = start_idx

        if self.rnn_flag:
            self.sequence_model = nn.Sequential(
                BiLSTM_Nets(
                    input_size,
                    hidden_size,
                    hidden_size
                ),
                BiLSTM_Nets(
                    hidden_size,
                    hidden_size,
                    hidden_size
                )
            )

    def init_weights(self):
        pass

    def _char_to_onehot(self, input_char, onehot_dim=38):
        input_char = input_char.unsqueeze(1)
        batch_size = input_char.size(0)
        one_hot = torch.FloatTensor(
            batch_size, onehot_dim
        ).zero_().to(input_char.device)
        one_hot = one_hot.scatter_(1, input_char, 1)
        return one_hot

    def forward_train(self, feat, out_enc, targets_dict, img_metas):
        assert feat.size(2) == 1, 'feature height must be 1'
        feat = feat.squeeze(2).permute(0, 2, 1)
        if self.rnn_flag:
            contextual_feature = self.sequence_model(feat).contiguous()
        else:
            contextual_feature = feat.contiguous()

        text = targets_dict['padded_targets'].to(feat.device)

        batch_size = contextual_feature.size(0)
        # num_steps = self.seq_len + 1  # +1 for [s] at end of sentence.
        num_steps = self.seq_len

        output_hiddens = torch.zeros(
            [batch_size, num_steps, self.hidden_size],
            dtype=torch.float
        ).to(contextual_feature.device)
        hidden, c = (
            torch.zeros(
                [batch_size, self.hidden_size],
                dtype=torch.float
            ).to(contextual_feature.device),
            torch.zeros(
                [batch_size, self.hidden_size],
                dtype=torch.float
            ).to(contextual_feature.device)
        )

        for i in range(num_steps):
            # one-hot vectors for a i-th char. in a batch
            char_onehots = self._char_to_onehot(
                text[:, i], onehot_dim=self.num_classes
            ).to(contextual_feature.device)
            # hidden : decoder's hidden s_{t-1}, batch_H : encoder's hidden H, char_onehots : one-hot(y_{t-1})
            hidden, c, alpha = self.attention_cell(
                hidden, c, contextual_feature, char_onehots
            )
            output_hiddens[:, i, :] = hidden
        probs = self.generator(output_hiddens)
        return probs

    def forward_test(self, feat, out_enc, img_metas):
        assert feat.size(2) == 1, 'feature height must be 1'
        feat = feat.squeeze(2).permute(0, 2, 1)
        if self.rnn_flag:
            contextual_feature = self.sequence_model(feat).contiguous()
        else:
            contextual_feature = feat.contiguous()

        batch_size = contextual_feature.size(0)
        # num_steps = self.seq_len + 1  # +1 for [s] at end of sentence.
        num_steps = self.seq_len

        hidden, c = (
            torch.zeros(
                [batch_size, self.hidden_size],
                dtype=torch.float
            ).to(contextual_feature.device),
            torch.zeros(
                [batch_size, self.hidden_size],
                dtype=torch.float
            ).to(contextual_feature.device)
        )

        # targets = torch.zeros(
        #     [batch_size], dtype=torch.long
        # ).to(contextual_feature.device)  # [GO] token
        targets = torch.full(
            [batch_size], self.start_idx, dtype=torch.long
        )
        probs = torch.zeros(
            [batch_size, num_steps, self.num_classes],
            dtype=torch.float
        ).to(contextual_feature.device)

        for i in range(num_steps):
            char_onehots = self._char_to_onehot(
                targets, onehot_dim=self.num_classes
            ).to(contextual_feature.device)
            hidden, c, alpha = self.attention_cell(
                hidden, c, contextual_feature, char_onehots
            )
            probs_step = self.generator(hidden)
            probs[:, i, :] = probs_step
            _, next_input = probs_step.max(1)
            targets = next_input

        return probs  # batch_size x num_steps x num_classes
