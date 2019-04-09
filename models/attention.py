import torch
import torch.nn as nn
import torch.nn.functional as F

class MatrixAttn(nn.Module):
    def __init__(self, linin, linout):
        super().__init__()
        self.attnlin = nn.Linear(linin, linout)

    def forward(self, dec, emb):
        emb, elen = emb
        emask = torch.arange(0 ,emb.size(1)).unsqueeze(0).repeat(emb.size(0),1).long().cuda()
        emask = (emask >= elen.unsqueeze(1)).unsqueeze(1)
        decsmall = self.attnlin(dec)
        uunorm = torch.bmm(decsmall, emb.transpose(1,2))
        uunorm.masked_fill_(emask, -float('inf'))
        attn = F.softmax(uunorm, dim=2)
        out = torch.bmm(attn, emb)
        return  out, attn

class BahdanauAttention(nn.Module):
    def __init__(self, num_units, query_size, memory_size):
        super(BahdanauAttention, self).__init__()
        self._num_units = num_units
        self._softmax = nn.Softmax()
        self.query_layer = nn.Linear(query_size, num_units, bias=False)
        self.memory_layer = nn.Linear(memory_size, num_units, bias=False)
        self.alignment_layer = nn.Linear(num_units, 1, bias=False)

    def _score(self, query, keys):
        processed_query = self.query_layer(query)
        values = self.memory_layer(keys) # B * s * num_units

        expanded_query = processed_query.unsqueeze(1) #[B, 1, num_units]
        alignment = self.alignment_layer(F.tanh(expanded_query + values)) # [B, s, 1]
        return alignment.squeeze(2) #[B, s]

    def forward(self, query, keys):
        alignment_score = self._score(query, keys)
        weight = F.softmax(alignment_score, dim=-1)
        context = weight.unsqueeze(2) * keys #[B , s, 1], [B, s, num_unit] -> [B, s, num_unit]
        total_context = context.sum(1) #[B, num_unit]
        return total_context, alignment_score

class LuongAttention(nn.Module):
    _SCORE_FN = {
        'dot' : '_dot_score',
        'general' : "_general_score",
        'concat' : '_concat_score'
    }

    def __init__(self, attention_window_size, num_units, query_size, memory_size, alignment='local', score_fn='dot'):
        super(LuongAttention, self).__init__()
        if score_fn not in self._SCORE_FN.keys():
            raise ValueError()
        self._attention_window_size = attention_window_size
        self._softmax = nn.Softmax()
        self._score_fn = score_fn
        self._alignment = alignment

        self.query_layer = nn.Linear(query_size, num_units, bias=False)
        self.predictive_alignment_layer = nn.Linear(num_units, 1, bias=False)
        self.alignment_layer = nn.Linear(num_units, 1, bias=False)

        if score_fn == 'general':
            self.general_memory_layer = nn.Linear(memory_size, query_size, bias=False)
        elif score_fn == 'concat':
            self.concat_memory_layer1 = nn.Linear(2 * memory_size, num_units, bias=False)
            self.concat_memory_layer2 = nn.Linear(num_units, 1, bias=False)

    def _dot_score(self, query, keys):
        depth = query.size(-1)
        key_units = keys.size(-1)
        if depth != key_units:
            raise ValueError()
        expanded_query = query.unsqueeze(1)
        tkeys = keys.transpose(1,2)
        alignment = torch.matmul(expanded_query, tkeys)
        return alignment.squeeze(1)

    def _general_score(self, query, keys):
        weighted_keys = self.general_memory_layer(keys)
        extended_query = query.unsqueeze(1)
        weighted_keys = weighted_keys.transpose(1,2)
        alignment = torch.matmul(extended_query, weighted_keys)
        return alignment.squeeze(1)

    def _concat_score(self, query, keys):
        expanded_query = query.unsqueeze(1).expand(*keys.size())
        concatenated_hidden = torch.cat([expanded_query, keys], dim=2)
        weighted_concatenated_hidden = self.concat_memory_layer1(concatenated_hidden)
        temp_score = F.tanh(weighted_concatenated_hidden)
        alignment = self.concat_memory_layer2(temp_score)
        return alignment.squeeze(2)

    def forward(self, query, keys, key_lengths):
        score_fn = getattr(self, self._SCORE_FN[self._score_fn])
        alignment_score = score_fn(query, keys)
        weight = F.softmax(alignment_score, dim=-1)
        if self._alignment == 'local':
            extended_key_lengths = key_lengths.unsqueeze(1)
            preprocessed_query = self.query_layer(query)

            activated_query = F.tanh(preprocessed_query)
            sigmoid_query = F.sigmoid(self.predictive_alignment_layer(activated_query))
            predictive_alignment = extended_key_lengths * sigmoid_query

            ai_start = predictive_alignment - self._attention_window_size
            ai_end = predictive_alignment + self._attention_window_size

            std = torch.FloatTensor([self._attention_window_size / 2.])
            alignment_point_dist = (extended_key_lengths - predictive_alignment).pow(2)
            alignment_point_dist = (-(alignment_point_dist / (2 * std[0]))).exp()
            weight = weight * alignment_point_dist

            contexts = []
            for i in range(weight.size(0)):
                start = ai_start[i].int().data.numpy()[0]
                end = ai_end[i].int().data.numpy()[0]

                aligned_weight = weight[i, start:end]
                aligned_keys = keys[i, start:end]

                aligned_context = aligned_weight.unsqueeze(1) * aligned_keys
                contexts.append(aligned_context.sum(0))

            total_context = torch.stack(contexts, 0)
        elif self._alignment == "global":
            context = weight.unsqueeze(2) * keys
            total_context = context.sum(1)

        return total_context, alignment_score

    @property
    def attention_window_size(self):
        return self._attention_window_size

class MultiHeadAttention(nn.Module):
    def __init__(self, query_dim, key_dim, num_units, dropout_p=0.5, h=8, is_masked=False):
        super(MultiHeadAttention, self).__init__()
        if query_dim != key_dim:
            raise ValueError()
        if num_units % h != 0:
            raise ValueError()
        if query_dim != num_units:
            raise ValueError()

        self._num_units = num_units
        self._h = h
        self.key_dim = torch.tensor(key_dim, requires_grad=False).float()
        self._dropout_p = dropout_p
        self._is_masked = is_masked

        self.query_layer = nn.Linear(query_dim, num_units, bias=False)
        self.key_layer = nn.Linear(key_dim, num_units, bias=False)
        self.value_layer = nn.Linear(key_dim, num_units, bias=False)

        self.bn = nn.BatchNorm1d(num_units)
        self.ln = nn.LayerNorm(num_units)

    def forward(self, query, keys, mask=None):
        Q = self.query_layer(query)
        K = self.key_layer(keys)
        V = self.value_layer(keys)

        chunk_size = int(self._num_units / self._h)
        Q = torch.cat(Q.split(split_size=chunk_size, dim=2), dim=0)
        K = torch.cat(K.split(split_size=chunk_size, dim=2), dim=0)
        V = torch.cat(V.split(split_size=chunk_size, dim=2), dim=0)

        # calculate QK^T
        attention = torch.matmul(Q, K.transpose(1, 2))
        # normalize with sqrt(dk)
        attention = attention / torch.sqrt(self._key_dim).cuda()

        if mask is not None:
            mask = mask.repeat(self._h, 1, 1)
            attention.masked_fill_(mask, -float('inf'))
        attention = F.softmax(attention, dim=-1)
        # apply dropout
        attention = F.dropout(attention, self._dropout_p)
        # multiplyt it with V
        attention = torch.matmul(attention, V)
        # convert attention back to its input original size
        restore_chunk_size = int(attention.size(0) / self._h)
        attention = torch.cat(
            attention.split(split_size=restore_chunk_size, dim=0), dim=2)
        # residual connection
        attention += query
        return attention

