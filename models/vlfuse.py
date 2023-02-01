import torch
import torch.nn.functional as F
from torch import nn


def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def permute_and_flatten(layer, N, A, C, H, W):
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class VLFuse(torch.nn.Module):
    """
    Early Fusion Module
    """
    def __init__(self, cfg=None):
        super(VLFuse, self).__init__()
        self.init_configs(cfg)
        self.cfg = cfg

        self.use_checkpoint = False
        
        # bi-direction (text->image, image->text)
        self.b_attn = BiAttentionBlockForCheckpoint(v_dim=self.joint_embedding_size,
                    l_dim=self.lang_dim,
                    embed_dim=self.embed_dim,
                    num_heads=self.n_head,
                    hidden_dim=self.i2t_hidden_dim,
                    dropout=0.1,
                    drop_path=.0,
                    init_values=1.0 / 8,
                    cfg=cfg
                    )

    def init_configs(self, cfg=None):
        # common params
        self.lang_model = 'bert-base-uncased'
        self.joint_embedding_size = 256
        self.joint_embedding_dropout = 0.1
        self.joint_mlp_layers = 2

        self.max_query_len = 256
        self.n_layers = 1
        self.coord_dim = 8
        self.joint_inp_dim = self.coord_dim + self.joint_embedding_size
        self.joint_out_dim = 256

        # mha params
        self.n_head = 8
        self.embed_dim = 2048
        self.t2i_hidden_dim = 1024  # 256 * 4
        self.i2t_hidden_dim = 3072  # 768 * 4

        if self.lang_model in ["bert-base-uncased", "roberta-base", "clip"]:
            self.lang_dim = 256
        else:
            self.lang_dim = 1024

    def forward(self, x):
        visual_features = x["visual"]
        language_dict_features = x["lang"]

        fused_visual_features = None
        fused_language_dict_features = None

        q0, q1, q2, q3, l0, l1, l2, l3 = self.b_attn(
            visual_features[0], visual_features[1],
            visual_features[2], visual_features[3],
            language_dict_features['hidden'],
            language_dict_features['masks'],
            None
        )

        fused_visual_features = [q0, q1, q2, q3]
        language_features = l0

        language_dict_features['hidden'] = language_features
        fused_language_dict_features = language_dict_features

        features_dict = {"visual": fused_visual_features,
                         "lang": fused_language_dict_features}

        return features_dict


class BiAttentionBlockForCheckpoint(nn.Module):
    def __init__(self, v_dim, l_dim, embed_dim, num_heads, hidden_dim=None, dropout=0.1,
                 drop_path=.0, init_values=1e-4, cfg=None):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super(BiAttentionBlockForCheckpoint, self).__init__()

        # pre layer norm
        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.layer_norm_l = nn.LayerNorm(l_dim)
        self.attn = BiMultiHeadAttention(v_dim=v_dim,
                                         l_dim=l_dim,
                                         embed_dim=embed_dim,
                                         num_heads=num_heads,
                                         dropout=dropout,
                                         cfg=cfg)

        # add layer scale for training stability
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.gamma_v = nn.Parameter(init_values * torch.ones((v_dim)), requires_grad=True)
        self.gamma_l = nn.Parameter(init_values * torch.ones((l_dim)), requires_grad=True)

        self.cfg = cfg

    def forward(self, q0, q1, q2, q3, l, attention_mask_l=None, dummy_tensor=None):

        visu_feat = []
        size_per_level, visual_features_flatten = [], []
        for ii, feat_per_level in enumerate([q0, q1, q2, q3]):
            bs, c, h, w = feat_per_level.shape
            size_per_level.append([h, w])
            feat = permute_and_flatten(feat_per_level, bs, 1, c, h, w)
            visual_features_flatten.append(feat)
        visual_features_flatten = cat(visual_features_flatten, dim=1)
        new_v, new_l = self.single_attention_call(visual_features_flatten, l, attention_mask_l=attention_mask_l)
        # [bs, N, C] -> [bs, C, N]
        new_v = new_v.transpose(1, 2).contiguous()

        start = 0
        for (h, w) in size_per_level:
            new_v_per_level = new_v[:, :, start:start + h * w].view(bs, -1, h, w).contiguous()
            visu_feat.append(new_v_per_level)
            start += h * w
        
        lang_feat = [new_l, None, None, None, None]

        return visu_feat[0], visu_feat[1], visu_feat[2], visu_feat[3], lang_feat[0], lang_feat[1], lang_feat[2], lang_feat[3]

    
    def single_attention_call(self, v, l, attention_mask_l=None, dummy_tensor=None):
        v = self.layer_norm_v(v)
        l = self.layer_norm_l(l)
        delta_v, delta_l = self.attn(v, l, attention_mask_l=attention_mask_l)
        # v, l = v + delta_v, l + delta_l
        v = v + self.drop_path(self.gamma_v * delta_v)
        l = l + self.drop_path(self.gamma_l * delta_l)
        return v, l


class BiMultiHeadAttention(nn.Module):
    def __init__(self, v_dim, l_dim, embed_dim, num_heads, dropout=0.1, cfg=None):
        super(BiMultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.v_dim = v_dim
        self.l_dim = l_dim

        assert (
                self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.l_proj = nn.Linear(self.l_dim, self.embed_dim)
        self.values_v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.values_l_proj = nn.Linear(self.l_dim, self.embed_dim)

        self.out_v_proj = nn.Linear(self.embed_dim, self.v_dim)
        self.out_l_proj = nn.Linear(self.embed_dim, self.l_dim)

        self.stable_softmax_2d = False
        self.clamp_min_for_underflow = True
        self.clamp_max_for_overflow = True

        self._reset_parameters()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.l_proj.weight)
        self.l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_v_proj.weight)
        self.values_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_l_proj.weight)
        self.values_l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_v_proj.weight)
        self.out_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_l_proj.weight)
        self.out_l_proj.bias.data.fill_(0)

    def forward(self, v, l, attention_mask_l=None):
        bsz, tgt_len, embed_dim = v.size()

        query_states = self.v_proj(v) * self.scale
        key_states = self._shape(self.l_proj(l), -1, bsz)
        value_v_states = self._shape(self.values_v_proj(v), -1, bsz)
        value_l_states = self._shape(self.values_l_proj(l), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_v_states = value_v_states.view(*proj_shape)
        value_l_states = value_l_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        # attn_weights_l = nn.functional.softmax(attn_weights.transpose(1, 2), dim=-1)

        if self.stable_softmax_2d:
            attn_weights = attn_weights - attn_weights.max()
        
        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(attn_weights, min=-50000) # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(attn_weights, max=50000) # Do not increase 50000, data type half has quite limited range

        attn_weights_T = attn_weights.transpose(1, 2)
        attn_weights_l = (attn_weights_T - torch.max(attn_weights_T, dim=-1, keepdim=True)[
            0])
        if self.clamp_min_for_underflow:
            attn_weights_l = torch.clamp(attn_weights_l, min=-50000) # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights_l = torch.clamp(attn_weights_l, max=50000) # Do not increase 50000, data type half has quite limited range

        attn_weights_l = attn_weights_l.softmax(dim=-1)

        if attention_mask_l is not None:
            assert (attention_mask_l.dim() == 2)
            attention_mask = attention_mask_l.unsqueeze(1).unsqueeze(1)
            attention_mask = attention_mask.expand(bsz, 1, tgt_len, src_len)
            attention_mask = attention_mask.masked_fill(attention_mask == 0, -9e15)

            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights_v = nn.functional.softmax(attn_weights, dim=-1)

        attn_probs_v = F.dropout(attn_weights_v, p=self.dropout, training=self.training)
        attn_probs_l = F.dropout(attn_weights_l, p=self.dropout, training=self.training)

        attn_output_v = torch.bmm(attn_probs_v, value_l_states)
        attn_output_l = torch.bmm(attn_probs_l, value_v_states)


        if attn_output_v.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output_v` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output_v.size()}"
            )

        if attn_output_l.size() != (bsz * self.num_heads, src_len, self.head_dim):
            raise ValueError(
                f"`attn_output_l` should be of size {(bsz, self.num_heads, src_len, self.head_dim)}, but is {attn_output_l.size()}"
            )

        attn_output_v = attn_output_v.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output_v = attn_output_v.transpose(1, 2)
        attn_output_v = attn_output_v.reshape(bsz, tgt_len, self.embed_dim)

        attn_output_l = attn_output_l.view(bsz, self.num_heads, src_len, self.head_dim)
        attn_output_l = attn_output_l.transpose(1, 2)
        attn_output_l = attn_output_l.reshape(bsz, src_len, self.embed_dim)

        attn_output_v = self.out_v_proj(attn_output_v)
        attn_output_l = self.out_l_proj(attn_output_l)

        return attn_output_v, attn_output_l
