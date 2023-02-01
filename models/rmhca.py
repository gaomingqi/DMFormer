# residual multi-head cross-modal attention
from torch import nn
from einops import rearrange

from .segmentation import VisionLanguageFusionModule

class RMHCA(nn.Module):

    def __init__(self, target_size: int, inter_size: int = 256):
        super(RMHCA, self).__init__()
        self.target_size = target_size
        self.inter_size = inter_size
        # linear layers
        self.linear_in = nn.Linear(target_size, inter_size)
        self.linear_out = nn.Linear(inter_size, target_size)
        # mhca
        self.mhca = VisionLanguageFusionModule(d_model=inter_size, nhead=8)
    
    def forward(self, visual_feats, lang_dict):
        """
        residual mhca
        visual_feats: visual features (bt, hiwi, ci)
        lang_feats: language features ()
        return enhanced visual features        
        """
        bt, hw, c = visual_feats.shape
        lang_feats, lang_masks, lang_pos = lang_dict['feats'], lang_dict['masks'], lang_dict['pos']
        bs = lang_feats.shape[1]
        t = bt // bs
        visual_feats = rearrange(visual_feats, '(b t) hw c -> b (t hw) c', b=bs, t=t, hw=hw)
        src = self.linear_in(visual_feats) 
        src = src.permute(1, 0, 2)  # hw, bt, c
        src = self.mhca(tgt=src, memory=lang_feats, 
                memory_key_padding_mask=lang_masks,
                pos=lang_pos,
                query_pos=None
        )
        src = src.permute(1, 0, 2)  # bt, hw, c
        src = self.linear_out(src)
        visual_feats = visual_feats + src
        visual_feats = rearrange(visual_feats, 'b (t hw) c -> (b t) hw c', b=bs, t=t, hw=hw)

        return visual_feats
