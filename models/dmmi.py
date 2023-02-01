"""
DMMI model class.
Modified from ReferFormer (https://github.com/wjn922/ReferFormer)
"""
import torch
import torch.nn.functional as F
import os
import math
import copy

from torch import nn
from einops import rearrange, repeat
from transformers import BertTokenizer, BertModel, RobertaModel, RobertaTokenizerFast, AutoTokenizer
from transformers.models.bert.modeling_bert import BertConfig
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       nested_tensor_from_videos_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .position_encoding import PositionEmbeddingSine1D
from .backbone import build_backbone
from .deformable_transformer import build_deforamble_transformer
from .segmentation import CrossModalFPNDecoder, VisionLanguageFusionModule
from .matcher import build_matcher
from .criterion import SetCriterion
from .postprocessors import build_postprocessors
from .bert import BertEncoder, build_bert_backbone
from .bertlayer import BertEncoderLayer
from .deformable_transformer import DeformableTransformerEncoderLayer


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # this disables a huggingface tokenizer warning (printed every epoch)

# for noun chunk extraction
import numpy as np
import spacy
nlp = spacy.load('en_core_web_sm')

class DMMI(nn.Module):
    """ This is the DMMI module that performs referring video object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels, 
                    num_frames, mask_dim, dim_feedforward,
                    controller_layers, dynamic_mask_channels, 
                    aux_loss=False, with_box_refine=False, two_stage=False, 
                    freeze_text_encoder=False, freeze_vision_encoder=False, rel_coord=True, use_glip=False, glip_checkpoint=None):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DMMI can detect in a video. For ytvos, we recommend 5 queries for each frame.
            num_frames:  number of clip frames
            mask_dim: dynamic conv inter layer channel number.
            dim_feedforward: vision-language fusion module ffn channel number.
            dynamic_mask_channels: the mask feature output channel number.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        self.use_glip = use_glip
        
        # Build Transformer
        # NOTE: different deformable detr, the query_embed out channels is
        # hidden_dim instead of hidden_dim * 2
        # This is because, the input to the decoder is text embedding feature
        self.query_embed = nn.Embedding(num_queries, hidden_dim) 
        
        # follow deformable-detr, we use the last three stages of backbone
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides[-3:])
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[-3:][_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs): # downsample 2x
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[-3:][0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        self.num_frames = num_frames
        self.mask_dim = mask_dim
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        assert two_stage == False, "args.two_stage must be false!"

        # initialization
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = transformer.decoder.num_layers
        
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None

        # Build Text Encoder
        if use_glip:
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.tokenizer_vocab = self.tokenizer.get_vocab()
            self.tokenizer_vocab_ids = [item for key,item in self.tokenizer_vocab.items()]
            self.text_encoder = BertEncoder()
        else:
            self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
            self.text_encoder = RobertaModel.from_pretrained('roberta-base')

################################################
        # for DMMI
        bert_config = BertConfig.from_pretrained('bert-base-uncased')
        bert_config.hidden_size = 256
        bert_config.num_attention_heads = 8

        self.num_interactions = 2   # number of basic interaction modules

        # Subject-aware and Context-aware fusion
        self.vlfuse_list_subject = nn.ModuleList()
        self.vlfuse_list_context = nn.ModuleList()  
        for itr in range(self.num_interactions):
            self.vlfuse_list_subject.append(VisionLanguageFusionModule(d_model=hidden_dim, nhead=8))
            self.vlfuse_list_context.append(VisionLanguageFusionModule(d_model=hidden_dim, nhead=8))

        # Subject-aware and Context-aware attentions
        # Vision
        self.level_embed_context = nn.Parameter(torch.Tensor(num_feature_levels, 256))
        self.level_embed_subject = nn.Parameter(torch.Tensor(num_feature_levels, 256))
        normal_(self.level_embed_context)
        normal_(self.level_embed_subject)

        self.visionlayer_list_context = nn.ModuleList()
        self.visionlayer_list_subject = nn.ModuleList()      
        for itr in range(self.num_interactions):
            self.visionlayer_list_context.append(DeformableTransformerEncoderLayer(256, dim_feedforward,
                                                          0.1, 'relu',
                                                          num_feature_levels,
                                                          8, 4))
            self.visionlayer_list_subject.append(DeformableTransformerEncoderLayer(256, dim_feedforward,
                                                          0.1, 'relu',
                                                          num_feature_levels,
                                                          8, 4))

        # 3.2. Language
        self.bertlayer_list_context = nn.ModuleList()
        self.bertlayer_list_subject = nn.ModuleList()
        for itr in range(self.num_interactions):
            self.bertlayer_list_context.append(BertEncoderLayer(bert_config))
            self.bertlayer_list_subject.append(BertEncoderLayer(bert_config))

        # resize the bert output channel to transformer d_model
        self.resizer_context_v = FeatureResizer(input_feat_size=hidden_dim, output_feat_size=hidden_dim//2, dropout=0.1)
        self.resizer_context_l = FeatureResizer(input_feat_size=hidden_dim, output_feat_size=hidden_dim//2, dropout=0.1)
        self.resizer_subject_v = FeatureResizer(input_feat_size=hidden_dim, output_feat_size=hidden_dim//2, dropout=0.1)
        self.resizer_subject_l = FeatureResizer(input_feat_size=hidden_dim, output_feat_size=hidden_dim//2, dropout=0.1)
        
        # Subject Perceptron
        self.subject_perceptron_1 = BertEncoderLayer(bert_config)
        self.subject_perceptron_2 = nn.Linear(hidden_dim, 2)
        self.subject_perceptron_3 = nn.Softmax(dim=-1)

################################################

        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)
        
        if freeze_vision_encoder:
            for n, p in self.backbone.named_parameters():
                if 'rmhca' in n:
                    continue
                p.requires_grad_(False)

        # resize the bert output channel to transformer d_model
        self.resizer = FeatureResizer(
            input_feat_size=768,
            output_feat_size=hidden_dim,
            dropout=0.1,
        )

        self.text_pos = PositionEmbeddingSine1D(hidden_dim, normalize=True)

        # Build FPN Decoder
        self.rel_coord = rel_coord
        feature_channels = [self.backbone.num_channels[0]] + 3 * [hidden_dim]
        self.pixel_decoder = CrossModalFPNDecoder(feature_channels=feature_channels, conv_dim=hidden_dim, 
                                                  mask_dim=mask_dim, dim_feedforward=dim_feedforward, norm="GN")

        # Build Dynamic Conv
        self.controller_layers = controller_layers 
        self.in_channels = mask_dim
        self.dynamic_mask_channels = dynamic_mask_channels
        self.mask_out_stride = 4
        self.mask_feat_stride = 4

        weight_nums, bias_nums = [], []
        for l in range(self.controller_layers):
            if l == 0:
                if self.rel_coord:
                    weight_nums.append((self.in_channels + 2) * self.dynamic_mask_channels)
                else:
                    weight_nums.append(self.in_channels * self.dynamic_mask_channels)
                bias_nums.append(self.dynamic_mask_channels)
            elif l == self.controller_layers - 1:
                weight_nums.append(self.dynamic_mask_channels * 1) # output layer c -> 1
                bias_nums.append(1)
            else:
                weight_nums.append(self.dynamic_mask_channels * self.dynamic_mask_channels)
                bias_nums.append(self.dynamic_mask_channels)

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)

        self.controller = MLP(hidden_dim, hidden_dim, self.num_gen_params, 3)
        for layer in self.controller.layers:
            nn.init.zeros_(layer.bias)
            nn.init.xavier_uniform_(layer.weight)   

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, samples: NestedTensor, captions, targets):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensors: image sequences, of shape [num_frames x 3 x H x W]
               - samples.mask: a binary mask of shape [num_frames x H x W], containing 1 on padded pixels
               - captions: list[str]
               - targets:  list[dict]

            It returns a dict with the following elements:
               - "pred_masks": Shape = [batch_size x num_queries x out_h x out_w]

               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        # Backbone
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_videos_list(samples) 

        # text features
        text_features, text_sentence_features, positive_maps = self.forward_text(captions, device=samples.tensors.device)
        text_pos = self.text_pos(text_features).permute(2, 0, 1)  # [length, batch_size, c]
        text_word_features, text_word_masks = text_features.decompose()
        text_word_features = text_word_features.permute(1, 0, 2)  # [length, batch_size, c]
        lang_dict = {'feats': text_word_features, 'masks': text_word_masks, 'pos': text_pos}

        # visual features (list[NestedTensor]): res2 -> res5, shape of tensors is [B*T, Ci, Hi, Wi]
        # pos (list[Tensor]): shape of [B*T, C, Hi, Wi]
        features, pos = self.backbone(samples, lang_dict) 
        text_word_features = text_word_features.permute(1, 0, 2)  # [length, batch_size, c]

        b = len(captions)
        t = pos[0].shape[0] // b

        # For A2D-Sentences and JHMDB-Sentencs dataset, only one frame is annotated for a clip
        if 'valid_indices' in targets[0]:
            valid_indices = torch.tensor([i * t + target['valid_indices'] for i, target in enumerate(targets)]).to(pos[0].device)
            for feature in features:
                feature.tensors = feature.tensors.index_select(0, valid_indices)
                feature.mask = feature.mask.index_select(0, valid_indices)
            for i, p in enumerate(pos):
                pos[i] = p.index_select(0, valid_indices)
            samples.mask = samples.mask.index_select(0, valid_indices)
            # t: num_frames -> 1
            t = 1
        
        # prepare vision and text features for transformer
        srcs = []
        masks = []
        poses = []

        # perceive the subject part from the text
        sub_1 = self.subject_perceptron_1(text_word_features, text_word_masks)
        sub_2 = self.subject_perceptron_2(sub_1)
        sub_3 = self.subject_perceptron_3(sub_2)
        subject_ = sub_3[:,:,-1]    # [batch_size, length]

        # select a noun phrase with the largest prob
        noun_idx_list = []
        noun_max_span = 0   # for batched subject-aware interaction
        for i, positive_map in enumerate(positive_maps):
            best_noun = 1
            best_prob = 0
            # each noun chunk
            for pk, pv in positive_map.items():
                prob = subject_[i, pv[0]:(pv[-1]+1)].max().item()

                if best_prob < prob:
                    best_prob = prob
                    best_noun = pk
                    noun_max_span = max(noun_max_span, (pv[-1]+1-pv[0]))

            noun_idx_list.append(best_noun)

        # Prepare input for basic interaction module (convert visual features to 256 dims)
        for l, (feat, pos_l) in enumerate(zip(features[-3:], pos[-3:])): 
            src, mask = feat.decompose()            
            src_proj_l = self.input_proj[l](src)

            srcs.append(src_proj_l)
            masks.append(mask)
            poses.append(pos_l)
            assert mask is not None
        
        if self.num_feature_levels > (len(features) - 1):
            _len_srcs = len(features) - 1 # fpn level
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)

                srcs.append(src)
                masks.append(mask)
                poses.append(pos_l)
        
        # srcs:                 bt, c, h, w
        # text_word_features:   l, bt, c
        # prepare visual inputs for context/subject-aware MM interaction
        srcs_context = []
        srcs_subject = []
        for src in srcs:
            srcs_context.append(src.clone())
            srcs_subject.append(src.clone())
        
        # prepare context text inputs
        language_word_context = text_word_features.clone()
        language_word_context = language_word_context.permute(1, 0, 2)  # [length, batch_size, c]

        # prepare subject text inputs
        batched_word_features = text_word_features.clone()
        # extract subject part from all text featurs
        language_word_subject = []
        language_mask_subject = []
        _, _, c = batched_word_features.shape
        for i, word_features in enumerate(batched_word_features):
            subject_features = torch.zeros([noun_max_span, c])
            subject_mask = torch.zeros([noun_max_span])
            current_span = positive_maps[i][noun_idx_list[i]][-1]+1 - positive_maps[i][noun_idx_list[i]][0]
            
            subject_features[:current_span] = word_features[
                positive_maps[i][noun_idx_list[i]][0]:(positive_maps[i][noun_idx_list[i]][-1]+1)]
            subject_mask[:current_span] = 1
            
            language_word_subject.append(subject_features.unsqueeze(0))
            language_mask_subject.append(subject_mask.unsqueeze(0).long())

        language_word_subject = torch.cat(language_word_subject, dim=0).to(pos[0].device)
        language_mask_subject = torch.cat(language_mask_subject, dim=0).to(pos[0].device)

        text_features_subject = NestedTensor(language_word_subject, language_mask_subject)
        text_pos_subject = self.text_pos(text_features_subject).permute(2, 0, 1)  # [length, batch_size, c]
        language_word_subject = language_word_subject.permute(1, 0, 2)  # [len, bs, c]
        text_word_features = text_word_features.permute(1, 0, 2)        # [len, bs, c]

        # 1. Context-aware Interaction
        is_reshape_list = [True, False]    # whether reshape visual features back to its original resolution
        for itr, is_reshape in enumerate(is_reshape_list):
            # 1.1. Vision-Language Attention
            for lvl, sc in enumerate(srcs_context):
                _, c, h, w = sc.shape
                sc = rearrange(sc, '(b t) c h w -> (t h w) b c', b=b, t=t)
                sc = self.vlfuse_list_context[itr](tgt=sc,
                                          memory=language_word_context,
                                          memory_key_padding_mask=text_word_masks,
                                          pos=text_pos,
                                          query_pos=None
                )
                sc = rearrange(sc, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)
                srcs_context[lvl] = sc
            
            # 1.2. Vision Attention
            srcs_context = self.visual_encoder_context(itr, srcs_context, masks, poses, is_reshape)
            
            # 1.3. Language Attention
            language_word_context = language_word_context.permute(1, 0, 2)  # [length, batch_size, c]
            language_word_context = self.bertlayer_list_context[itr](language_word_context, text_word_masks)
            if itr < (len(is_reshape_list)-1):
                language_word_context = language_word_context.permute(1, 0, 2)  # [length, batch_size, c]

        srcs_context = self.resizer_context_v(srcs_context)
        text_sentence_context = self.resizer_context_l(text_sentence_features)
        text_word_context = self.resizer_context_l(language_word_context)

        # 2. Self Interaction
        is_reshape_list = [True, False]
        for itr, is_reshape in enumerate(is_reshape_list):
            # 2.1. Vision-Language Attention
            for lvl, ss in enumerate(srcs_subject):
                _, c, h, w = ss.shape
                ss = rearrange(ss, '(b t) c h w -> (t h w) b c', b=b, t=t)
                ss = self.vlfuse_list_subject[itr](tgt=ss,
                                          memory=language_word_subject,
                                          memory_key_padding_mask=language_mask_subject,
                                          pos=text_pos_subject,
                                          query_pos=None
                )
                ss = rearrange(ss, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)
                srcs_subject[lvl] = ss
            
            # 2.2. Vision Attention
            srcs_subject = self.visual_encoder_subject(itr, srcs_subject, masks, poses, is_reshape)
            
            # 2.3. Language Attention
            language_word_subject = language_word_subject.permute(1, 0, 2)      # [length, batch_size, c]
            language_word_subject = self.bertlayer_list_subject[itr](language_word_subject, language_mask_subject)
            if itr < (len(is_reshape_list)-1):
                language_word_subject = language_word_subject.permute(1, 0, 2)  # [length, batch_size, c]
        
        srcs_subject = self.resizer_subject_v(srcs_subject)
        text_sentence_subject = self.resizer_subject_l(text_sentence_features)
        text_word_subject = self.resizer_subject_l(language_word_subject)

        # concatenate decoupled interaction results
        text_context = text_sentence_context + 0.5*text_word_context.mean(dim=1)
        text_subject = text_sentence_subject + 0.5*text_word_subject.mean(dim=1)

        memory = torch.concat([srcs_context, srcs_subject], dim=-1)
        text_embed = torch.concat([text_context, text_subject], dim=-1)

        # transformer decoder 
        query_embeds = self.query_embed.weight  # [num_queries, c]
        text_embed = repeat(text_embed, 'b c -> b t q c', t=t, q=self.num_queries)
        hs, memory, init_reference, inter_references, _, _, _ = \
                                            self.transformer(srcs, memory, text_embed, masks, poses, query_embeds)
        # hs: [l, batch_size*time, num_queries_per_frame, c]
        # memory: list[Tensor], shape of tensor is [batch_size*time, c, hi, wi]
        # init_reference: [batch_size*time, num_queries_per_frame, 2]
        # inter_references: [l, batch_size*time, num_queries_per_frame, 4]
        
        del srcs
        for (src_c, src_s) in zip(srcs_context, srcs_subject):
            del src_c
            del src_s

        out = {}
        
        # prediction
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid() # cxcywh, range in [0,1]
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        # rearrange
        outputs_class = rearrange(outputs_class, 'l (b t) q k -> l b t q k', b=b, t=t)
        outputs_coord = rearrange(outputs_coord, 'l (b t) q n -> l b t q n', b=b, t=t)
        out['pred_logits'] = outputs_class[-1] # [batch_size, time, num_queries_per_frame, num_classes]
        out['pred_boxes'] = outputs_coord[-1]  # [batch_size, time, num_queries_per_frame, 4]
        out['subject'] = sub_2      # [batch_size, max_length]

        # Segmentation
        mask_features = self.pixel_decoder(features, text_features, pos, memory, nf=t) # [batch_size*time, c, out_h, out_w]
        mask_features = rearrange(mask_features, '(b t) c h w -> b t c h w', b=b, t=t)

        # dynamic conv
        outputs_seg_masks = []
        for lvl in range(hs.shape[0]):
            dynamic_mask_head_params = self.controller(hs[lvl])   # [batch_size*time, num_queries_per_frame, num_params]
            dynamic_mask_head_params = rearrange(dynamic_mask_head_params, '(b t) q n -> b (t q) n', b=b, t=t)
            lvl_references = inter_references[lvl, ..., :2]
            lvl_references = rearrange(lvl_references, '(b t) q n -> b (t q) n', b=b, t=t)
            outputs_seg_mask = self.dynamic_mask_with_coords(mask_features, dynamic_mask_head_params, lvl_references, targets)
            outputs_seg_mask = rearrange(outputs_seg_mask, 'b (t q) h w -> b t q h w', t=t)
            outputs_seg_masks.append(outputs_seg_mask)
        out['pred_masks'] = outputs_seg_masks[-1]  # [batch_size, time, num_queries_per_frame, out_h, out_w]

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_seg_masks)
        
        if not self.training:
            # for visualization
            inter_references = inter_references[-2, :, :, :2]  # [batch_size*time, num_queries_per_frame, 2]
            inter_references = rearrange(inter_references, '(b t) q n -> b t q n', b=b, t=t) 
            out['reference_points'] = inter_references  # the reference points of last layer input
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a, "pred_boxes": b, "pred_masks": c} 
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_seg_masks[:-1])]

    def forward_text(self, captions, device):

        if self.use_glip:
            tokenized = self.tokenizer.batch_encode_plus(captions, max_length=256, \
                padding="max_length", return_special_tokens_mask=True, return_tensors="pt", truncation=True).to(device)
            
            # noun chunks
            positive_maps = []
            max_len = 0
            max_bs = 0
            for b, caption in enumerate(captions):
                doc = nlp(caption)
                noun_chunks = list(doc.noun_chunks)
                positive_map = {}
                for i, nc in enumerate(noun_chunks):
                    start_ids = [i for i, word_id in enumerate(tokenized.encodings[b].word_ids) if word_id == nc.start]
                    end_ids = [i for i, word_id in enumerate(tokenized.encodings[b].word_ids) if word_id == (nc.end-1)]
                    try:
                        positive_map[i+1] = np.arange(min(start_ids), max(end_ids)+1).tolist()
                    except:
                        continue
                    
                if len(positive_map) == 0:
                    # set positive map from the embedded language features
                    positive_map = {1: tokenized['attention_mask'][b].nonzero(as_tuple=True)[0].tolist()}
                positive_maps.append(positive_map)
            # end of noun chunks
                if tokenized.data['attention_mask'][b].sum() > max_len:
                    max_len = tokenized.data['attention_mask'][b].sum()
                    max_bs = b

            input_ids = tokenized.input_ids
            tokenizer_input = {"input_ids": input_ids, "attention_mask": tokenized.attention_mask}
            language_dict_features = self.text_encoder(tokenizer_input)
            # take sentence features
            text_sentence_features = language_dict_features['aggregate']
            text_sentence_features = self.resizer(text_sentence_features)
            # take token features
            
            indices = tokenized.attention_mask[max_bs].nonzero(as_tuple=True)[0]
            text_features = torch.index_select(language_dict_features['embedded'], 1, indices)
            text_features = self.resizer(text_features)
            text_masks = torch.index_select(tokenized.attention_mask, 1, indices)
            text_features = NestedTensor(text_features, text_masks)     # NestedTensor        
        else:
            tokenized = self.tokenizer.batch_encode_plus(captions, padding="longest", return_tensors="pt").to(device)
            encoded_text = self.text_encoder(**tokenized)
            # encoded_text.last_hidden_state: [batch_size, length, 768]
            # encoded_text.pooler_output: [batch_size, 768]
            text_attention_mask = tokenized.attention_mask.ne(1).bool()
            # text_attention_mask: [batch_size, length]

            text_features = encoded_text.last_hidden_state 
            text_features = self.resizer(text_features)    
            text_masks = text_attention_mask              
            text_features = NestedTensor(text_features, text_masks) # NestedTensor

            text_sentence_features = encoded_text.pooler_output  
            text_sentence_features = self.resizer(text_sentence_features)  
        
        return text_features, text_sentence_features, positive_maps

    def dynamic_mask_with_coords(self, mask_features, mask_head_params, reference_points, targets):
        """
        Add the relative coordinates to the mask_features channel dimension,
        and perform dynamic mask conv.

        Args:
            mask_features: [batch_size, time, c, h, w]
            mask_head_params: [batch_size, time * num_queries_per_frame, num_params]
            reference_points: [batch_size, time * num_queries_per_frame, 2], cxcy
            targets (list[dict]): length is batch size
                we need the key 'size' for computing location.
        Return:
            outputs_seg_mask: [batch_size, time * num_queries_per_frame, h, w]
        """
        device = mask_features.device
        b, t, c, h, w = mask_features.shape
        # this is the total query number in all frames
        _, num_queries = reference_points.shape[:2]  
        q = num_queries // t  # num_queries_per_frame

        # prepare reference points in image size (the size is input size to the model)
        new_reference_points = [] 
        for i in range(b):
            img_h, img_w = targets[i]['size']
            scale_f = torch.stack([img_w, img_h], dim=0) 
            tmp_reference_points = reference_points[i] * scale_f[None, :] 
            new_reference_points.append(tmp_reference_points)
        new_reference_points = torch.stack(new_reference_points, dim=0) 
        # [batch_size, time * num_queries_per_frame, 2], in image size
        reference_points = new_reference_points  

        # prepare the mask features
        if self.rel_coord:
            reference_points = rearrange(reference_points, 'b (t q) n -> b t q n', t=t, q=q) 
            locations = compute_locations(h, w, device=device, stride=self.mask_feat_stride) 
            relative_coords = reference_points.reshape(b, t, q, 1, 1, 2) - \
                                    locations.reshape(1, 1, 1, h, w, 2) # [batch_size, time, num_queries_per_frame, h, w, 2]
            relative_coords = relative_coords.permute(0, 1, 2, 5, 3, 4) # [batch_size, time, num_queries_per_frame, 2, h, w]

            # concat features
            mask_features = repeat(mask_features, 'b t c h w -> b t q c h w', q=q) # [batch_size, time, num_queries_per_frame, c, h, w]
            mask_features = torch.cat([mask_features, relative_coords], dim=3)
        else:
            mask_features = repeat(mask_features, 'b t c h w -> b t q c h w', q=q) # [batch_size, time, num_queries_per_frame, c, h, w]
        mask_features = mask_features.reshape(1, -1, h, w) 

        # parse dynamic params
        mask_head_params = mask_head_params.flatten(0, 1) 
        weights, biases = parse_dynamic_params(
            mask_head_params, self.dynamic_mask_channels,
            self.weight_nums, self.bias_nums
        )

        # dynamic mask conv
        mask_logits = self.mask_heads_forward(mask_features, weights, biases, mask_head_params.shape[0]) 
        mask_logits = mask_logits.reshape(-1, 1, h, w)

        # upsample predicted masks
        assert self.mask_feat_stride >= self.mask_out_stride
        assert self.mask_feat_stride % self.mask_out_stride == 0

        mask_logits = aligned_bilinear(mask_logits, int(self.mask_feat_stride / self.mask_out_stride))
        mask_logits = mask_logits.reshape(b, num_queries, mask_logits.shape[-2], mask_logits.shape[-1])

        return mask_logits  # [batch_size, time * num_queries_per_frame, h, w]

    def mask_heads_forward(self, features, weights, biases, num_insts):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def visual_encoder_context(self, idx, srcs, masks, poses, back=False):
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, poses)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2) # [batch_size, hi*wi, c]
            mask = mask.flatten(1)               # [batch_size, hi*wi]
            pos_embed = pos_embed.flatten(2).transpose(1, 2) # [batch_size, hi*wi, c]
            lvl_pos_embed = pos_embed + self.level_embed_context[lvl].view(1, 1, -1)

            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

        # For a clip, concat all the features, first fpn layer size, then frame size
        src_flatten = torch.cat(src_flatten, 1)     # [bs*t, \sigma(hi*wi), c] 
        mask_flatten = torch.cat(mask_flatten, 1)   # [bs*t, \sigma(hi*wi)]
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1) 
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src_flatten
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src_flatten.device)
        # for _, layer in enumerate(self.layers):
        output = self.visionlayer_list_context[idx](output, lvl_pos_embed_flatten, reference_points, spatial_shapes, level_start_index, mask_flatten)
        # bt, Sigma(hi*wi), C
        
        # back to 2-d
        if back:
            q = []
            output = output.transpose(1, 2)
            b, c, _ = output.shape
            for lvl, (spatial_shape, start_index) in enumerate(zip(spatial_shapes, level_start_index)):
                qs = output[:, :, start_index:(start_index+spatial_shape[0]*spatial_shape[1])].view(b, c, spatial_shape[0], spatial_shape[1])
                q.append(qs)
            return q

        return output

        # encoder
        # memory: [bs*t, \sigma(hi*wi), c]
        # memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

    def visual_encoder_subject(self, idx, srcs, masks, poses, back=False):
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, poses)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2) # [batch_size, hi*wi, c]
            mask = mask.flatten(1)               # [batch_size, hi*wi]
            pos_embed = pos_embed.flatten(2).transpose(1, 2) # [batch_size, hi*wi, c]
            lvl_pos_embed = pos_embed + self.level_embed_subject[lvl].view(1, 1, -1)

            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

        # For a clip, concat all the features, first fpn layer size, then frame size
        src_flatten = torch.cat(src_flatten, 1)     # [bs*t, \sigma(hi*wi), c] 
        mask_flatten = torch.cat(mask_flatten, 1)   # [bs*t, \sigma(hi*wi)]
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1) 
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src_flatten
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src_flatten.device)
        # for _, layer in enumerate(self.layers):
        output = self.visionlayer_list_subject[idx](output, lvl_pos_embed_flatten, reference_points, spatial_shapes, level_start_index, mask_flatten)
        # bt, Sigma(hi*wi), C

        # back to 2-d
        if back:
            q = []
            output = output.transpose(1, 2)
            b, c, _ = output.shape
            for lvl, (spatial_shape, start_index) in enumerate(zip(spatial_shapes, level_start_index)):
                qs = output[:, :, start_index:(start_index+spatial_shape[0]*spatial_shape[1])].view(b, c, spatial_shape[0], spatial_shape[1])
                q.append(qs)
            return q

        return output


    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio


def parse_dynamic_params(params, channels, weight_nums, bias_nums):
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)

    num_insts = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(torch.split_with_sizes(params, weight_nums + bias_nums, dim=1))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]

    for l in range(num_layers):
        if l < num_layers - 1:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
        else:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts)

    return weight_splits, bias_splits

def aligned_bilinear(tensor, factor):
    assert tensor.dim() == 4 
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow),
        mode='bilinear',
        align_corners=True
    )
    tensor = F.pad(
        tensor, pad=(factor // 2, 0, factor // 2, 0),
        mode="replicate"
    )

    return tensor[:, :, :oh - 1, :ow - 1]


def compute_locations(h, w, device, stride=1):
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device)

    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device)

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


def build(args):
    if args.binary:
        num_classes = 1
    else:
        if args.dataset_file == 'ytvos':
            num_classes = 65 
        elif args.dataset_file == 'davis':
            num_classes = 78
        elif args.dataset_file == 'a2d' or args.dataset_file == 'jhmdb':
            num_classes = 1
        else: 
            num_classes = 91 # for coco
    device = torch.device(args.device)

    # backbone
    if 'video_swin' in args.backbone:
        from .video_swin_transformer import build_video_swin_backbone
        backbone = build_video_swin_backbone(args)
    elif 'swin' in args.backbone:
        from .swin_transformer import build_swin_backbone
        backbone = build_swin_backbone(args) 
    else:
        backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)

    model = DMMI(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        num_frames=args.num_frames,
        mask_dim=args.mask_dim,
        dim_feedforward=args.dim_feedforward,
        controller_layers=args.controller_layers,
        dynamic_mask_channels=args.dynamic_mask_channels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        freeze_text_encoder=args.freeze_text_encoder,
        freeze_vision_encoder=args.freeze_backbone,
        rel_coord=args.rel_coord,
        use_glip=args.use_glip,
        glip_checkpoint=args.glip_checkpoint
    )
    matcher = build_matcher(args)
    weight_dict = {}
    weight_dict['loss_ce'] = args.cls_loss_coef
    weight_dict['loss_bbox'] = args.bbox_loss_coef
    weight_dict['loss_giou'] = args.giou_loss_coef
    weight_dict['subject'] = 2
    if args.masks: # always true
        weight_dict['loss_mask'] = args.mask_loss_coef
        weight_dict['loss_dice'] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes']
    if args.masks:
        losses += ['masks']
    criterion = SetCriterion(
            num_classes, 
            matcher=matcher,
            weight_dict=weight_dict, 
            eos_coef=args.eos_coef, 
            losses=losses,
            focal_alpha=args.focal_alpha)
    criterion.to(device)

    # postprocessors, this is used for coco pretrain but not for rvos
    postprocessors = build_postprocessors(args, args.dataset_file)
    return model, criterion, postprocessors
