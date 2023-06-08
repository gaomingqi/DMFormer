## dmformer
PyTorch implementation for the paper: Decoupled Multimodal Transformers for Referring Video Object Segmentation (accepted by IEEE Transactions on Circuits and Systems for Video Technology, TCSVT).

### Abstract
Referring Video Object Segmentation (RVOS) aims to segment the text-depicted object from video sequences. With excellent capabilities in long-range modelling and information interaction, transformers have been increasingly applied in existing RVOS architectures. To better leverage multimodal data, most efforts focus on the interaction between visual and textual features. However, they ignore the syntactic structures of the text during the interaction, where all textual components are intertwined, resulting in ambiguous vision-language alignment. In this paper, we improve the multimodal interaction by DECOUPLING the interweave. Specifically, we train a lightweight subject perceptron, which extracts the subject part from the input text. Then, the subject and text features are fed into two parallel branches to interact with visual features. This enables us to perform subject-aware and context-aware interactions, respectively, thus encouraging more explicit and discriminative feature embedding and alignment. Moreover, we find the decoupled architecture also facilitates incorporating the vision-language pre-trained alignment into RVOS, further improving the segmentation performance. 



### Installation
```
pip install -r requirements.txt
python setup.py build develop --user
cd models/ops
python setup.py build install
cd ../..
```

### Training
Pre-Training
```
python3 -m torch.distributed.launch --nproc_per_node=[Number of GPUs] --master_port=[Master Port] --use_env main_pretrain.py --with_box_refine --binary --use_glip --freeze_text_encoder --freeze_vision_encoder --epochs 10 --lr_drop 6 8 --output_dir [Output Path] --backbone swin_t_p4w7 --dataset_file all --coco_path [COCO Path] --glip_checkpoint [GLIP checkpoint Path] --booster_checkpoint [GLIP checkpoint Path] --booster_config_file ./config/glip_Swin_L.yaml --use_rmhca --batch_size [Batch Size]
```
Fine-Tuning
```
python3 -m torch.distributed.launch --nproc_per_node=[Number of GPUs] --master_port=[Master Port] --use_env main.py --with_box_refine --binary --use_glip --freeze_text_encoder --freeze_vision_encoder --epochs 6 --lr_drop 3 5 --output_dir [Output Path] --backbone swin_t_p4w7 --ytvos_path [Ref-YouTube-VOS Path] --glip_checkpoint [GLIP checkpoint Path] --booster_checkpoint [GLIP checkpoint Path] --booster_config_file ./config/glip_Swin_L.yaml --use_rmhca --pretrained_weights [Pre-Trained checkpoint Path] --batch_size [Batch Size]
```
### Inference

```
python3 inference_ytvos.py --with_box_refine --binary --freeze_text_encoder --output_dir [Output Path] --resume [Checkpoint Path] --ngpu 1 --batch_size 1 --backbone swin_t_p4w7 --ytvos_path [Ref-YouTube-VOS Path] --use_glip --glip_checkpoint [GLIP checkpoint] --use_rmhca
```

### Acknowledgement
This work is based on [ReferFormer](https://github.com/wjn922/ReferFormer) and [GLIP](https://github.com/microsoft/GLIP). Thanks for the authors for their efforts!
