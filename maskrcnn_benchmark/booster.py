# interface for glip booster

from .config import cfg
from .modeling.detector.generalized_vl_rcnn import GeneralizedVLRCNN


def build_glip_booster(args, rank):
    # configuration
    cfg.local_rank = rank
    cfg.num_gpus = args.ngpu

    # opts from glip large configurations
    opts = ['OUTPUT_DIR', '', 'TEST.IMS_PER_BATCH', f'{args.batch_size}', 'SOLVER.IMS_PER_BATCH', f'{args.batch_size}', \
        'TEST.MDETR_STYLE_AGGREGATE_CLASS_NUM', '100', 'TEST.EVAL_TASK', 'grounding', \
            'MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS', 'False']

    cfg.merge_from_file(args.booster_config_file)

    cfg.merge_from_list(opts)
    cfg.freeze()
    booster_model = GeneralizedVLRCNN(cfg)
    
    return booster_model, cfg
