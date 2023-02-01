# visualise images/intermediate tensors via tensorboard
import numpy as np


def func_iou(bb, gtbb):
    iou = 0 
    iw = min(bb[2],gtbb[2]) - max(bb[0],gtbb[0]) + 1 
    ih = min(bb[3],gtbb[3]) - max(bb[1],gtbb[1]) + 1 
    if iw>0 and ih>0:
        ua = (bb[2]-bb[0]+1)*(bb[3]-bb[1]+1) + (gtbb[2]-gtbb[0]+1)*(gtbb[3]-gtbb[1]+1) - iw*ih 
        iou = np.float32(iw*ih*1.0/ua)

    return iou

def index_boxes(predictions, gt):
    """
    predictions: n x 4
    gt: 4
    return the box index and corresponding iou
    """
    if len(predictions) == 0:
        return 0, 0

    gtbb = gt.cpu().numpy().tolist()
    ious = []

    for pred in predictions:
        pred = pred.cpu().numpy().tolist()
        ious.append(func_iou(pred, gtbb))

    return np.argmax(ious), np.max(ious)


def generate_labels(images, boxes, targets, positive_map, pseudo_label):
    """
    generate pseudo labels by corresponding glip predictions to gt boxes
    boxes: glip predictions
    targets: gt data
    pseudo_label: initialised pseudo labels (all 0)
    positive_map: indices of noun phrases in the tokenised features
    """
    
    _, _, h, w = images.shape
    tboxes = targets['boxes'].clone()
    confidence_scores = []          # correspondence between noun phraase and box
    best_labels = []                # best index

    # for each frame in the training sample
    for (box, tbox) in zip(boxes, tboxes):
        box = box.resize((w, h))
        pred_boxes = box.bbox           # n x 4
        # cxcywh to xyxy, de-nomalise
        tbox[0], tbox[1], tbox[2], tbox[3] = \
            (tbox[0]-(tbox[2]/2))*w, (tbox[1]-(tbox[3]/2))*h, \
                (tbox[0]+(tbox[2]/2))*w, (tbox[1]+(tbox[3]/2))*h

        # retrieve the predicted box most close to the gt box
        max_idx, iou = index_boxes(pred_boxes, tbox)         # max_idx (pred bbox, 5)
        
        # retrieve the phrase noun align the best with the (max_idx) box
        if iou > 0.4:
            best_labels.append(box.extra_fields['indices'][max_idx][1].item())   # best_label (noun phrase, 100)
            confidence_scores.append(box.extra_fields['scores'][max_idx])
        else:   # if iou < a threshold, probably glip cannot generate the true box
            best_labels.append(-1)
            confidence_scores.append(0.)

    # select the most frequent label (noun) in the training sample
    best_label = max(best_labels, key=best_labels.count)

    # if glip cannot generate the true box, select the first noun phrase by default
    # from two aspects: (iou with gt box) (correspondence with noun phrase)
    if best_label == -1:
        best_label = 0
    else:
        best_idx = [confidence_scores[i] for i, b in enumerate(best_labels) if b == (best_label)]
        if max(best_idx).item() < 0.4:
            best_label = 0
    
    # +1: positive_map is a dict, whose keys start from 1 (the first noun phrase is indexed by 1 not 0)
    pseudo_label[positive_map[best_label+1][0]:(positive_map[best_label+1][-1]+1)] = 1

    del tboxes

    return pseudo_label.long()
