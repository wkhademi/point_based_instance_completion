import trimesh
import numpy as np

import multiprocessing
from pykeops.numpy import LazyTensor


def chamfer_distance(x, y):
    x_i = LazyTensor(x[:, None, :])
    y_j = LazyTensor(y[None, ...])
    D_ij = ((x_i - y_j)**2).sum(-1)**0.5
    chamfer_dist = (np.mean(D_ij.min(dim=0, backend="CPU")) + np.mean(D_ij.min(dim=1, backend="CPU"))) / 2.
        
    return chamfer_dist


def sample_and_calc_chamfer(pred, gt):
    # pred, gt: trimesh
    pred_points, _ = trimesh.sample.sample_surface(pred, 4096)
    gt_points, _ = trimesh.sample.sample_surface(gt, 4096)
    x = chamfer_distance(np.array(pred_points), np.array(gt_points))
    return x


def voc_ap(rec, prec, use_07_metric):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def eval_det_cls_w_mesh(queue, pred, gt, thresh, use_07_metric):

    # per-class! pred = {scan_id: [bbox, score, mesh]}
    
    # construct gt
    npos = 0
    all_stats = {} # {scan_id: {'bbox': bbox list, 'det': matched list}}
    for scan_id in gt.keys():
        mesh = gt[scan_id]
        matched = [False] * len(mesh)
        npos += len(mesh)
        all_stats[scan_id] = {'meshes_gt': mesh, 'matched': matched}
    
    # if pred has a scan_id not in gt, also add a dummy gt.
    for scan_id in pred.keys():
        if scan_id not in gt:
            all_stats[scan_id] = {'meshes_gt': [], 'matched': []}

    # construct preds
    scan_ids = []
    confidence = []
    meshes_pred = []

    for scan_id in pred.keys():
        for mesh, score in pred[scan_id]:
            scan_ids.append(scan_id)
            meshes_pred.append(mesh)
            confidence.append(score)

    confidence = np.array(confidence)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    meshes_pred = [meshes_pred[x] for x in sorted_ind]
    scan_ids = [scan_ids[x] for x in sorted_ind]

    # go down preds and mark TPs and FPs
    nd = len(meshes_pred) # number of predicted meshes
    tp_mesh_list = np.zeros(nd) # 0/1 if TP
    fp_mesh_list = np.zeros(nd) # 0/1 if FP
    tp_mesh_iou = np.zeros(nd) # float, the best IoU if TP
    
    # queue = multiprocessing.Queue()

    # for all detected bboxes
    for d in range(nd):
        #if d % 100 == 0: print(d)
        print('[eval mesh]', d + 1, '/', nd)

        mesh_pred = meshes_pred[d]

        # compare with per-GT
        best_val = np.inf
        stats = all_stats[scan_ids[d]] # current scan's stats
        meshes_gt = stats['meshes_gt'] # all gt meshes in this scan
        if len(meshes_gt) > 0:
            # compute overlaps
            for j in range(len(meshes_gt)):
                
                # metric is implemented here.
                val = sample_and_calc_chamfer(mesh_pred, meshes_gt[j])
                # p = multiprocessing.Process(target=sample_and_calc_chamfer, args=(queue, mesh_pred, meshes_gt[j]))
                # p.start()
                # p.join()
                # val = queue.get()

                if val < best_val:
                    best_val = val
                    best_idx = j

            print(f'[eval mesh] best mesh CD {best_val}, from {best_idx} in {j}')

        if best_val < thresh and not stats['matched'][best_idx]:
            tp_mesh_list[d] = 1
            tp_mesh_iou[d] = best_val
            stats['matched'][best_idx] = 1
            print(f'[eval mesh] mesh TP ++')
        else:
            fp_mesh_list[d] = 1


    # for mesh
    fp_mesh_cumsum = np.cumsum(fp_mesh_list)
    tp_mesh_cumsum = np.cumsum(tp_mesh_list)
    rec_mesh = tp_mesh_cumsum / np.maximum(npos, np.finfo(np.float64).eps)
    prec_mesh = tp_mesh_cumsum / np.maximum(tp_mesh_cumsum + fp_mesh_cumsum, np.finfo(np.float64).eps)
    ap_mesh = voc_ap(rec_mesh, prec_mesh, use_07_metric)
    
    tp_mesh = tp_mesh_cumsum[-1]
    fp_mesh = fp_mesh_cumsum[-1]
    fn_mesh = npos - tp_mesh
    RQ_mesh = tp_mesh / (tp_mesh + fp_mesh/2 + fn_mesh/2)
    SQ_mesh = np.sum(tp_mesh_iou) / np.maximum(tp_mesh, np.finfo(np.float64).eps)
    PQ_mesh = SQ_mesh * RQ_mesh

    print(f'[eval mesh] tp = {tp_mesh}, fp = {fp_mesh}, fn = {fn_mesh}, npos = {npos}')

    # return (rec_mesh[-1], prec_mesh[-1], ap_mesh), (PQ_mesh, SQ_mesh, RQ_mesh)
    queue.put([(rec_mesh[-1], prec_mesh[-1], ap_mesh), (PQ_mesh, SQ_mesh, RQ_mesh)])


def eval_det_cls_wrapper_w_mesh(arguments):
    # pred: {scan_id: (label, score, mesh)}
    # gt: {scan_id: (label, mesh)}
    pred, gt, thresh, use_07_metric = arguments
    # (rec_mesh, prec_mesh, ap_mesh), (PQ_mesh, SQ_mesh, RQ_mesh) = eval_det_cls_w_mesh(pred, gt, thresh, use_07_metric)
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=eval_det_cls_w_mesh, args=(queue, pred, gt, thresh, use_07_metric))
    p.start()
    p.join()
    (rec_mesh, prec_mesh, ap_mesh), (PQ_mesh, SQ_mesh, RQ_mesh) = queue.get()
    return (rec_mesh, prec_mesh, ap_mesh), (PQ_mesh, SQ_mesh, RQ_mesh)

# RfD use 07
def eval_det_multiprocessing_w_mesh(pred_all, gt_all, thresh=0.25, use_07_metric=True):
    """ Generic functions to compute precision/recall for object detection
        for multiple classes.
        Input:
            pred_all: map of {scan_id: [(label, score, mesh)]}
            gt_all: map of {scan_id: [(label, mesh)]}
            thresh: scalar, CD threshold
            use_07_metric: bool, if true use VOC07 11 point method
        Output:
            rec: {label: rec}
            prec: {label: prec_all}
            ap: {label: scalar}
    """
    pred = {}  # map {label: pred}
    gt = {}  # map {label: gt}

    # scan_id == scan_id
    for scan_id in pred_all.keys():
        for label, mesh, score in pred_all[scan_id]:
            # default dict behaviour
            if label not in pred: pred[label] = {}
            if scan_id not in pred[label]: pred[label][scan_id] = []
            if label not in gt: gt[label] = {}
            if scan_id not in gt[label]: gt[label][scan_id] = []
            # record
            pred[label][scan_id].append((mesh, score))

    for scan_id in gt_all.keys():
        for label, mesh in gt_all[scan_id]:
            # default dict behaviour
            if label not in gt: gt[label] = {}
            if scan_id not in gt[label]: gt[label][scan_id] = []
            # record
            gt[label][scan_id].append(mesh)

    rec_mesh = {}
    prec_mesh = {}
    ap_mesh = {}

    PQ_mesh = {}
    SQ_mesh = {}
    RQ_mesh = {}

    # fallback to single-thread
    ret_values = []
    for label in gt.keys():
        if label not in pred: continue
        
        ret_value = eval_det_cls_wrapper_w_mesh((pred[label], gt[label], thresh, use_07_metric))
        ret_values.append(ret_value)

    for i, label in enumerate(gt.keys()):
        if label in pred:
            (rec_mesh[label], prec_mesh[label], ap_mesh[label]), (PQ_mesh[label], SQ_mesh[label], RQ_mesh[label]) = ret_values[i]
        else:
            (rec_mesh[label], prec_mesh[label], ap_mesh[label]), (PQ_mesh[label], SQ_mesh[label], RQ_mesh[label]) = (0, 0, 0), (0, 0, 0)

    return (rec_mesh, prec_mesh, ap_mesh), (PQ_mesh, SQ_mesh, RQ_mesh)


###################
# mAP calculation

class APCalculator(object):
    

    def __init__(self, ap_iou_thresh=0.25, class2type_map=None):
        self.ap_iou_thresh = ap_iou_thresh
        self.class2type_map = class2type_map
        self.reset()

    def step(self, info_mesh_preds, info_mesh_gts):
        self.pred_map_cls[self.scan_cnt] = info_mesh_preds
        self.gt_map_cls[self.scan_cnt] = info_mesh_gts
        self.scan_cnt += 1

    def compute_metrics(self):
        
        (rec_mesh, prec_mesh, ap_mesh), (PQ_mesh, SQ_mesh, RQ_mesh) = eval_det_multiprocessing_w_mesh(self.pred_map_cls, self.gt_map_cls, thresh=self.ap_iou_thresh)

        ret_dict = {}
       
        # for mesh
        for key in sorted(ap_mesh.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            ret_dict['AP_mesh        %s' % (clsname)] = ap_mesh[key]
            ret_dict['Precision_mesh %s' % (clsname)] = prec_mesh[key]
            ret_dict['Recall_mesh    %s' % (clsname)] = rec_mesh[key]
        ret_dict['mAP_mesh'] = np.mean(list(ap_mesh.values()))
        ret_dict['mean Precision_mesh'] = np.mean(list(prec_mesh.values()))
        ret_dict['mean Recall_mesh'] = np.mean(list(rec_mesh.values()))

        # for PQ
        for key in sorted(PQ_mesh.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            ret_dict['PQ_mesh %s' % (clsname)] = PQ_mesh[key]
            ret_dict['SQ_mesh %s' % (clsname)] = SQ_mesh[key]
            ret_dict['RQ_mesh %s' % (clsname)] = RQ_mesh[key]

        ret_dict['PQ_mesh'] = np.mean(list(PQ_mesh.values()))
        ret_dict['SQ_mesh'] = np.mean(list(SQ_mesh.values()))
        ret_dict['RQ_mesh'] = np.mean(list(RQ_mesh.values()))

        return ret_dict

    def reset(self):
        self.gt_map_cls = {}  
        self.pred_map_cls = {}  
        self.scan_cnt = 0