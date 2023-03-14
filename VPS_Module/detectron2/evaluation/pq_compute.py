import os, sys
import numpy as np
import json
import time
from datetime import timedelta
from collections import defaultdict
import argparse
import multiprocessing

import PIL.Image as Image

from panopticapi.utils import get_traceback, rgb2id

OFFSET = 256 * 256 * 256
VOID = 0

class PQStatCat():
        def __init__(self):
            self.iou = 0.0
            self.tp = 0
            self.fp = 0
            self.fn = 0

        def __iadd__(self, pq_stat_cat):
            self.iou += pq_stat_cat.iou
            self.tp += pq_stat_cat.tp
            self.fp += pq_stat_cat.fp
            self.fn += pq_stat_cat.fn
            return self


class PQStat():
    def __init__(self):
        self.pq_per_cat = defaultdict(PQStatCat)

    def __getitem__(self, i):
        return self.pq_per_cat[i]

    def __iadd__(self, pq_stat):
        for label, pq_stat_cat in pq_stat.pq_per_cat.items():
            self.pq_per_cat[label] += pq_stat_cat
        return self

    def pq_average(self, categories, isthing):
        pq, sq, rq, n = 0, 0, 0, 0
        per_class_results = {}
        
        for label, label_info in categories.items():
            if isthing is not None:
                cat_isthing = label_info['isthing'] == 1
                if isthing != cat_isthing:
                    continue
            iou = self.pq_per_cat[label].iou
            tp = self.pq_per_cat[label].tp
            fp = self.pq_per_cat[label].fp
            fn = self.pq_per_cat[label].fn
            if tp + fp + fn == 0:
                per_class_results[label] = {'pq': 0.0, 'sq': 0.0, 'rq': 0.0}
                continue
            n += 1
            pq_class = iou / (tp + 0.5 * fp + 0.5 * fn)
            sq_class = iou / tp if tp != 0 else 0
            rq_class = tp / (tp + 0.5 * fp + 0.5 * fn)
            per_class_results[label] = {'pq': pq_class, 'sq': sq_class, 'rq': rq_class, 'iou':iou, 
                                        'tp': tp, 'fp': fp, 'fn': fn}
            pq += pq_class
            sq += sq_class
            rq += rq_class
        
        return {'pq': pq / n, 'sq': sq / n, 'rq': rq / n, 'n': n}, per_class_results


@get_traceback
def pq_compute_single_core(proc_id, annotation_set, gt_folder, pred_folder, categories):
    pq_stat = PQStat()

    idx = 0
    for gt_ann, pred_ann in annotation_set:
        if idx % 100 == 0:
            print('Core: {}, {} from {} images processed'.format(proc_id, idx, len(annotation_set)))
        idx += 1

        pan_gt = np.array(Image.open(os.path.join(gt_folder, gt_ann['file_name'])), dtype=np.uint32)
        pan_gt = rgb2id(pan_gt)
        pan_pred = np.array(Image.open(os.path.join(pred_folder, pred_ann['file_name'])), dtype=np.uint32)
        pan_pred = rgb2id(pan_pred)

        gt_segms = {el['id']: el for el in gt_ann['segments_info']}
        pred_segms = {el['id']: el for el in pred_ann['segments_info']}

        # predicted segments area calculation + prediction sanity checks
        pred_labels_set = set(el['id'] for el in pred_ann['segments_info'])
        labels, labels_cnt = np.unique(pan_pred, return_counts=True) # 返回预测的segm_map上存在的label数，和每个label的个数（即masks的大小）
        # print('pan_pred_label: ',labels)
        for label, label_cnt in zip(labels, labels_cnt):
            if label not in pred_segms:
                if label == VOID:
                    continue
                raise KeyError('In the image with ID {} segment with ID {} is presented in PNG and not presented in JSON.'.format(gt_ann['image_id'], label))
            pred_segms[label]['area'] = label_cnt
            pred_labels_set.remove(label)
            if pred_segms[label]['category_id'] not in categories:
                raise KeyError('In the image with ID {} segment with ID {} has unknown category_id {}.'.format(gt_ann['image_id'], label, pred_segms[label]['category_id']))
        if len(pred_labels_set) != 0:
            raise KeyError('In the image with ID {} the following segment IDs {} are presented in JSON and not presented in PNG.'.format(gt_ann['image_id'], list(pred_labels_set)))

        # confusion matrix calculation
        pan_gt_pred = pan_gt.astype(np.uint64) * OFFSET + pan_pred.astype(np.uint64)
        gt_pred_map = {}
        labels, labels_cnt = np.unique(pan_gt_pred, return_counts=True)
        for label, intersection in zip(labels, labels_cnt):
            gt_id = label // OFFSET
            pred_id = label % OFFSET
            gt_pred_map[(gt_id, pred_id)] = intersection # 一个例子:(15213556.0, 31.0): 48
        # count all matched pairs  === TP ===
        gt_matched = set()
        pred_matched = set()
        for label_tuple, intersection in gt_pred_map.items():
            gt_label, pred_label = label_tuple
            if gt_label not in gt_segms:
                continue
            if pred_label not in pred_segms:
                continue
            if gt_segms[gt_label]['iscrowd'] == 1:
                continue
            if gt_segms[gt_label]['category_id'] != pred_segms[pred_label]['category_id']:
                continue

            union = pred_segms[pred_label]['area'] + gt_segms[gt_label]['area'] \
                        - intersection - gt_pred_map.get((VOID, pred_label), 0)
            iou = intersection / union
            if iou > 0.5:
                pq_stat[gt_segms[gt_label]['category_id']].tp += 1
                pq_stat[gt_segms[gt_label]['category_id']].iou += iou
                gt_matched.add(gt_label)
                pred_matched.add(pred_label)

        # count false positives  === FP ===
        crowd_labels_dict = {}
        for gt_label, gt_info in gt_segms.items():
            if gt_label in gt_matched:
                continue
            # crowd segments are ignored
            if gt_info['iscrowd'] == 1:
                crowd_labels_dict[gt_info['category_id']] = gt_label
                continue
            pq_stat[gt_info['category_id']].fn += 1

        # count false positives   === FP ===
        for pred_label, pred_info in pred_segms.items():
            if pred_label in pred_matched:
                continue
            # intersection of the segment with VOID
            intersection = gt_pred_map.get((VOID, pred_label), 0)
            # plus intersection with corresponding CROWD region if it exists
            if pred_info['category_id'] in crowd_labels_dict:
                intersection += gt_pred_map.get((crowd_labels_dict[pred_info['category_id']], pred_label), 0)
            # predicted segment is ignored if more than half of the segment correspond to VOID and CROWD regions
            if intersection / pred_info['area'] > 0.5:
                continue
            pq_stat[pred_info['category_id']].fp += 1

    print('Core: {}, all {} images processed'.format(proc_id, len(annotation_set)))
    return pq_stat


def pq_compute_multi_core(matched_annotations_list, gt_folder, pred_folder, categories):
    cpu_num = int(multiprocessing.cpu_count() // 2)
    annotations_split = np.array_split(matched_annotations_list, cpu_num)
    print("Number of cores: {}, images per core: {}\n".format(cpu_num, len(annotations_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    for proc_id, annotation_set in enumerate(annotations_split):
        p = workers.apply_async(pq_compute_single_core,
                                (proc_id, annotation_set, gt_folder, pred_folder, categories))
        processes.append(p)
    pq_stat = PQStat()
    for p in processes:
        pq_stat += p.get()
    workers.terminate()
    workers.join()
    return pq_stat


def pq_compute(gt_json_file, pred_json_file, gt_folder=None, pred_folder=None):

    start_time = time.time()
    with open(gt_json_file, 'r') as f:
        gt_json = json.load(f)
    with open(pred_json_file, 'r') as f:
        pred_json = json.load(f)

    if gt_folder is None:
        gt_folder = gt_json_file.replace('.json', '')
    if pred_folder is None:
        pred_folder = pred_json_file.replace('.json', '')
    if 'kitti_panoptic' in gt_json_file:
        categories = {el['trainId']: el for el in gt_json['categories']}
    else:
        categories = {el['id']: el for el in gt_json['categories']}
    print("---------------------------------",len(categories))

    print("Evaluation panoptic segmentation metrics:")
    print("Ground truth:")
    print("\tSegmentation folder: {}".format(gt_folder))
    print("\tJSON file: {}".format(gt_json_file))
    print("Prediction:")
    print("\tSegmentation folder: {}".format(pred_folder))
    print("\tJSON file: {}".format(pred_json_file))

    if not os.path.isdir(gt_folder):
        raise Exception("Folder {} with ground truth segmentations doesn't exist".format(gt_folder))
    if not os.path.isdir(pred_folder):
        raise Exception("Folder {} with predicted segmentations doesn't exist".format(pred_folder))

    pred_annotations = {el['image_id']: el for el in pred_json['annotations']}
    matched_annotations_list = []
    for gt_ann in gt_json['annotations']:
        image_id = gt_ann['image_id']
        if image_id not in pred_annotations:
            raise Exception('no prediction for the image with id: {}'.format(image_id))
        matched_annotations_list.append((gt_ann, pred_annotations[image_id]))

    pq_stat = pq_compute_multi_core(matched_annotations_list, gt_folder, pred_folder, categories)
    metrics = [("All", None), ("Things", True), ("Stuff", False)]
    results = {}
    print("------------------------------------------------")
    for name, isthing in metrics:
        results[name], per_class_results = pq_stat.pq_average(categories, isthing=isthing)
        if name != 'All':
            for k,v in per_class_results.items():
                print(k,"-",categories[k]['name']," : ",v)
            print("------------------------------------------------")
        if name == 'All':
            results['per_class'] = per_class_results
    print("{:10s}| {:>5s}  {:>5s}  {:>5s} {:>5s}".format("", "PQ", "SQ", "RQ", "N"))
    print("-" * (10 + 7 * 4))

    for name, _isthing in metrics:
        print("{:10s}| {:5.1f}  {:5.1f}  {:5.1f} {:5d}".format(
            name,
            100 * results[name]['pq'],
            100 * results[name]['sq'],
            100 * results[name]['rq'],
            results[name]['n'])
        )

    t_delta = time.time() - start_time
    print("Time elapsed: {:0.2f} seconds".format(t_delta))

    return results


def vpq_compute_single_core(gt_pred_set, categories, nframes=2):
    OFFSET = 256 * 256 * 256
    VOID = 255
    vpq_stat = PQStat()

    # Iterate over the video frames 0::T-λ
    for idx in range(0, len(gt_pred_set)-nframes+1): 
        vid_pan_gt, vid_pan_pred = [], []
        gt_segms_list, pred_segms_list = [], []

        # Matching nframes-long tubes.
        # Collect tube IoU, TP, FP, FN
        for i, (gt_json, pred_json, gt_pan, pred_pan, gt_image_json) in enumerate(gt_pred_set[idx:idx+nframes]):
            #### Step1. Collect frame-level pan_gt, pan_pred, etc.
            gt_pan, pred_pan = np.uint32(gt_pan), np.uint32(pred_pan)
            pan_gt = gt_pan[:, :, 0] + gt_pan[:, :, 1] * 256 + gt_pan[:, :, 2] * 256 * 256
            pan_pred = pred_pan[:, :, 0] + pred_pan[:, :, 1] * 256 + pred_pan[:, :, 2] * 256 * 256
            gt_segms = {}
            for el in gt_json['segments_info']:
                if el['id'] in gt_segms:
                    gt_segms[el['id']]['area'] += el['area']
                else:
                    gt_segms[el['id']] = copy.deepcopy(el)
            pred_segms = {}
            for el in pred_json['segments_info']:
                if el['id'] in pred_segms:
                    pred_segms[el['id']]['area'] += el['area']
                else:
                    pred_segms[el['id']] = copy.deepcopy(el)
            # predicted segments area calculation + prediction sanity checks
            pred_labels_set = set(el['id'] for el in pred_json['segments_info'])
            labels, labels_cnt = np.unique(pan_pred, return_counts=True)
            for label, label_cnt in zip(labels, labels_cnt):
                if label not in pred_segms:
                    if label == VOID:
                        continue
                    raise KeyError('Segment with ID {} is presented in PNG and not presented in JSON.'.format(label))
                pred_segms[label]['area'] = label_cnt
                pred_labels_set.remove(label)
                if pred_segms[label]['category_id'] not in categories:
                    raise KeyError('Segment with ID {} has unknown category_id {}.'.format(label, pred_segms[label]['category_id']))
            if len(pred_labels_set) != 0:
                raise KeyError(
                    'The following segment IDs {} are presented in JSON and not presented in PNG.'.format(list(pred_labels_set)))

            vid_pan_gt.append(pan_gt)
            vid_pan_pred.append(pan_pred)
            gt_segms_list.append(gt_segms)
            pred_segms_list.append(pred_segms)

        #### Step 2. Concatenate the collected items -> tube-level. 
        vid_pan_gt = np.stack(vid_pan_gt) # [nf,H,W]
        vid_pan_pred = np.stack(vid_pan_pred) # [nf,H,W]
        vid_gt_segms, vid_pred_segms = {}, {}
        for gt_segms, pred_segms in zip(gt_segms_list, pred_segms_list):
            # aggregate into tube 'area'
            for k in gt_segms.keys():
                if not k in vid_gt_segms:
                    vid_gt_segms[k] = gt_segms[k]
                else:
                    vid_gt_segms[k]['area'] += gt_segms[k]['area']
            for k in pred_segms.keys():
                if not k in vid_pred_segms:
                    vid_pred_segms[k] = pred_segms[k]
                else:
                    vid_pred_segms[k]['area'] += pred_segms[k]['area']

        #### Step3. Confusion matrix calculation
        vid_pan_gt_pred = vid_pan_gt.astype(np.uint64) * OFFSET + vid_pan_pred.astype(np.uint64)
        gt_pred_map = {}
        labels, labels_cnt = np.unique(vid_pan_gt_pred, return_counts=True)
        for label, intersection in zip(labels, labels_cnt):
            gt_id = label // OFFSET
            pred_id = label % OFFSET
            gt_pred_map[(gt_id, pred_id)] = intersection

        # count all matched pairs
        gt_matched = set()
        pred_matched = set()
        tp = 0
        fp = 0
        fn = 0

        #### Step4. Tube matching
        for label_tuple, intersection in gt_pred_map.items():
            gt_label, pred_label = label_tuple

            if gt_label not in vid_gt_segms:
                continue
            if pred_label not in vid_pred_segms:
                continue
            if vid_gt_segms[gt_label]['iscrowd'] == 1:
                continue
            if vid_gt_segms[gt_label]['category_id'] != \
                    vid_pred_segms[pred_label]['category_id']:
                continue

            union = vid_pred_segms[pred_label]['area'] + vid_gt_segms[gt_label]['area'] - intersection - gt_pred_map.get(
                (VOID, pred_label), 0)
            iou = intersection / union
            assert iou <= 1.0, 'INVALID IOU VALUE : %d'%(gt_label)
            # count true positives
            if iou > 0.5:
                vpq_stat[vid_gt_segms[gt_label]['category_id']].tp += 1
                vpq_stat[vid_gt_segms[gt_label]['category_id']].iou += iou
                gt_matched.add(gt_label)
                pred_matched.add(pred_label)
                tp += 1

        # count false negatives
        crowd_labels_dict = {}
        for gt_label, gt_info in vid_gt_segms.items():
            if gt_label in gt_matched:
                continue
            # crowd segments are ignored
            if gt_info['iscrowd'] == 1:
                crowd_labels_dict[gt_info['category_id']] = gt_label
                continue
            vpq_stat[gt_info['category_id']].fn += 1
            fn += 1

        # count false positives
        for pred_label, pred_info in vid_pred_segms.items():
            if pred_label in pred_matched:
                continue
            # intersection of the segment with VOID
            intersection = gt_pred_map.get((VOID, pred_label), 0)
            # plus intersection with corresponding CROWD region if it exists
            if pred_info['category_id'] in crowd_labels_dict:
                intersection += gt_pred_map.get((crowd_labels_dict[pred_info['category_id']], pred_label), 0)
            # predicted segment is ignored if more than half of the segment correspond to VOID and CROWD regions
            if intersection / pred_info['area'] > 0.5:
                continue
            vpq_stat[pred_info['category_id']].fp += 1
            fp += 1

    return vpq_stat

def vpq_compute(gt_pred_split, categories, nframes, output_dir):
    start_time = time.time()
    vpq_stat = PQStat()
    for idx, gt_pred_set in enumerate(gt_pred_split):
        tmp = vpq_compute_single_core(gt_pred_set, categories, nframes=nframes)
        vpq_stat += tmp

    # hyperparameter: window size k
    k = (nframes-1)*5
    print('==> %d-frame vpq_stat:'%(k), time.time()-start_time, 'sec')
    metrics = [("All", None), ("Things", True), ("Stuff", False)]
    results = {}
    for name, isthing in metrics:
        results[name], per_class_results = vpq_stat.pq_average(categories, isthing=isthing)
        if name == 'All':
            results['per_class'] = per_class_results

    vpq_all = 100 * results['All']['pq']
    vpq_thing = 100 * results['Things']['pq']
    vpq_stuff = 100 * results['Stuff']['pq']

    save_name = os.path.join(output_dir, 'vpq-%d.txt'%(k))
    f = open(save_name, 'w') if save_name else None
    f.write("================================================\n")
    f.write("{:10s}| {:>5s}  {:>5s}  {:>5s} {:>5s}".format("", "PQ", "SQ", "RQ", "N\n"))
    f.write("-" * (10 + 7 * 4)+'\n')
    for name, _isthing in metrics:
        f.write("{:10s}| {:5.1f}  {:5.1f}  {:5.1f} {:5d}\n".format(name, 100 * results[name]['pq'], 100 * results[name]['sq'], 100 * results[name]['rq'], results[name]['n']))
    f.write("{:4s}| {:>5s} {:>5s} {:>5s} {:>6s} {:>7s} {:>7s} {:>7s}\n".format("IDX", "PQ", "SQ", "RQ", "IoU", "TP", "FP", "FN"))
    for idx, result in results['per_class'].items():
        f.write("{:4d} | {:5.1f} {:5.1f} {:5.1f} {:6.1f} {:7d} {:7d} {:7d}\n".format(idx, 100 * result['pq'], 100 * result['sq'], 100 * result['rq'], result['iou'], result['tp'], result['fp'], result['fn']))
    if save_name:
        f.close()

    return vpq_all, vpq_thing, vpq_stuff

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_json_file', type=str,
                        help="JSON file with ground truth data")
    parser.add_argument('--pred_json_file', type=str,
                        help="JSON file with predictions data")
    parser.add_argument('--gt_folder', type=str, default=None,
                        help="Folder with ground turth COCO format segmentations. \
                              Default: X if the corresponding json file is X.json")
    parser.add_argument('--pred_folder', type=str, default=None,
                        help="Folder with prediction COCO format segmentations. \
                              Default: X if the corresponding json file is X.json")
    args = parser.parse_args()
    pq_compute(args.gt_json_file, args.pred_json_file, args.gt_folder, args.pred_folder)
