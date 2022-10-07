import copy
from os import path as osp

import numpy as np
import torch

import mmocr.utils as utils
from mmdet.datasets.builder import DATASETS
from mmocr.core import compute_f1_score
from mmocr.datasets.base_dataset import BaseDataset
from mmocr.datasets.pipelines.crop import sort_vertex


@DATASETS.register_module()
class KIEDataset(BaseDataset):
    """
    Args:
        ann_file (str): Annotation file path.
        pipeline (list[dict]): Processing pipeline.
        loader (dict): Dictionary to construct loader
            to load annotation infos.
        img_prefix (str, optional): Image prefix to generate full
            image path.
        test_mode (bool, optional): If True, try...except will
            be turned off in __getitem__.
        dict_file (str): Character dict file path.
        norm (float): Norm to map value from one range to another.
    """

    def __init__(self,
                 ann_file,
                 loader,
                 dict_file,
                 img_prefix='',
                 pipeline=None,
                 norm=10.,
                 directed=False,
                 test_mode=True,
                 **kwargs):
        super().__init__(
            ann_file,
            loader,
            pipeline,
            img_prefix=img_prefix,
            test_mode=test_mode)
        assert osp.exists(dict_file)

        self.norm = norm
        self.directed = directed

        self.dict = dict({'': 0})
        with open(dict_file, 'r') as fr:
            idx = 1
            for line in fr:
                char = line.strip()
                self.dict[char] = idx
                idx += 1

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['bbox_fields'] = []

    def _parse_anno_info(self, annotations):
        """Parse annotations of boxes, texts and labels for one image.
        Args:
            annotations (list[dict]): Annotations of one image, where
                each dict is for one character.

        Returns:
            dict: A dict containing the following keys:

                - bboxes (np.ndarray): Bbox in one image with shape:
                    box_num * 4.
                - relations (np.ndarray): Relations between bbox with shape:
                    box_num * box_num * D.
                - texts (np.ndarray): Text index with shape:
                    box_num * text_max_len.
                - labels (np.ndarray): Box Labels with shape:
                    box_num * (box_num + 1).
        """

        assert utils.is_type_list(annotations, dict)
        assert 'box' in annotations[0]
        assert 'text' in annotations[0]
        assert 'label' in annotations[0]

        boxes, texts, text_inds, labels, edges = [], [], [], [], []
        for ann in annotations:
            box = ann['box']
            x_list, y_list = box[0:8:2], box[1:9:2]
            sorted_x_list, sorted_y_list = sort_vertex(x_list, y_list)
            sorted_box = []
            for x, y in zip(sorted_x_list, sorted_y_list):
                sorted_box.append(x)
                sorted_box.append(y)
            boxes.append(sorted_box)
            text = ann['text']
            texts.append(ann['text'])
            text_ind = [self.dict[c] for c in text if c in self.dict]
            text_inds.append(text_ind)
            labels.append(ann['label'])
            edges.append(ann.get('edge', 0))

        ann_infos = dict(
            boxes=boxes,
            texts=texts,
            text_inds=text_inds,
            edges=edges,
            labels=labels)

        return self.list_to_numpy(ann_infos)

    def prepare_train_img(self, index):
        """Get training data and annotations from pipeline.

        Args:
            index (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """
        img_ann_info = self.data_infos[index]
        img_info = {
            'filename': img_ann_info['file_name'],
            'height': img_ann_info['height'],
            'width': img_ann_info['width']
        }
        ann_info = self._parse_anno_info(img_ann_info['annotations'])
        results = dict(img_info=img_info, ann_info=ann_info)

        self.pre_pipeline(results)

        return self.pipeline(results)

    def evaluate(self,
                 results,
                 metric='macro_f1',
                 metric_options=dict(macro_f1=dict(ignores=[])),
                 **kwargs):
        # allow some kwargs to pass through
        assert set(kwargs).issubset(['logger'])

        # Protect ``metric_options`` since it uses mutable value as default
        metric_options = copy.deepcopy(metric_options)

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['macro_f1']
        for m in metrics:
            if m not in allowed_metrics:
                raise KeyError(f'metric {m} is not supported')

        return self.compute_macro_f1(results, **metric_options['macro_f1'])

    def compute_macro_f1(self, results, ignores=[]):
        node_preds = []
        node_gts = []
        for idx, result in enumerate(results):
            node_preds.append(result['nodes'])
            box_ann_infos = self.data_infos[idx]['annotations']
            node_gt = [box_ann_info['label'] for box_ann_info in box_ann_infos]
            node_gts.append(torch.Tensor(node_gt))

        node_preds = torch.cat(node_preds)
        node_gts = torch.cat(node_gts).int().to(node_preds.device)

        node_f1s = compute_f1_score(node_preds, node_gts, ignores)

        return {
            'macro_f1': node_f1s.mean(),
        }

    def list_to_numpy(self, ann_infos):
        """Convert bboxes, relations, texts and labels to ndarray."""
        boxes, text_inds = ann_infos['boxes'], ann_infos['text_inds']
        boxes = np.array(boxes, np.int32)
        relations, bboxes = self.compute_relation(boxes)

        labels = ann_infos.get('labels', None)
        if labels is not None:
            labels = np.array(labels, np.int32)
            edges = ann_infos.get('edges', None)
            if edges is not None:
                labels = labels[:, None]
                edges = np.array(edges)
                edges = (edges[:, None] == edges[None, :]).astype(np.int32)
                if self.directed:
                    edges = (edges & labels == 1).astype(np.int32)
                np.fill_diagonal(edges, -1)
                labels = np.concatenate([labels, edges], -1)
        padded_text_inds = self.pad_text_indices(text_inds)

        return dict(
            bboxes=bboxes,
            relations=relations,
            texts=padded_text_inds,
            labels=labels)

    def pad_text_indices(self, text_inds):
        """Pad text index to same length."""
        max_len = max([len(text_ind) for text_ind in text_inds])
        padded_text_inds = -np.ones((len(text_inds), max_len), np.int32)
        for idx, text_ind in enumerate(text_inds):
            padded_text_inds[idx, :len(text_ind)] = np.array(text_ind)
        return padded_text_inds

    def compute_relation(self, boxes):
        """Compute relation between every two boxes."""
        x1s, y1s = boxes[:, 0:1], boxes[:, 1:2]
        x2s, y2s = boxes[:, 4:5], boxes[:, 5:6]
        ws, hs = x2s - x1s + 1, np.maximum(y2s - y1s + 1, 1)
        dxs = (x1s[:, 0][None] - x1s) / self.norm
        dys = (y1s[:, 0][None] - y1s) / self.norm
        xhhs, xwhs = hs[:, 0][None] / hs, ws[:, 0][None] / hs
        whs = ws / hs + np.zeros_like(xhhs)
        relations = np.stack([dxs, dys, whs, xhhs, xwhs], -1)
        bboxes = np.concatenate([x1s, y1s, x2s, y2s], -1).astype(np.float32)
        return relations, bboxes
