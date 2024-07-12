# Copyright (C) 2024 Xiaomi Corporation.

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, 
# software distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and limitations under the License.

import torch
import math
from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS
import numpy as np
from mmdet.core.bbox.transforms import bbox_cxcywh_to_xyxy
import copy
import networkx as nx
from shapely.geometry import LineString


def denormalize_2d_bbox(bboxes, pc_range):

    bboxes = bbox_cxcywh_to_xyxy(bboxes)
    bboxes[..., 0::2] = (bboxes[..., 0::2]*(pc_range[3] -
                            pc_range[0]) + pc_range[0])
    bboxes[..., 1::2] = (bboxes[..., 1::2]*(pc_range[4] -
                            pc_range[1]) + pc_range[1])

    return bboxes


def denormalize_2d_pts(pts, pc_range):
    new_pts = pts.clone()
    new_pts[...,0:1] = (pts[..., 0:1]*(pc_range[3] -
                            pc_range[0]) + pc_range[0])
    new_pts[...,1:2] = (pts[...,1:2]*(pc_range[4] -
                            pc_range[1]) + pc_range[1])
    return new_pts



@BBOX_CODERS.register_module()
class CGNetNMSFreeCoder(BaseBBoxCoder):

    def __init__(self,
                 pc_range,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 num_classes=10,
                 adj_threshold=0.9, 
                 vis_mode=False, 
                 simple_distance=0.2, 
                 decode_lvl=-1,
                 *args, 
                 **kwargs):
        super(CGNetNMSFreeCoder, self).__init__(*args, **kwargs)
        self.pc_range = pc_range
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes
        self.adj_threshold = adj_threshold
        self.vis_mode = vis_mode
        self.simple_distance = simple_distance
        self.decode_lvl = decode_lvl

    def decode_single(self, cls_scores, bbox_preds, pts_preds, sm_preds):

        max_num = self.max_num

        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.view(-1).topk(max_num)
        labels = indexs % self.num_classes
        bbox_index = indexs // self.num_classes
        bbox_preds = bbox_preds[bbox_index]
        pts_preds = pts_preds[bbox_index]
       
        final_box_preds = denormalize_2d_bbox(bbox_preds, self.pc_range) 
        final_pts_preds = denormalize_2d_pts(pts_preds, self.pc_range) #num_q,num_p,2
        # final_box_preds = bbox_preds 
        final_scores = scores 
        final_preds = labels 

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
            tmp_score = self.score_threshold
            while thresh_mask.sum() == 0:
                tmp_score *= 0.9
                if tmp_score < 0.01:
                    thresh_mask = final_scores > -1
                    break
                thresh_mask = final_scores >= tmp_score

        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(
                self.post_center_range, device=scores.device)
            mask = (final_box_preds[..., :4] >=
                    self.post_center_range[:4]).all(1)
            mask &= (final_box_preds[..., :4] <=
                     self.post_center_range[4:]).all(1)

            if self.score_threshold:
                mask &= thresh_mask

            scores = final_scores[mask]
            pts = final_pts_preds[mask]
            labels = final_preds[mask]
            index = bbox_index[mask]
            sm = sm_preds[index][:, index]
            
            am = torch.where(sm > self.adj_threshold, 1, 0)
            am = (1 - torch.eye(len(am)).to(am.device)) * am

            for i in range(len(pts)):
                connection = torch.where(am[i]==1)[0]
                if len(connection)>0:
                    val = (pts[i][-1] + pts[connection][:, 0, :].sum(0)) / (len(connection)+1)
                    pts[i][-1] = val
                    for idx in connection:
                        pts[idx][0] = val

            if self.vis_mode:
                simple_pts = []
                line_length = []
                for i in range(len(pts)):
                    pts_i = LineString(pts[i].cpu().numpy())
                    pts_i = pts_i.simplify(self.simple_distance, preserve_topology=True).coords
                    line_length.append(len(pts_i))
                    simple_pts.extend(pts_i)
                instance_points = torch.from_numpy(np.array(simple_pts)).cuda()
            else:
                instance_points = pts.reshape(-1, 2)
                line_length = [len(line) for line in pts]
            nums_pts = instance_points.shape[0]
            adj_matrix = np.zeros((nums_pts, nums_pts))
            start_loc = [0] + list(np.cumsum(line_length)[:-1])
            for i in range(len(line_length)):
                index = start_loc[i] + np.arange(line_length[i])
                for j in range(1, len(index)):
                        adj_matrix[index[j-1], index[j]] = 1


            remove_list = []
            _, indices = torch.unique(instance_points, dim=0, return_inverse=True)
            indices = indices.cpu().numpy()
            duplicate = list(set([x for x in list(indices) if list(indices).count(x) > 1]))
            for dup in duplicate:
                dup_pts = np.where(indices == dup)[0]
                remove_list.extend(list(dup_pts[1:]))
                for idx in dup_pts[1:]:
                    adj_matrix[dup_pts[0]] += adj_matrix[idx]
                    adj_matrix[:, dup_pts[0]] += adj_matrix[:, idx]

            if remove_list:
                remove_list = np.array(remove_list)
                adj_matrix = np.delete(adj_matrix, remove_list, axis=0)
                adj_matrix = np.delete(adj_matrix, remove_list, axis=1)
            
                adj_matrix = np.clip(adj_matrix, 0, 1)
                new_instance_points = instance_points.cpu().numpy()
                new_instance_points = np.delete(new_instance_points, remove_list, axis=0)

            else:
                new_instance_points = instance_points.cpu().numpy()

            G = nx.DiGraph(adj_matrix)
            for i, node in enumerate(G.nodes()):
                G.nodes[node]['pos'] = new_instance_points[i]     

            predictions_dict = {
                'line_scores': scores,
                'line_labels': labels,
                'line': pts.cpu(),
                'pts_scores': torch.ones(len(new_instance_points)).long(),
                'pts_labels': torch.zeros(len(new_instance_points)).long(),
                'pts': torch.from_numpy(new_instance_points),
                'graph': G, 
                'adj_matrix':sm
            }

        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')
        return predictions_dict

    def encode(self):
        pass

    def decode(self, preds_dicts):

        all_cls_scores = preds_dicts['all_cls_scores'][self.decode_lvl]
        all_bbox_preds = preds_dicts['all_bbox_preds'][self.decode_lvl]
        all_pts_preds = preds_dicts['all_pts_preds'][self.decode_lvl]
        all_sms_preds = preds_dicts['all_sms_preds'][self.decode_lvl]
        batch_size = all_cls_scores.size()[0]
        predictions_list = []

        for i in range(batch_size):
            predictions_list.append(self.decode_single(all_cls_scores[i], all_bbox_preds[i],
                                                        all_pts_preds[i], all_sms_preds[i]))
        return predictions_list