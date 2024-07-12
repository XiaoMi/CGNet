# Copyright (C) 2024 Xiaomi Corporation.

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, 
# software distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and limitations under the License.

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmcv.runner import force_fp32, auto_fp16
from mmcv.runner.base_module import BaseModule, Sequential
from mmcv.cnn import Linear, bias_init_with_prob, xavier_init, constant_init
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy
from mmdet.core import (multi_apply, multi_apply, reduce_mean, build_assigner)
from mmcv.utils import TORCH_VERSION, digit_version
import math
import numpy as np
import cv2

def normalize_2d_bbox(bboxes, pc_range):

    patch_h = pc_range[4]-pc_range[1]
    patch_w = pc_range[3]-pc_range[0]
    cxcywh_bboxes = bbox_xyxy_to_cxcywh(bboxes)
    cxcywh_bboxes[...,0:1] = cxcywh_bboxes[..., 0:1] - pc_range[0]
    cxcywh_bboxes[...,1:2] = cxcywh_bboxes[...,1:2] - pc_range[1]
    factor = bboxes.new_tensor([patch_w, patch_h,patch_w,patch_h])

    normalized_bboxes = cxcywh_bboxes / factor
    return normalized_bboxes

def normalize_2d_pts(pts, pc_range):
    patch_h = pc_range[4]-pc_range[1]
    patch_w = pc_range[3]-pc_range[0]
    new_pts = pts.clone()
    new_pts[...,0:1] = pts[..., 0:1] - pc_range[0]
    new_pts[...,1:2] = pts[...,1:2] - pc_range[1]
    factor = pts.new_tensor([patch_w, patch_h])
    normalized_pts = new_pts / factor
    return normalized_pts

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

def pos2posemb2d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x), dim=-1)
    return posemb


@HEADS.register_module()
class CGTopoHead(DETRHead):

    def __init__(self,
                *args,
                with_box_refine=False,
                as_two_stage=False,
                transformer=None,
                bbox_coder=None,
                num_cls_fcs=2,
                code_weights=None,
                bev_h=30,
                bev_w=30,
                num_vec=20,
                num_pts_per_vec=2,
                num_pts_per_gt_vec=2,
                query_embed_type='all_pts',
                transform_method='minmax',
                gt_shift_pts_pattern='v0',
                dir_interval=1,
                loss_pts=None,
                loss_dir=None,
                loss_ctp=None,
                nums_ctp=4,
                edge_weight=0.8,
                dilate_radius=9,
                weight_kp=0.1,
                weight_adj=1,
                **kwargs):
        
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.bev_encoder_type = transformer.encoder.type
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1
        

        self.query_embed_type = query_embed_type
        self.transform_method = transform_method
        self.gt_shift_pts_pattern = gt_shift_pts_pattern
        num_query = num_vec * num_pts_per_vec
        self.num_query = num_query
        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec
        self.num_pts_per_gt_vec = num_pts_per_gt_vec
        self.dir_interval = dir_interval

        super(CGTopoHead, self).__init__(
            *args, transformer=transformer, **kwargs)
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)
        self.loss_pts = build_loss(loss_pts)
        self.loss_dir = build_loss(loss_dir)
        num_query = num_vec * num_pts_per_vec
        self.num_query = num_query
        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec
        self.num_pts_per_gt_vec = num_pts_per_gt_vec
        self._init_layers()

        self.nums_ctp = nums_ctp
        self.loss_adj = nn.BCELoss()
        self.loss_ctp = build_loss(loss_ctp)

        self.vertex_inteact = GNN(self.embed_dims, self.embed_dims*2, edge_weight=edge_weight)
        self.GRU = GRU(self.embed_dims, self.embed_dims)

        self.lclc_branch = TopologyHead(self.embed_dims)

        self.inv_B = self.get_inv_bernstein_basis(self.num_pts_per_vec * 2, self.nums_ctp)
        self.beizer_transform = MLP(self.embed_dims, self.embed_dims//2, 2, 2)

        self.dilate_radius = dilate_radius

        self.weight_kp = weight_kp
        self.weight_adj = weight_adj

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers
        self.num_pred = num_pred
        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])

        if not self.as_two_stage:
            if self.bev_encoder_type == 'BEVFormerEncoder':
                self.bev_embedding = nn.Embedding(
                    self.bev_h * self.bev_w, self.embed_dims)
            else:
                self.bev_embedding = None
            if self.query_embed_type == 'all_pts':
                self.query_embedding = nn.Embedding(self.num_query,
                                                    self.embed_dims * 2)
            elif self.query_embed_type == 'instance_pts':
                self.query_embedding = None
                self.instance_embedding = nn.Embedding(self.num_vec, self.embed_dims * 2)
                self.pts_embedding = nn.Embedding(self.num_pts_per_vec, self.embed_dims * 2)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
    

    def get_inv_bernstein_basis(self, nums_pts, nums_ctp):
        
        def comb(n, k):
                return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))
        B = torch.zeros((nums_pts, nums_ctp))
        t = torch.arange(nums_pts)/(nums_pts-1)
        for i in range(nums_pts):
            for j in range(nums_ctp):
                B[i,j] = comb(nums_ctp-1, j)*torch.pow(t[i], j) *torch.pow(1-t[i], nums_ctp-1-j) 
        inv_B = torch.linalg.pinv(B)
        return inv_B

    @force_fp32(apply_to=('mlvl_feats', 'prev_bev'))
    def forward(self, mlvl_feats, lidar_feat, img_metas, prev_bev=None,  only_bev=False):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder. 
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """

        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        
        if self.query_embed_type == 'all_pts':
            object_query_embeds = self.query_embedding.weight.to(dtype)
        elif self.query_embed_type == 'instance_pts':
            pts_embeds = self.pts_embedding.weight.unsqueeze(0)
            instance_embeds = self.instance_embedding.weight.unsqueeze(1)
            object_query_embeds = (pts_embeds + instance_embeds).flatten(0, 1).to(dtype)
        if self.bev_embedding is not None:
            bev_queries = self.bev_embedding.weight.to(dtype)

            bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                                device=bev_queries.device).to(dtype)
            bev_pos = self.positional_encoding(bev_mask).to(dtype)
        else:
            bev_queries = None
            bev_mask = None
            bev_pos = None

        if only_bev:
            return self.transformer.get_bev_features(
                mlvl_feats,
                lidar_feat,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
        else:
            outputs = self.transformer(
                mlvl_feats,
                lidar_feat,
                bev_queries,
                object_query_embeds,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None,
                img_metas=img_metas,
                prev_bev=prev_bev
        )

        bev_embed, hs, init_reference, inter_references, kp_bev_preds = outputs
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        outputs_pts_coords = []
        outputs_sms = []
        out_sm = torch.zeros((bs, self.num_vec, self.num_vec),
                                  dtype=hs.dtype, device=hs.device)
        vertex_feat_out = None
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)

            outputs_class = self.cls_branches[lvl](hs[lvl]
                                            .view(bs,self.num_vec, self.num_pts_per_vec,-1)
                                            .mean(2))
            tmp = self.reg_branches[lvl](hs[lvl])
            

            vertex_feat = hs[lvl].view(bs,self.num_vec, self.num_pts_per_vec,-1).mean(2)
            vertex_feat = self.vertex_inteact(vertex_feat.permute(1, 0, 2), out_sm).permute(1, 0, 2)
            vertex_feat_out = self.GRU(vertex_feat, vertex_feat_out)
            out_sm = self.lclc_branch(vertex_feat_out)

            # TODO: check the shape of reference
            assert reference.shape[-1] == 2
            tmp[..., 0:2] += reference[..., 0:2]

            tmp = tmp.sigmoid()

            outputs_coord, outputs_pts_coord = self.transform_box(tmp)
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_pts_coords.append(outputs_pts_coord)
            outputs_sms.append(out_sm)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outputs_pts_coords = torch.stack(outputs_pts_coords)
        outputs_sms = torch.stack(outputs_sms)

        outs = {
            'bev_embed': bev_embed,
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'all_pts_preds': outputs_pts_coords,
            'all_sms_preds': outputs_sms,
            'all_hs': hs,
            'kp_bev_preds': kp_bev_preds,
        }

        return outs

    def transform_box(self, pts, y_first=False):
        """
        Converting the points set into bounding box.

        Args:
            pts: the input points sets (fields), each points
                set (fields) is represented as 2n scalar.
            y_first: if y_fisrt=True, the point set is represented as
                [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
                represented as [x1, y1, x2, y2 ... xn, yn].
        Returns:
            The bbox [cx, cy, w, h] transformed from points.
        """
        pts_reshape = pts.view(pts.shape[0], self.num_vec,
                                self.num_pts_per_vec,2)
        pts_y = pts_reshape[:, :, :, 0] if y_first else pts_reshape[:, :, :, 1]
        pts_x = pts_reshape[:, :, :, 1] if y_first else pts_reshape[:, :, :, 0]
        if self.transform_method == 'minmax':
            # import pdb;pdb.set_trace()

            xmin = pts_x.min(dim=2, keepdim=True)[0]
            xmax = pts_x.max(dim=2, keepdim=True)[0]
            ymin = pts_y.min(dim=2, keepdim=True)[0]
            ymax = pts_y.max(dim=2, keepdim=True)[0]
            bbox = torch.cat([xmin, ymin, xmax, ymax], dim=2)
            bbox = bbox_xyxy_to_cxcywh(bbox)
        else:
            raise NotImplementedError
        return bbox, pts_reshape

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_bboxes_ignore=None,
             img_metas=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'
        gt_vecs_list = copy.deepcopy(gt_bboxes_list)
        # import pdb;pdb.set_trace()
        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        all_pts_preds  = preds_dicts['all_pts_preds']
        all_sms_preds  = preds_dicts['all_sms_preds']
        all_hs_preds = preds_dicts['all_hs']
        kp_bev_preds = preds_dicts['kp_bev_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device


        gt_bboxes_list = [
            gt_bboxes.bbox.to(device) for gt_bboxes in gt_vecs_list]

        gt_shifts_pts_list = [gt_bboxes.fixed_num_sampled_points_ambiguity.to(device) for gt_bboxes in gt_vecs_list]

        gt_sms_list = [torch.from_numpy(gt_bboxes.adj_matrix).to(device) for gt_bboxes in gt_vecs_list]
        gt_control_pts_list = [gt_bboxes.get_beizer_control_pts.to(device) for gt_bboxes in gt_vecs_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_shifts_pts_list = [gt_shifts_pts_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        all_gt_sms_list = [gt_sms_list for _ in range(num_dec_layers)]
        all_gt_control_pts_list = [gt_control_pts_list for _ in range(num_dec_layers)]

        # import pdb;pdb.set_trace()
        losses_cls, losses_bbox, losses_iou, losses_pts, losses_dir, losses_adj, losses_beizer = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,all_pts_preds,all_sms_preds,
            all_hs_preds, all_gt_bboxes_list, all_gt_labels_list,all_gt_shifts_pts_list,
            all_gt_sms_list, all_gt_control_pts_list, all_gt_bboxes_ignore_list)

        loss_dict = dict()

        gt_bev_kp_list = [self.get_bev_keypoint(gt_bboxes).to(device) for gt_bboxes in gt_vecs_list]
        gt_bev_kp = torch.stack(gt_bev_kp_list)
        loss_dict['loss_kp'] = _neg_loss(kp_bev_preds, gt_bev_kp, weights=2) * self.weight_kp

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_pts'] = losses_pts[-1]
        loss_dict['loss_dir'] = losses_dir[-1]
        loss_dict['loss_adj'] = losses_adj[-1]
        loss_dict['loss_beizer'] = losses_beizer[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_pts_i, loss_dir_i, loss_adj_i, loss_beizer_i in zip(
                                            losses_cls[:-1],
                                           losses_pts[:-1],
                                           losses_dir[:-1],
                                           losses_adj[:-1],
                                           losses_beizer[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_pts'] = loss_pts_i
            loss_dict[f'd{num_dec_layer}.loss_dir'] = loss_dir_i
            loss_dict[f'd{num_dec_layer}.loss_adj'] = loss_adj_i
            loss_dict[f'd{num_dec_layer}.loss_beizer'] = loss_beizer_i
            num_dec_layer += 1
        return loss_dict, None, None

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    pts_preds,
                    sms_preds,
                    hs_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_shifts_pts_list,
                    gt_sms_list,
                    gt_control_pts_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_pts_list (list[Tensor]): Ground truth pts for each image
                with shape (num_gts, fixed_num, 2) in [x,y] format.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        pts_preds_list = [pts_preds[i] for i in range(num_imgs)]
        sms_preds_list = [sms_preds[i] for i in range(num_imgs)]
        hs_preds_list = [hs_preds[i] for i in range(num_imgs)]
        # import pdb;pdb.set_trace()
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,pts_preds_list,sms_preds_list,
                                           hs_preds_list, gt_bboxes_list, gt_labels_list,gt_shifts_pts_list,
                                           gt_sms_list, gt_control_pts_list, gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pts_targets_list, pts_weights_list, loss_adj_list, loss_beizer_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        # import pdb;pdb.set_trace()
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        pts_targets = torch.cat(pts_targets_list, 0)
        pts_weights = torch.cat(pts_weights_list, 0)

        # adj loss
        loss_adj = sum(loss_adj_list) / len(loss_adj_list)

        #beizer loss
        loss_beizer = sum(loss_beizer_list) / len(loss_beizer_list)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # import pdb;pdb.set_trace()
        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_2d_bbox(bbox_targets, self.pc_range)
        # normalized_bbox_targets = bbox_targets
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :4], normalized_bbox_targets[isnotnan,
                                                               :4], bbox_weights[isnotnan, :4],
            avg_factor=num_total_pos)

        # regression pts CD loss
        # pts_preds = pts_preds
        # import pdb;pdb.set_trace()
        
        # num_samples, num_order, num_pts, num_coords
        normalized_pts_targets = normalize_2d_pts(pts_targets, self.pc_range)

        # num_samples, num_pts, num_coords
        pts_preds = pts_preds.reshape(-1, pts_preds.size(-2),pts_preds.size(-1))
        if self.num_pts_per_vec != self.num_pts_per_gt_vec:
            pts_preds = pts_preds.permute(0,2,1)
            pts_preds = F.interpolate(pts_preds, size=(self.num_pts_per_gt_vec), mode='linear',
                                    align_corners=True)
            pts_preds = pts_preds.permute(0,2,1).contiguous()

        # import pdb;pdb.set_trace()
        loss_pts = self.loss_pts(
            pts_preds[isnotnan,:,:], normalized_pts_targets[isnotnan,
                                                            :,:], 
            pts_weights[isnotnan,:,:],
            avg_factor=num_total_pos)
        dir_weights = pts_weights[:, :-self.dir_interval,0]
        denormed_pts_preds = denormalize_2d_pts(pts_preds, self.pc_range)
        denormed_pts_preds_dir = denormed_pts_preds[:,self.dir_interval:,:] - denormed_pts_preds[:,:-self.dir_interval,:]
        pts_targets_dir = pts_targets[:, self.dir_interval:,:] - pts_targets[:,:-self.dir_interval,:]
        # dir_weights = pts_weights[:, indice,:-1,0]
        # import pdb;pdb.set_trace()
        loss_dir = self.loss_dir(
            denormed_pts_preds_dir[isnotnan,:,:], pts_targets_dir[isnotnan,
                                                                          :,:],
            dir_weights[isnotnan,:],
            avg_factor=num_total_pos)

        bboxes = denormalize_2d_bbox(bbox_preds, self.pc_range)
        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes[isnotnan, :4], bbox_targets[isnotnan, :4], bbox_weights[isnotnan, :4], 
            avg_factor=num_total_pos)

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_iou = torch.nan_to_num(loss_iou)
            loss_pts = torch.nan_to_num(loss_pts)
            loss_dir = torch.nan_to_num(loss_dir)
            loss_adj = torch.nan_to_num(loss_adj)
        return loss_cls, loss_bbox, loss_iou, loss_pts, loss_dir, loss_adj, loss_beizer

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    pts_preds_list,
                    sms_preds_list,
                    hs_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_shifts_pts_list,
                    gt_sms_list,
                    gt_control_pts_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pts_targets_list, pts_weights_list,
         loss_adj_list, loss_beizer_list,
         pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,pts_preds_list,sms_preds_list,hs_preds_list,
            gt_labels_list, gt_bboxes_list, gt_shifts_pts_list,gt_sms_list, gt_control_pts_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, pts_targets_list, pts_weights_list,
                loss_adj_list, loss_beizer_list, num_total_pos, num_total_neg)

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           pts_pred,
                           sm_pred,
                           hs_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_shifts_pts,
                           gt_sms,
                           gt_control_pts,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
        # import pdb;pdb.set_trace()
        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_c = gt_bboxes.shape[-1]
        # import pdb;pdb.set_trace()
        assign_result, order_index = self.assigner.assign(bbox_pred, cls_score, pts_pred,
                                             gt_bboxes, gt_labels, gt_shifts_pts,
                                             gt_bboxes_ignore)

        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        # pts_sampling_result = self.sampler.sample(assign_result, pts_pred,
        #                                       gt_pts)

        
        # import pdb;pdb.set_trace()
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        pos_gt_inds = sampling_result.pos_assigned_gt_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # adj matrix targets
        sm_pred = sm_pred[pos_inds][:, pos_inds]
        sm_target = gt_sms[pos_gt_inds][:, pos_gt_inds].to(torch.float32)

        # connectivity loss
        loss_adj = self.loss_adj(sm_pred, sm_target) * self.weight_adj

        # beizer loss
        pred_control_pts = torch.zeros((len(pos_inds), len(pos_inds), self.nums_ctp, 2)).to(pts_pred.device)
        gt_control_pts = gt_control_pts[pos_gt_inds][:, pos_gt_inds]
        hs_pred_permute = hs_pred.view(self.num_vec, self.num_pts_per_vec,-1)[pos_inds]
        for i in range(len(sm_target)):
            connection = torch.where(sm_target[i]==1)[0]
            for c in connection:
                new_line_embed = torch.cat((hs_pred_permute[i], hs_pred_permute[c]))
                beizer_space_embed = torch.matmul(self.inv_B.to(pts_pred.device), new_line_embed)
                control_pts = self.beizer_transform(beizer_space_embed)
                pred_control_pts[i, c] = denormalize_2d_pts(torch.sigmoid(control_pts), self.pc_range)
        loss_beizer = self.loss_ctp(pred_control_pts, 
                                    gt_control_pts,
                                    avg_factor=len(pos_inds))

        # pts targets
        # import pdb;pdb.set_trace()
        # pts_targets = torch.zeros_like(pts_pred)
        # num_query, num_order, num_points, num_coords
        if order_index is None:
            # import pdb;pdb.set_trace()
            assigned_shift = gt_labels[sampling_result.pos_assigned_gt_inds]
        else:
            assigned_shift = order_index[sampling_result.pos_inds, sampling_result.pos_assigned_gt_inds]
        pts_targets = pts_pred.new_zeros((pts_pred.size(0),
                        pts_pred.size(1), pts_pred.size(2)))
        pts_weights = torch.zeros_like(pts_targets)
        pts_weights[pos_inds] = 1.0

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        pts_targets[pos_inds] = gt_shifts_pts[sampling_result.pos_assigned_gt_inds,assigned_shift,:,:]
        return (labels, label_weights, bbox_targets, bbox_weights,
                pts_targets, pts_weights, loss_adj, loss_beizer,
                pos_inds, neg_inds)
    
    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts):

        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)
        ret_list = []

        for i in range(num_samples):
            preds = preds_dicts[i]
            line_scores = preds['line_scores']
            line_labels = preds['line_labels']
            line = preds['line']
            pts_scores = preds['pts_scores']
            pts_labels = preds['pts_labels']
            pts = preds['pts']
            graphs = preds['graph']
            adj_matrix = preds['adj_matrix']
            ret_list.append([line_scores, line_labels, line, 
                            pts_scores, pts_labels, pts, graphs, adj_matrix])

        return ret_list

    def get_bev_keypoint(self, lines):
        bev_keypoint_map = np.zeros((self.bev_h, self.bev_w))
        keypoints = lines.key_points
        for key in keypoints:
            key = np.array(key)[::-1]
            key[0] = (key[0] - self.pc_range[1])/(self.pc_range[4] - self.pc_range[1])* self.bev_h
            key[1] = (key[1] - self.pc_range[0])/(self.pc_range[3] - self.pc_range[0])* self.bev_w
            key[0] = np.clip(key[0], 0, self.bev_h-1)
            key[1] = np.clip(key[1], 0, self.bev_w-1)
            bev_keypoint_map[int(key[0])][int(key[1])] = 1

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.dilate_radius, self.dilate_radius))
        bev_keypoint_map = cv2.dilate(bev_keypoint_map, kernel)

        return torch.tensor(bev_keypoint_map)


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

def _neg_loss(pred, gt, weights=None):
    """Modified focal loss.

    Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    # prevent lidar overflow ! carefully use ! 
    eps = 1e-6
    pos_loss = torch.log(pred + eps) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred + eps) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    # pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if weights is not None:
        pos_loss = (pos_loss * weights).sum()
    else:
        pos_loss = pos_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


# Topology head
class TopologyHead(nn.Module):
    def __init__(self,
                 in_channels,
                 activate=True):
        super().__init__()

        self.association_embed_maker = MLP(in_channels, in_channels//2, in_channels//4, 2)
        self.association_classifier = MLP(in_channels//2, in_channels//4, 1, 2)
        self.activate = activate

    def forward(self, vertex):
        vertex_embed = torch.sigmoid(self.association_embed_maker(vertex))
        vertex_features1 = torch.unsqueeze(vertex_embed,dim=2).repeat(1, 1, vertex_embed.size(1),1)
        vertex_features2 = torch.unsqueeze(vertex_embed,dim=1).repeat(1, vertex_embed.size(1),1,1)
        vertex_features = torch.cat([vertex_features1, vertex_features2],dim=-1)
        similarity = torch.squeeze(self.association_classifier(vertex_features),dim=-1)
        if self.activate:
            similarity = torch.sigmoid(similarity)

        return similarity

class GNN(BaseModule):
    def __init__(self,
                 embed_dims=256,
                 feedforward_channels=512,
                 num_fcs=2,
                 ffn_drop=0.1,
                 init_cfg=None,
                 edge_weight=0.5,
                 **kwargs):
        super(GNN, self).__init__(init_cfg)
        assert num_fcs >= 2, 'num_fcs should be no less ' \
            f'than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.activate = nn.ReLU(inplace=True)

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                Sequential(
                    Linear(in_channels, feedforward_channels), self.activate,
                    nn.Dropout(ffn_drop)))
            in_channels = feedforward_channels
        layers.append(
            Sequential(
                Linear(feedforward_channels, embed_dims), self.activate,
                nn.Dropout(ffn_drop)))
        self.layers = Sequential(*layers)
        self.edge_weight = edge_weight

        self.lclc_gnn_layer = LclcGNNLayer(embed_dims, embed_dims, edge_weight=edge_weight)

        self.downsample = nn.Linear(embed_dims, embed_dims)

        self.gnn_dropout1 = nn.Dropout(ffn_drop)
        self.gnn_dropout2 = nn.Dropout(ffn_drop)

    def forward(self, lc_query, lclc_adj):

        out = self.layers(lc_query)
        out = out.permute(1, 0, 2)

        out = self.lclc_gnn_layer(out, lclc_adj)

        out = self.activate(out)
        out = self.gnn_dropout1(out)
        out = self.downsample(out)
        out = self.gnn_dropout2(out)

        out = out.permute(1, 0, 2)

        return lc_query + out

class LclcGNNLayer(nn.Module):

    def __init__(self, in_features, out_features, edge_weight=0.5):
        super(LclcGNNLayer, self).__init__()
        self.edge_weight = edge_weight

        if self.edge_weight != 0:
            self.weight_forward = torch.Tensor(in_features, out_features)
            self.weight_forward = nn.Parameter(nn.init.xavier_uniform_(self.weight_forward))
            self.weight_backward = torch.Tensor(in_features, out_features)
            self.weight_backward = nn.Parameter(nn.init.xavier_uniform_(self.weight_backward))

        self.weight = torch.Tensor(in_features, out_features)
        self.weight = nn.Parameter(nn.init.xavier_uniform_(self.weight))
        self.edge_weight = edge_weight

    def forward(self, input, adj):

        support_loop = torch.matmul(input, self.weight)
        output = support_loop

        if self.edge_weight != 0:
            support_forward = torch.matmul(input, self.weight_forward)
            output_forward = torch.matmul(adj, support_forward)
            output += self.edge_weight * output_forward

            support_backward = torch.matmul(input, self.weight_backward)
            output_backward = torch.matmul(adj.permute(0, 2, 1), support_backward)
            output += self.edge_weight * output_backward

        return output

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)

        self.reset_parameters()


    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, hx):

        if hx is None:
            hx = Variable(input.new_zeros(input.size(0), input.size(1), self.hidden_size))

        x_t = self.x2h(input)
        h_t = self.h2h(hx)

        x_reset, x_upd, x_new = x_t.chunk(3, 2)
        h_reset, h_upd, h_new = h_t.chunk(3, 2)

        reset_gate = torch.sigmoid(x_reset + h_reset)
        update_gate = torch.sigmoid(x_upd + h_upd)
        new_gate = torch.tanh(x_new + (reset_gate * h_new))

        hy = update_gate * hx + (1 - update_gate) * new_gate

        return hy
