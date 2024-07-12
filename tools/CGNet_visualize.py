# Copyright (C) 2024 Xiaomi Corporation.

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, 
# software distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and limitations under the License.

import argparse
import mmcv
import os
import torch
import warnings

import sys
dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(dir_path)

from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet3d.utils import collect_env, get_root_logger
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.datasets import replace_ImageToTensor
import os.path as osp
import numpy as np
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description='vis map gt and pred')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--samples', default=2000, help='samples to visualize')
    parser.add_argument(
        '--show-dir', help='directory where visualizations will be saved')
    parser.add_argument('--save-video', action='store_true', help='generate video')
    parser.add_argument(
        '--gt-format',
        type=str,
        nargs='+',
        default=['se_points',],
        help='vis format, default should be "points",'
        'support ["se_pts","bbox","fixed_num_pts","polyline_pts"]')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    if args.show_dir is None:
        args.show_dir = osp.join('./work_dirs', 
                                osp.splitext(osp.basename(args.config))[0],
                                'vis_pred')
    # create vis_label dir
    mmcv.mkdir_or_exist(osp.abspath(args.show_dir))
    cfg.dump(osp.join(args.show_dir, osp.basename(args.config)))
    logger = get_root_logger()
    logger.info(f'DONE create vis_pred dir: {args.show_dir}')


    dataset = build_dataset(cfg.data.test)
    dataset.is_vis_on_test = True #TODO, this is a hack
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        # workers_per_gpu=cfg.data.workers_per_gpu,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    logger.info('Done build test data set')

    # build the model and load checkpoint
    # import pdb;pdb.set_trace()
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    model.pts_bbox_head.bbox_coder.vis_mode = True
    model.pts_bbox_head.bbox_coder.score_threshold = 0.4
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    logger.info('loading check point')
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE
    logger.info('DONE load check point')
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    logger.info('BEGIN vis test dataset samples gt label & pred')


    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    out_dir_list = []
    # import pdb;pdb.set_trace()
    for i, data in enumerate(data_loader):
        if ~(data['gt_labels_3d'].data[0][0] != -1).any():
            # import pdb;pdb.set_trace()
            logger.error(f'\n empty gt for index {i}, continue')
            prog_bar.update()  
            continue
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        
        # import pdb;pdb.set_trace()
        img_metas = data['img_metas'][0].data[0]
        token = img_metas[0]['scene_token']
        gt_bboxes_3d = data['gt_bboxes_3d'].data[0]
        gt_labels_3d = data['gt_labels_3d'].data[0]

        gt_lines_fixed_num_pts = (gt_bboxes_3d[0].fixed_num_sampled_points)
        gt_graph = gt_bboxes_3d[0].get_graph_gt_vis
        img = img_metas[0]['ori_image']

        result_dic = result[0]['pts_bbox']
        line = np.array([l.cpu().numpy() for l in result_dic['line']])
        line_scores = result_dic['line_scores'].cpu().numpy()
        line_labels = result_dic['line_labels'].cpu().numpy()

        pts = result_dic['pts'].cpu().numpy()
        pts_scores = result_dic['pts_scores'].cpu().numpy()
        pts_labels = result_dic['pts_labels'].cpu().numpy()
        graph = result_dic['graph']
        
        if gt_labels_3d[0].shape[0] != gt_lines_fixed_num_pts.shape[0]:
            gt_labels_3d = np.zeros(gt_lines_fixed_num_pts.shape[0])
        else:
            gt_labels_3d = gt_labels_3d[0].cpu().numpy()

        gts = [{'pts':gt_lines_fixed_num_pts[i,...].cpu().numpy(), 'type':gt_labels_3d[i]} for i in range(gt_lines_fixed_num_pts.shape[0])]
        line_preds = [{'line':line[i,...], 'type':line_labels[i],'confidence_level':line_scores[i]} for i in range(line.shape[0])]
        pts_preds = [{'pts':pts[i,...], 'type':pts_labels[i],'confidence_level':pts_scores[i]} for i in range(pts.shape[0])]
        out_dir = os.path.join(args.show_dir, token)
        if out_dir not in out_dir_list:
            out_dir_list.append(out_dir)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        out_path = os.path.join(out_dir, "{}.jpg".format(i))
        render(img, out_path, gts, graph, gt_graph, )

        prog_bar.update()

    # # create video for scenes
    if args.save_video: 
        logger.info('\n Creat Video')
        for path in out_dir_list:
            creat_video(path)

    logger.info('\n DONE vis test dataset samples gt label & pred')


def render(imgs, out_file, gt, graph, gt_graph, height=900):
        

        COLOR = ((0, 0, 0), (116, 92, 75), (83, 97, 96), (255, 0, 0), (0, 0, 255))

        scale = height // 30
        map_img_p = np.ones((height*2, height, 3), dtype=np.uint8) * 255
        map_img_gt = np.ones((height*2, height, 3), dtype=np.uint8) * 255

        if gt is not None:
            for e in gt_graph.edges():
                x1 = gt_graph.nodes[e[0]]['pos'][0] *scale
                y1 = gt_graph.nodes[e[0]]['pos'][1] *scale
                x2 = gt_graph.nodes[e[1]]['pos'][0] *scale
                y2 = gt_graph.nodes[e[1]]['pos'][1] *scale
                x1 += height//2
                y1 = height - y1
                x2 += height//2
                y2 = height - y2
                distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                cv2.circle(map_img_gt, (np.int32(x1), np.int32(y1)), 8, COLOR[-1], -1)
                cv2.circle(map_img_gt, (np.int32(x2), np.int32(y2)), 8, COLOR[-1], -1)
                cv2.arrowedLine(map_img_gt, 
                        (np.int32(x1), np.int32(y1)), 
                        (np.int32(x2), np.int32(y2)), 
                        color = COLOR[0], 
                        thickness=3,
                        tipLength=float(20/(distance+1)))
                
        for e in graph.edges():
            x1 = graph.nodes[e[0]]['pos'][0] *scale
            y1 = graph.nodes[e[0]]['pos'][1] *scale
            x2 = graph.nodes[e[1]]['pos'][0] *scale
            y2 = graph.nodes[e[1]]['pos'][1] *scale
            x1 += height//2
            y1 = height - y1
            x2 += height//2
            y2 = height - y2
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            cv2.circle(map_img_p, (np.int32(x1), np.int32(y1)), 8, COLOR[-1], -1)
            cv2.circle(map_img_p, (np.int32(x2), np.int32(y2)), 8, COLOR[-1], -1)
            cv2.arrowedLine(map_img_p, 
                    (np.int32(x1), np.int32(y1)), 
                    (np.int32(x2), np.int32(y2)), 
                    color = COLOR[3], 
                    thickness=3,
                    tipLength=float(20/(distance+1)))


        f, fr, fl, b, bl, br = imgs
        canvas = cv2.vconcat([cv2.hconcat([fl, f, fr]), cv2.hconcat([bl, b, br])])
        canvas = cv2.hconcat([canvas, map_img_gt.astype(np.float32), map_img_p.astype(np.float32), ])
        cv2.imwrite(out_file, canvas)

def creat_video(folder_path):
    video_name = folder_path + '/output.mp4'

    file_names = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    file_names.sort(key=lambda x:int(x.split('.')[0]))
    
    img = cv2.imread(os.path.join(folder_path, file_names[0]))
    height, width, channels = img.shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_name, fourcc, 5, (width, height))

    for file_name in file_names:
        img = cv2.imread(os.path.join(folder_path, file_name))
        video_writer.write(img)

    video_writer.release()


if __name__ == '__main__':
    main()
