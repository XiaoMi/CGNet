import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
import warnings
from scipy.spatial.distance import cdist
from .apls import execute_apls
from geotopo import f1_score

def filter_graph(target, source, threshold=100):

    if len(source.nodes()) == 0:
        return source

    pos_target = np.array([list(target.nodes[n]['pos']) for n in target.nodes()])
    pos_source = np.array([list(source.nodes[n]['pos']) for n in source.nodes()])

    distance_matrix = cdist(pos_target, pos_source)


    source_ = source.copy(as_view=False)
    is_close_to_target = np.min(distance_matrix, axis=0) < threshold
    for i, n in enumerate(list(source_.nodes())):
        if not is_close_to_target[i]:
            if n in source.nodes():
                source.remove_node(n)

    return source


def calc_sda(graph_gt, graph_pred, threshold=1):
    """
    Calculates the split detection accuracy (SDA) metric for a pair of graphs and a given threshold.
    """

    split_point_positions_gt = []
    split_point_positions_pred = []

    for n in graph_gt.nodes():
        if graph_gt.out_degree(n) >= 2 or graph_gt.in_degree(n) >= 2:
            split_point_positions_gt.append(graph_gt.nodes[n]['pos'])

    for n in graph_pred.nodes():
        if graph_pred.out_degree(n) >= 2 or graph_pred.in_degree(n) >= 2: 
            split_point_positions_pred.append(graph_pred.nodes[n]['pos'])

    if len(split_point_positions_gt) == 0:
        # print("     No lane splits in ground truth graph")
        # warnings.warn("No lane splits in ground truth graph")
        return np.nan

    if len(split_point_positions_pred) == 0:
        # print("     No lane splits in predicted graph")
        # warnings.warn("No lane splits in predicted graph")
        return 0.0

    # build up cost matrix between gt and pred split points
    split_point_positions_gt = np.array(split_point_positions_gt)
    split_point_positions_pred = np.array(split_point_positions_pred)

    cost_matrix = np.linalg.norm(split_point_positions_gt[:, None, :] - split_point_positions_pred[None, :, :], axis=-1)

    # find the minimum cost for each gt split point
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    tp = np.sum(cost_matrix[row_ind, col_ind] < threshold)
    fp = len(split_point_positions_pred) - tp
    fn = len(split_point_positions_gt) - tp

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    F1 = f1_score(precision, recall)

    return F1


def calc_apls(g_gt, g_pred):

    g_gt = [g_gt]
    g_pred = [g_pred]

    apls_dict = execute_apls(g_gt, g_pred, verbose=False)

    return apls_dict['APLS']


def render_graph(graph, imsize=[256, 256], width=10):
    """
    Render a graph as an image.
    Args:
        graph: networkx graph
        imsize: image size
        width: line width of edges
    Returns: rendered graph
    """

    im = np.zeros(imsize).astype(np.uint8)

    for e in graph.edges():
        start = graph.nodes[e[0]]['pos']
        end = graph.nodes[e[1]]['pos']
        x1 = int(start[0])
        y1 = int(start[1])
        x2 = int(end[0])
        y2 = int(end[1])
        cv2.line(im, (x1, y1), (x2, y2), 255, width)
    return im


def calc_iou(graph_gt, graph_pred, area_size=[256, 256], lane_width=10):
    """
    Calculate IoU of two graphs.
    :param graph_gt: ground truth graph
    :param graph_pred: predicted graph
    :return: IoU
    """

    render_gt = render_graph(graph_gt, imsize=area_size, width=lane_width)
    render_pred = render_graph(graph_pred, imsize=area_size, width=lane_width)

    # import matplotlib.pyplot as plt
    # fig, axarr = plt.subplots(1, 2, figsize=(10, 10), sharex=True, sharey=True)
    # axarr[0].imshow(render_gt)
    # axarr[1].imshow(render_pred)
    # plt.show()

    # Calculate IoU
    intersection = np.logical_and(render_gt, render_pred)
    union = np.logical_or(render_gt, render_pred)
    iou = np.sum(intersection) / (1e-8 + np.sum(union))

    return iou


def nx_to_geo_topo_format(nx_graph):
    """
    Convert a networkx graph to the format used for calculating the GEO and TOPO metrics.
    """

    neighbors = {}

    for e in nx_graph.edges():
        x1 = nx_graph.nodes[e[0]]['pos'][0]
        y1 = nx_graph.nodes[e[0]]['pos'][1]
        x2 = nx_graph.nodes[e[1]]['pos'][0]
        y2 = nx_graph.nodes[e[1]]['pos'][1]

        k1 = (int(x1*10), int(y1*10))
        k2 = (int(x2*10), int(y2*10))

        if k1 not in neighbors:
            neighbors[k1] = []

        if k2 not in neighbors[k1]:
            neighbors[k1].append(k2)

    return neighbors
