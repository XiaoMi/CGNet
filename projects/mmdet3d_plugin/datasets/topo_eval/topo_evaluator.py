import numpy as np
import networkx as nx
from .geotopo import Evaluator as GeoTopoEvaluator
from .topo_metrics import calc_iou, calc_apls, calc_sda
from .topo_metrics import nx_to_geo_topo_format
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d
import random
import warnings
from shapely.geometry import LineString
import uuid

random.seed(0)

class GraphEvaluator():

    """
    Evaluator for graphs
    """

    def __init__(self, radius=8, interp_dist=2, prop_dist=400, area_size=[30, 60], lane_width=1, gmode='direct'):
        self.radius = radius
        self.interp_dist = interp_dist
        self.prop_dist = prop_dist
        self.area_size = area_size
        self.lane_width = lane_width
        self.gmode = gmode

    def evaluate_graph(self, graph_gt: nx.DiGraph, graph_pred: nx.DiGraph, verbose=False):

        graph_gt = assign_edge_lengths(graph_gt)
        graph_pred = assign_edge_lengths(graph_pred)

        iou = calc_iou(graph_gt, graph_pred, area_size=self.area_size, lane_width=self.lane_width)
        
        sda = calc_sda(graph_gt, graph_pred)

        graph_gt_for_apls = prepare_graph_apls(graph_gt)
        graph_pred_for_apls = prepare_graph_apls(graph_pred)

        # Try to calculate APLS metric
        try:
            apls = calc_apls(graph_gt_for_apls, graph_pred_for_apls)
        except Exception as e:
            apls = 0
            warnings.warn("Error calculating APLS metric: {}.".format(e))
        
        if verbose:
            print("iou : ", iou)
            print("sda : ", sda)
            print("apls : ", apls)

        # Try to calculate GEO and TOPO metrics
        graph_gt_ = nx_to_geo_topo_format(graph_gt)
        graph_pred_ = nx_to_geo_topo_format(graph_pred)

        evaluator = GeoTopoEvaluator(graph_gt_, graph_pred_, self.interp_dist, self.prop_dist, self.gmode)
        
        (geo_precision, 
        geo_recall, 
        topo_precision, 
        topo_recall,
        geo_f1,
        topo_f1,
        jtopo_f1) = \
        evaluator.topoMetric(thr=self.radius, verbose=verbose)


        metrics_dict = {
            'Graph IoU': iou,
            'APLS': apls,
            'GEO Precision': geo_precision,
            'GEO Recall': geo_recall,
            'GEO F1' : geo_f1,
            'TOPO Precision': topo_precision,
            'TOPO Recall': topo_recall,
            'TOPO F1' : topo_f1,
            'JTOPO F1' : jtopo_f1,
            'SDA': sda
        }

        return metrics_dict


def truncated_uuid4():
    return str(int(uuid.uuid4()))[0:6]

def prepare_graph_apls(g):

    # relabel nodes with their node position and a random string to avoid name collisions between graphs
    g = nx.relabel_nodes(g, {n: str(g.nodes[n]['pos']) + truncated_uuid4() for n in g.nodes()})

    g = nx.to_undirected(g)

    # add x,y coordinates to graph properties
    for n, d in g.nodes(data=True):
        d['x'] = d['pos'][0]
        d['y'] = d['pos'][1]

    # add length to graph properties
    for u, v, d in g.edges(data=True):
        d['geometry'] = LineString([(g.nodes[u]['x'], g.nodes[u]['y']),
                                    (g.nodes[v]['x'], g.nodes[v]['y'])])
        d['length'] = d['geometry'].length

    return g

def assign_edge_lengths(G):
    for u, v, d in G.edges(data=True):
        d['length'] = np.linalg.norm(np.array(G.nodes[u]['pos']) - np.array(G.nodes[v]['pos']))
    return G