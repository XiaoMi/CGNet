import sys
import json
import pickle
import numpy as np
import rtree
import cv2
import scipy.ndimage


class Evaluator():
    def __init__(self, gt_graph, pred_graph, interp_dist=2, prop_dist=400, gmode='direct'):
        self.gt_graph = gt_graph
        self.prop_graph = pred_graph
        self.interp_dist  = interp_dist
        self.prop_dist = prop_dist
        self.gmode = gmode

    def interpolateGraph(self, graph):
        newgraph = {}
        exist = set()

        for nid, nei in graph.items():
            for nn in nei:
                if (nid, nn) in exist or (nn, nid) in exist:
                    continue
                x1, y1 = nid
                x2, y2 = nn

                exist.add((nid, nn))

                L = int(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) / self.interp_dist) + 1
                L = max(2, L)
                last_node = (x1, y1)
                for i in range(1, L):
                    a = 1.0 - float(i) / (L - 1)
                    x = x1 * a + x2 * (1 - a)
                    y = y1 * a + y2 * (1 - a)

                    nk1 = last_node
                    nk2 = (x, y)

                    if x < -150 or x >= 150 or y < -300 or y >= 300:
                        last_node = (x, y)
                        continue

                    if last_node[0] < -150 or last_node[0] >= 150 or last_node[1] < -300 or last_node[1] >= 300:
                        last_node = (x, y)
                        continue

                    if nk1 not in newgraph:
                        newgraph[nk1] = [nk2]
                    elif nk2 not in newgraph[nk1]:
                        newgraph[nk1].append(nk2)

                    if self.gmode =='direct':
                        if nk2 not in newgraph:
                            newgraph[nk2] = []
                    else:
                        if nk2 not in newgraph:
                            newgraph[nk2] = [nk1]
                        elif nk1 not in newgraph[nk2]:
                            newgraph[nk2].append(nk1)

                    last_node = (x, y)

        return newgraph

    def propagateByDistance(self, graph, nid, steps=400):
        visited = set()
        queue = [(nid, 0)]

        def distance(p1, p2):
            a = (p1[0] - p2[0]) ** 2
            b = (p1[1] - p2[1]) ** 2
            return np.sqrt(a + b)

        while len(queue) > 0:
            cur_nid, depth = queue.pop()
            visited.add(cur_nid)

            if depth >= steps:
                continue

            # print(cur_nid, len(graph[cur_nid]))

            for nei in graph[cur_nid]:
                if nei in visited:
                    continue

                queue.append((nei, depth + distance(cur_nid, nei)))

        return list(visited)

    def match(self, nodes1, nodes2, thr=8):
        idx = rtree.index.Index()
        for i in range(len(nodes1)):
            x, y = nodes1[i]
            idx.insert(i, (x - 1, y - 1, x + 1, y + 1))

        pairs = []

        m = thr

        for i in range(len(nodes2)):
            x, y = nodes2[i]

            candidates = list(idx.intersection((x - m, y - m, x + m, y + m)))
            for n in candidates:
                x2, y2 = nodes1[n]

                r = (x2 - x) ** 2 + (y2 - y) ** 2
                if r < thr * thr:
                    pairs.append((i, n, r))

        pairs = sorted(pairs, key=lambda x: x[2])

        matched = 0
        prop_set = set()
        gt_set = set()

        for pair in pairs:
            n1, n2, _ = pair
            if n1 in prop_set or n2 in gt_set:
                continue

            prop_set.add(n1)
            gt_set.add(n2)
            matched += 1

        precision = float(matched) / len(nodes1)
        recall = float(matched) / len(nodes2)

        return precision, recall

    def topoMetric(self, thr=8, mask=None, verbose=False):

        prop_graph = self.interpolateGraph(self.prop_graph)
        gt_graph = self.interpolateGraph(self.gt_graph)

        prop_nodes = prop_graph.keys()
        gt_nodes = gt_graph.keys()

        prop_nodes = list(prop_nodes)
        gt_nodes = list(gt_nodes)

        if len(prop_nodes)==0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        if mask is not None:
            mask = scipy.ndimage.imread(mask)
            if len(np.shape(mask)) == 3:
                mask = mask[:, :, 0]

        if verbose:
            print("nums of nodes after interpolate:" , len(prop_nodes), len(gt_nodes))

        idx = rtree.index.Index()

        for i in range(len(gt_nodes)):
            x, y = list(gt_nodes)[i]
            if mask is not None and mask[int(x), int(y)] < 127:
                continue
            idx.insert(i, (x - 1, y - 1, x + 1, y + 1))
        pairs = []

        m = thr

        for i in range(len(prop_nodes)):
            x, y = list(prop_nodes)[i]

            if mask is not None and mask[int(x), int(y)] < 127:
                continue

            candidates = list(idx.intersection((x - m, y - m, x + m, y + m)))
            for n in candidates:
                x2, y2 = list(gt_nodes)[n]

                r = (x2 - x) ** 2 + (y2 - y) ** 2
                if r < thr * thr:
                    pairs.append((i, n, r))

        pairs = sorted(pairs, key=lambda x: x[2])

        matched = 0
        prop_set = set()
        gt_set = set()
        ps, rs = [], []
        jps, jrs = [], []
        for pair in pairs:
            n1, n2, _ = pair
            if n1 in prop_set or n2 in gt_set:
                continue

            prop_set.add(n1)
            gt_set.add(n2)

            # compute precision and recall

            if len(gt_graph[gt_nodes[n2]])>1:
                nodes1 = self.propagateByDistance(prop_graph, prop_nodes[n1], self.prop_dist)
                nodes2 = self.propagateByDistance(gt_graph, gt_nodes[n2], self.prop_dist)

                p, r = self.match(nodes1, nodes2, thr=thr)

                jps.append(p)
                jrs.append(r)

            if matched % 10 == 0:
                nodes1 = self.propagateByDistance(prop_graph, prop_nodes[n1], self.prop_dist)
                nodes2 = self.propagateByDistance(gt_graph, gt_nodes[n2], self.prop_dist)

                p, r = self.match(nodes1, nodes2, thr=thr)

                ps.append(p)
                rs.append(r)

            matched += 1

            # if matched % 10 == 0:
            #     sys.stdout.write("\rmatched %d precision:%.3f recall:%.3f " % (matched, np.mean(ps), np.mean(rs)))
            #     sys.stdout.flush()

        if verbose:
            print("matched: ", matched)

            print("geo precision : ", float(matched) / len(prop_nodes))
            print("geo recall : ", float(matched) / len(gt_nodes))

            print("topo precision : ", float(matched) / len(prop_nodes) * np.mean(ps))
            print("topo recall : ", float(matched) / len(gt_nodes) * np.mean(rs))


        geo_precision = float(matched) / len(prop_nodes)
        geo_recall = float(matched) / len(gt_nodes)
        geo_f1 = f1_score(geo_precision, geo_recall)

        topo_precision = float(matched) / len(prop_nodes) * np.mean(ps)
        topo_recall = float(matched) / len(gt_nodes) * np.mean(rs)
        topo_f1 = f1_score(topo_precision, topo_recall)

        jtopo_precision = float(matched) / len(prop_nodes) * np.mean(jps)
        jtopo_recall = float(matched) / len(gt_nodes) * np.mean(jrs)
        jtopo_f1 = f1_score(jtopo_precision, jtopo_recall)

        return geo_precision, geo_recall, topo_precision, topo_recall, geo_f1, topo_f1, jtopo_f1

def f1_score(precision, recall):
    """
    Calculate F1 score given precision and recall.
    """
    if precision == 0 or recall == 0:
        return 0
    return 2 * ((precision * recall) / (precision + recall))