#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 14:05:30 2019

@author: avanetten
"""

import numpy as np
#from osgeo import gdal, ogr, osr
import scipy.spatial
import time
import os
import sys
from math import sqrt, radians, cos, sin, asin
path_apls_src = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_apls_src)


###############################################################################
def nodes_near_point(x, y, kdtree, kd_idx_dic, x_coord='x', y_coord='y',
                     n_neighbors=-1,
                     radius_m=150,
                     verbose=False):
    """
    Get nodes near the given point.

    Notes
    -----
    if n_neighbors < 0, query based on distance,
    else just return n nearest neighbors

    Arguments
    ---------
    x : float
        x coordinate of point
    y: float
        y coordinate of point
    kdtree : scipy.spatial.kdtree
        kdtree of nondes in graph
    kd_idx_dic : dict
        Dictionary mapping kdtree entry to node name
    x_coord : str
        Name of x_coordinate, can be 'x' or 'lon'. Defaults to ``'x'``.
    y_coord : str
        Name of y_coordinate, can be 'y' or 'lat'. Defaults to ``'y'``.
    n_neighbors : int
        Neareast number of neighbors to return. If < 0, ignore.
        Defaults to ``-1``.
    radius_meters : float
        Radius to search for nearest neighbors
    Returns
    -------
    kd_idx_dic, kdtree, arr : tuple
        kd_idx_dic maps kdtree entry to node name
        kdree is the actual kdtree
        arr is the numpy array of node positions
    """

    point = [x, y]

    # query kd tree for nodes of interest
    if n_neighbors > 0:
        node_names, idxs_refine, dists_m_refine = _query_kd_nearest(
            kdtree, kd_idx_dic, point, n_neighbors=n_neighbors)
    else:
        node_names, idxs_refine, dists_m_refine = _query_kd_ball(
            kdtree, kd_idx_dic, point, radius_m)

    if verbose:
        print(("subgraph node_names:", node_names))

    # get subgraph
    # G_sub = G_.subgraph(node_names)

    return node_names, dists_m_refine  # G_sub


###############################################################################
def _nodes_near_origin(G_, node, kdtree, kd_idx_dic,
                       x_coord='x', y_coord='y', radius_m=150, verbose=False):
    '''Get nodes a given radius from the desired node.  G_ should be the
    maximally simplified graph'''

    # get node coordinates
    n_props = G_.nodes[node]
    x0, y0 = n_props[x_coord], n_props[y_coord]
    point = [x0, y0]

    # query kd tree for nodes of interest
    node_names, idxs_refine, dists_m_refine = _query_kd_ball(
        kdtree, kd_idx_dic, point, radius_m)
    if verbose:
        print(("subgraph node_names:", node_names))

    # get subgraph
    # G_sub = G_.subgraph(node_names)

    return node_names, dists_m_refine  # G_sub


###############################################################################
def G_to_kdtree(G_, x_coord='x', y_coord='y', verbose=False):
    """
    Create kd tree from node positions.

    Notes
    -----
    (x, y) = (lon, lat)
    kd_idx_dic maps kdtree entry to node name:
        kd_idx_dic[i] = n (n in G.nodes())
    x_coord can be in utm (meters), or longitude

    Arguments
    ---------
    G_ : networkx graph
        Input networkx graph, with nodes assumed to have a dictioary of
        properties that includes position
    x_coord : str
        Name of x_coordinate, can be 'x' or 'lon'. Defaults to ``'x'``.
    y_coord : str
        Name of y_coordinate, can be 'y' or 'lat'. Defaults to ``'y'``.

    Returns
    -------
    kd_idx_dic, kdtree, arr : tuple
        kd_idx_dic maps kdtree entry to node name
        kdree is the actual kdtree
        arr is the numpy array of node positions
    """

    nrows = len(G_.nodes())
    ncols = 2
    kd_idx_dic = {}
    arr = np.zeros((nrows, ncols))
    # populate node array
    t1 = time.time()
    for i, n in enumerate(G_.nodes()):
        n_props = G_.nodes[n]
        if x_coord == 'lon':
            lat, lon = n_props['lat'], n_props['lon']
            x, y = lon, lat
        else:
            x, y = n_props[x_coord], n_props[y_coord]

        arr[i] = [x, y]
        kd_idx_dic[i] = n

    # now create kdtree from numpy array
    kdtree = scipy.spatial.KDTree(arr)
    if verbose:
        print("Time to create k-d tree:", time.time() - t1, "seconds")
    return kd_idx_dic, kdtree, arr



###############################################################################
def _query_kd_nearest(kdtree, kd_idx_dic, point, n_neighbors=10, distance_upper_bound=1000, keep_point=True):
    '''
    Query the kd-tree for neighbors
    Return nearest node names, distances, nearest node indexes
    If not keep_point, remove the origin point from the list
    '''

    dists_m, idxs = kdtree.query(point, k=n_neighbors, distance_upper_bound=10000000)

    idxs_refine = list(np.asarray(idxs))
    dists_m_refine = list(dists_m)

    node_names = [kd_idx_dic[i] for i in idxs_refine]

    return node_names, idxs_refine, dists_m_refine


###############################################################################
def _query_kd_ball(kdtree, kd_idx_dic, point, r_meters, keep_point=True):
    '''
    Query the kd-tree for neighbors within a distance r of the point
    Return nearest node names, distances, nearest node indexes
    if not keep_point, remove the origin point from the list
    '''

    dists_m, idxs = kdtree.query(point, k=500, distance_upper_bound=r_meters)
    # keep only points within distance and greaater than 0?
    if not keep_point:
        f0 = np.where((dists_m <= r_meters) & (dists_m > 0))
    else:
        f0 = np.where((dists_m <= r_meters))
    idxs_refine = list(np.asarray(idxs)[f0])
    dists_m_refine = list(dists_m[f0])
    node_names = [kd_idx_dic[i] for i in idxs_refine]

    return node_names, idxs_refine, dists_m_refine



###############################################################################
def _haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points in m
    on the earth (specified in decimal degrees)
    http://stackoverflow.com/questions/15736995/how-can-i-
        quickly-estimate-the-distance-between-two-latitude-longitude-points
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = list(map(radians, [lon1, lat1, lon2, lat2]))
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    km = 6367 * c
    m = 1000. * km
    return m


###############################################################################
def get_gsd(im_test_file):
    '''return gsd in meters'''
    srcImage = gdal.Open(im_test_file)
    geoTrans = srcImage.GetGeoTransform()
    ulX = geoTrans[0]
    ulY = geoTrans[3]
    # xDist = geoTrans[1]
    yDist = geoTrans[5]
    # rtnX = geoTrans[2]
    # rtnY = geoTrans[4]

    # get haversine distance
    # dx = _haversine(ulX, ulY, ulX+xDist, ulY) #haversine(lon1, lat1, lon2, lat2)
    dy = _haversine(ulX, ulY, ulX, ulY + yDist)  # haversine(lon1, lat1, lon2, lat2)

    return dy  # dx


###############################################################################
def _get_node_positions(G_, x_coord='x', y_coord='y'):
    '''Get position array for all nodes'''
    nrows = len(G_.nodes())
    ncols = 2
    arr = np.zeros((nrows, ncols))
    # populate node array
    for i, n in enumerate(G_.nodes()):
        n_props = G_.nodes[n]
        x, y = n_props[x_coord], n_props[y_coord]
        arr[i] = [x, y]
    return arr