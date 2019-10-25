#!/usr/bin/python3
from ctypes import cdll
from ctypes import cast
from ctypes import c_int, c_int64
from ctypes import c_uint, c_uint64
from ctypes import c_float, c_double
from ctypes import c_void_p, c_char_p
from ctypes import Structure
from ctypes import POINTER
from ctypes import CFUNCTYPE
from ctypes import addressof
from ctypes.util import find_library
import collections
import functools
import itertools
import warnings
import weakref
import numpy as np
import numpy.ctypeslib as npc
import struct
import math
import copy
import sys
import os

libname = "libtvg.dylib" if sys.platform == "darwin" else "libtvg.so"
filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), libname)
lib = cdll.LoadLibrary(filename)
libc = cdll.LoadLibrary(find_library('c'))

LIBTVG_API_VERSION  = 0x00000009

TVG_FLAGS_NONZERO   = 0x00000001
TVG_FLAGS_POSITIVE  = 0x00000002
TVG_FLAGS_DIRECTED  = 0x00000004
TVG_FLAGS_STREAMING = 0x00000008

TVG_FLAGS_LOAD_NEXT = 0x00010000
TVG_FLAGS_LOAD_PREV = 0x00020000

OBJECTID_NONE   = 0
OBJECTID_INT    = 1
OBJECTID_OID    = 2

class c_objectid(Structure):
    _fields_ = [("lo",       c_uint64),
                ("hi",       c_uint),
                ("type",     c_uint)]

class c_vector(Structure):
    _fields_ = [("refcount", c_uint64),
                ("flags",    c_uint),
                ("revision", c_uint64),
                ("eps",      c_float)]

class c_graph(Structure):
    _fields_ = [("refcount", c_uint64),
                ("flags",    c_uint),
                ("revision", c_uint64),
                ("eps",      c_float),
                ("ts",       c_uint64),
                ("objectid", c_objectid)]

class c_node(Structure):
    _fields_ = [("refcount", c_uint64),
                ("index",    c_uint64)]

class c_tvg(Structure):
    _fields_ = [("refcount", c_uint64),
                ("flags",    c_uint),
                ("verbosity", c_int)]

class c_mongodb_config(Structure):
    _fields_ = [("uri",          c_char_p),
                ("database",     c_char_p),
                ("col_articles", c_char_p),
                ("article_id",   c_char_p),
                ("article_time", c_char_p),
                ("filter_key",   c_char_p),
                ("filter_value", c_char_p),
                ("col_entities", c_char_p),
                ("entity_doc",   c_char_p),
                ("entity_sen",   c_char_p),
                ("entity_ent",   c_char_p),
                ("use_pool",     c_int),
                ("load_nodes",   c_int),
                ("sum_weights",  c_int),
                ("norm_weights", c_int),
                ("max_distance", c_uint64)]

class c_mongodb(Structure):
    _fields_ = [("refcount", c_uint64)]

class c_bfs_entry(Structure):
    _fields_ = [("weight",    c_double),
                ("count",     c_uint64),
                ("edge_from", c_uint64),
                ("edge_to",   c_uint64)]

# Hacky: we need optional ndpointer parameters at some places.
def or_null(t):
    class wrap:
        def from_param(cls, obj):
            if obj is None: return None
            return t.from_param(obj)
    return wrap()

c_double_p       = POINTER(c_double)
c_objectid_p     = POINTER(c_objectid)
c_vector_p       = POINTER(c_vector)
c_graph_p        = POINTER(c_graph)
c_node_p         = POINTER(c_node)
c_tvg_p          = POINTER(c_tvg)
c_mongodb_config_p = POINTER(c_mongodb_config)
c_mongodb_p      = POINTER(c_mongodb)
c_bfs_entry_p    = POINTER(c_bfs_entry)
c_bfs_callback_p = CFUNCTYPE(c_int, c_graph_p, c_bfs_entry_p, c_void_p)

# Before proceeding with any other function calls, first make sure that the library
# is compatible. This is especially important since there is no stable API yet.

lib.init_libtvg.argtypes = (c_uint64,)
lib.init_libtvg.restype = c_int

if not lib.init_libtvg(LIBTVG_API_VERSION):
    raise RuntimeError("Incompatible %s library! Try to run 'make'." % libname)

# Vector functions

lib.alloc_vector.argtypes = (c_uint,)
lib.alloc_vector.restype = c_vector_p

lib.free_vector.argtypes = (c_vector_p,)
lib.free_vector.restype = None

lib.vector_duplicate.argtypes = (c_vector_p,)
lib.vector_duplicate.restype = c_vector_p

lib.vector_memory_usage.argtypes = (c_vector_p,)
lib.vector_memory_usage.restype = c_uint64

lib.vector_clear.argtypes = (c_vector_p,)
lib.vector_clear.restype = c_int

lib.vector_set_eps.argtypes = (c_vector_p, c_float)
lib.vector_set_eps.restype = c_int

lib.vector_empty.argtypes = (c_vector_p,)
lib.vector_empty.restype = c_int

lib.vector_has_entry.argtypes = (c_vector_p, c_uint64)
lib.vector_has_entry.restype = c_int

lib.vector_get_entry.argtypes = (c_vector_p, c_uint64)
lib.vector_get_entry.restype = c_float

lib.vector_num_entries.argtypes = (c_vector_p,)
lib.vector_num_entries.restype = c_uint64

lib.vector_get_entries.argtypes = (c_vector_p, or_null(npc.ndpointer(dtype=np.uint64)), or_null(npc.ndpointer(dtype=np.float32)), c_uint64)
lib.vector_get_entries.restype = c_uint64

lib.vector_set_entry.argtypes = (c_vector_p, c_uint64, c_float)
lib.vector_set_entry.restype = c_int

lib.vector_set_entries.argtypes = (c_vector_p, npc.ndpointer(dtype=np.uint64), or_null(npc.ndpointer(dtype=np.float32)), c_uint64)
lib.vector_set_entries.restype = c_int

lib.vector_add_entry.argtypes = (c_vector_p, c_uint64, c_float)
lib.vector_add_entry.restype = c_int

lib.vector_add_entries.argtypes = (c_vector_p, npc.ndpointer(dtype=np.uint64), or_null(npc.ndpointer(dtype=np.float32)), c_uint64)
lib.vector_add_entries.restype = c_int

lib.vector_add_vector.argtypes = (c_vector_p, c_vector_p, c_float)
lib.vector_add_vector.restype = c_int

lib.vector_sub_entry.argtypes = (c_vector_p, c_uint64, c_float)
lib.vector_sub_entry.restype = c_int

lib.vector_sub_entries.argtypes = (c_vector_p, npc.ndpointer(dtype=np.uint64), or_null(npc.ndpointer(dtype=np.float32)), c_uint64)
lib.vector_sub_entries.restype = c_int

lib.vector_sub_vector.argtypes = (c_vector_p, c_vector_p, c_float)
lib.vector_sub_vector.restype = c_int

lib.vector_del_entry.argtypes = (c_vector_p, c_uint64)
lib.vector_del_entry.restype = c_int

lib.vector_del_entries.argtypes = (c_vector_p, npc.ndpointer(dtype=np.uint64), c_uint64)
lib.vector_del_entries.restype = c_int

lib.vector_mul_const.argtypes = (c_vector_p, c_float)
lib.vector_mul_const.restype = c_int

lib.vector_sum_weights.argtypes = (c_vector_p,)
lib.vector_sum_weights.restype = c_double

lib.vector_norm.argtypes = (c_vector_p,)
lib.vector_norm.restype = c_double

lib.vector_mul_vector.argtypes = (c_vector_p, c_vector_p)
lib.vector_mul_vector.restype = c_double

lib.vector_sub_vector_norm.argtypes = (c_vector_p, c_vector_p)
lib.vector_sub_vector_norm.restype = c_double

# Graph functions

lib.alloc_graph.argtypes = (c_uint,)
lib.alloc_graph.restype = c_graph_p

lib.free_graph.argtypes = (c_graph_p,)
lib.free_graph.restype = None

lib.unlink_graph.argtypes = (c_graph_p,)
lib.unlink_graph.restype = None

lib.graph_duplicate.argtypes = (c_graph_p,)
lib.graph_duplicate.restype = c_graph_p

lib.graph_clear.argtypes = (c_graph_p,)
lib.graph_clear.restype = c_int

lib.graph_memory_usage.argtypes = (c_graph_p,)
lib.graph_memory_usage.restype = c_uint64

lib.prev_graph.argtypes = (c_graph_p,)
lib.prev_graph.restype = c_graph_p

lib.next_graph.argtypes = (c_graph_p,)
lib.next_graph.restype = c_graph_p

lib.graph_set_eps.argtypes = (c_graph_p, c_float)
lib.graph_set_eps.restype = c_int

lib.graph_empty.argtypes = (c_graph_p,)
lib.graph_empty.restype = c_int

lib.graph_has_edge.argtypes = (c_graph_p, c_uint64, c_uint64)
lib.graph_has_edge.restype = c_int

lib.graph_get_edge.argtypes = (c_graph_p, c_uint64, c_uint64)
lib.graph_get_edge.restype = c_float

lib.graph_num_edges.argtypes = (c_graph_p,)
lib.graph_num_edges.restype = c_uint64

lib.graph_get_edges.argtypes = (c_graph_p, or_null(npc.ndpointer(dtype=np.uint64)), or_null(npc.ndpointer(dtype=np.float32)), c_uint64)
lib.graph_get_edges.restype = c_uint64

lib.graph_get_top_edges.argtypes = (c_graph_p, or_null(npc.ndpointer(dtype=np.uint64)), or_null(npc.ndpointer(dtype=np.float32)), c_uint64)
lib.graph_get_top_edges.restype = c_uint64

lib.graph_get_adjacent_edges.argtypes = (c_graph_p, c_uint64, or_null(npc.ndpointer(dtype=np.uint64)), or_null(npc.ndpointer(dtype=np.float32)), c_uint64)
lib.graph_get_adjacent_edges.restype = c_uint64

lib.graph_set_edge.argtypes = (c_graph_p, c_uint64, c_uint64, c_float)
lib.graph_set_edge.restype = c_int

lib.graph_set_edges.argtypes = (c_graph_p, npc.ndpointer(dtype=np.uint64), or_null(npc.ndpointer(dtype=np.float32)), c_uint64)
lib.graph_set_edges.restype = c_int

lib.graph_add_edge.argtypes = (c_graph_p, c_uint64, c_uint64, c_float)
lib.graph_add_edge.restype = c_int

lib.graph_add_edges.argtypes = (c_graph_p, npc.ndpointer(dtype=np.uint64), or_null(npc.ndpointer(dtype=np.float32)), c_uint64)
lib.graph_add_edges.restype = c_int

lib.graph_add_graph.argtypes = (c_graph_p, c_graph_p, c_float)
lib.graph_add_graph.restype = c_int

lib.graph_sub_edge.argtypes = (c_graph_p, c_uint64, c_uint64, c_float)
lib.graph_sub_edge.restype = c_int

lib.graph_sub_edges.argtypes = (c_graph_p, npc.ndpointer(dtype=np.uint64), or_null(npc.ndpointer(dtype=np.float32)), c_uint64)
lib.graph_sub_edges.restype = c_int

lib.graph_sub_graph.argtypes = (c_graph_p, c_graph_p, c_float)
lib.graph_sub_graph.restype = c_int

lib.graph_del_edge.argtypes = (c_graph_p, c_uint64, c_uint64)
lib.graph_del_edge.restype = c_int

lib.graph_del_edges.argtypes = (c_graph_p, npc.ndpointer(dtype=np.uint64), c_uint64)
lib.graph_del_edges.restype = c_int

lib.graph_mul_const.argtypes = (c_graph_p, c_float)
lib.graph_mul_const.restype = c_int

lib.graph_mul_vector.argtypes = (c_graph_p, c_vector_p)
lib.graph_mul_vector.restype = c_vector_p

lib.graph_in_degrees.argtypes = (c_graph_p,)
lib.graph_in_degrees.restype = c_vector_p

lib.graph_in_weights.argtypes = (c_graph_p,)
lib.graph_in_weights.restype = c_vector_p

lib.graph_out_degrees.argtypes = (c_graph_p,)
lib.graph_out_degrees.restype = c_vector_p

lib.graph_out_weights.argtypes = (c_graph_p,)
lib.graph_out_weights.restype = c_vector_p

lib.graph_degree_anomalies.argtypes = (c_graph_p,)
lib.graph_degree_anomalies.restype = c_vector_p

lib.graph_weight_anomalies.argtypes = (c_graph_p,)
lib.graph_weight_anomalies.restype = c_vector_p

lib.graph_sum_weights.argtypes = (c_graph_p,)
lib.graph_sum_weights.restype = c_double

lib.graph_power_iteration.argtypes = (c_graph_p, c_vector_p, c_uint, c_double, c_double_p)
lib.graph_power_iteration.restype = c_vector_p

lib.graph_filter_nodes.argtypes = (c_graph_p, c_vector_p)
lib.graph_filter_nodes.restype = c_graph_p

lib.graph_normalize.argtypes = (c_graph_p,)
lib.graph_normalize.restype = c_graph_p

lib.graph_bfs.argtypes = (c_graph_p, c_uint64, c_int, c_bfs_callback_p, c_void_p)
lib.graph_bfs.restype = c_int

lib.graph_get_distance_count.argtypes = (c_graph_p, c_uint64, c_uint64)
lib.graph_get_distance_count.restype = c_uint64

lib.graph_get_distance_weight.argtypes = (c_graph_p, c_uint64, c_uint64)
lib.graph_get_distance_weight.restype = c_double

# Node functions

lib.alloc_node.argtypes = ()
lib.alloc_node.restype = c_node_p

lib.free_node.argtypes = (c_node_p,)
lib.free_node.restype = None

lib.unlink_node.argtypes = (c_node_p,)
lib.unlink_node.restype = None

lib.node_set_attribute.argtypes = (c_node_p, c_char_p, c_char_p)
lib.node_set_attribute.restype = c_int

lib.node_get_attribute.argtypes = (c_node_p, c_char_p)
lib.node_get_attribute.restype = c_char_p

lib.node_get_attributes.argtypes = (c_node_p,)
lib.node_get_attributes.restype = POINTER(c_char_p)

# TVG functions

lib.alloc_tvg.argtypes = (c_uint,)
lib.alloc_tvg.restype = c_tvg_p

lib.free_tvg.argtypes = (c_tvg_p,)
lib.free_tvg.restype = None

lib.tvg_set_verbosity.argtypes = (c_tvg_p, c_int)
lib.tvg_set_verbosity.restype = None

lib.tvg_memory_usage.argtypes = (c_tvg_p,)
lib.tvg_memory_usage.restype = c_uint64

lib.tvg_link_graph.argtypes = (c_tvg_p, c_graph_p, c_uint64)
lib.tvg_link_graph.restype = c_int

lib.tvg_set_primary_key.argtypes = (c_tvg_p, c_char_p)
lib.tvg_set_primary_key.restype = c_int

lib.tvg_link_node.argtypes = (c_tvg_p, c_node_p, POINTER(c_node_p), c_uint64)
lib.tvg_link_node.restype = c_int

lib.tvg_get_node_by_index.argtypes = (c_tvg_p, c_uint64)
lib.tvg_get_node_by_index.restype = c_node_p

lib.tvg_get_node_by_primary_key.argtypes = (c_tvg_p, c_node_p)
lib.tvg_get_node_by_primary_key.restype = c_node_p

lib.tvg_load_graphs_from_file.argtypes = (c_tvg_p, c_char_p)
lib.tvg_load_graphs_from_file.restype = c_int

lib.tvg_load_nodes_from_file.argtypes = (c_tvg_p, c_char_p, c_char_p)
lib.tvg_load_nodes_from_file.restype = c_int

lib.tvg_enable_mongodb_sync.argtypes = (c_tvg_p, c_mongodb_p, c_uint64, c_uint64)
lib.tvg_enable_mongodb_sync.restype = c_int

lib.tvg_disable_mongodb_sync.argtypes = (c_tvg_p,)
lib.tvg_disable_mongodb_sync.restype = None

lib.tvg_enable_query_cache.argtypes = (c_tvg_p, c_uint64)
lib.tvg_enable_query_cache.restype = c_int

lib.tvg_disable_query_cache.argtypes = (c_tvg_p,)
lib.tvg_disable_query_cache.restype = None

lib.tvg_invalidate_queries.argtypes = (c_tvg_p, c_uint64, c_uint64)
lib.tvg_invalidate_queries.restype = None

lib.tvg_lookup_graph_ge.argtypes = (c_tvg_p, c_uint64)
lib.tvg_lookup_graph_ge.restype = c_graph_p

lib.tvg_lookup_graph_le.argtypes = (c_tvg_p, c_uint64)
lib.tvg_lookup_graph_le.restype = c_graph_p

lib.tvg_lookup_graph_near.argtypes = (c_tvg_p, c_uint64)
lib.tvg_lookup_graph_near.restype = c_graph_p

lib.tvg_compress.argtypes = (c_tvg_p, c_uint64, c_uint64)
lib.tvg_compress.restype = c_int

# Query functions

lib.tvg_sum_edges.argtypes = (c_tvg_p, c_uint64, c_uint64, c_float)
lib.tvg_sum_edges.restype = c_graph_p

lib.tvg_sum_edges_exp.argtypes = (c_tvg_p, c_uint64, c_uint64, c_float, c_float, c_float)
lib.tvg_sum_edges_exp.restype = c_graph_p

lib.tvg_count_edges.argtypes = (c_tvg_p, c_uint64, c_uint64)
lib.tvg_count_edges.restype = c_graph_p

lib.tvg_count_nodes.argtypes = (c_tvg_p, c_uint64, c_uint64)
lib.tvg_count_nodes.restype = c_vector_p

lib.tvg_count_graphs.argtypes = (c_tvg_p, c_uint64, c_uint64)
lib.tvg_count_graphs.restype = c_uint64

lib.tvg_topics.argtypes = (c_tvg_p, c_uint64, c_uint64, c_uint64, c_uint64)
lib.tvg_topics.restype = c_graph_p

# Metric functions

lib.metric_graph_avg.argtypes = (POINTER(c_graph_p), c_uint64)
lib.metric_graph_avg.restype = c_graph_p

lib.metric_vector_avg.argtypes = (POINTER(c_vector_p), c_uint64)
lib.metric_vector_avg.restype = c_vector_p

lib.metric_graph_std.argtypes = (POINTER(c_graph_p), c_uint64)
lib.metric_graph_std.restype = c_graph_p

lib.metric_vector_std.argtypes = (POINTER(c_vector_p), c_uint64)
lib.metric_vector_std.restype = c_vector_p

lib.metric_graph_pareto.argtypes = (c_graph_p, c_graph_p, c_int, c_int, c_float)
lib.metric_graph_pareto.restype = c_graph_p

lib.metric_vector_pareto.argtypes = (c_vector_p, c_vector_p, c_int, c_int, c_float)
lib.metric_vector_pareto.restype = c_vector_p

# MongoDB functions

lib.alloc_mongodb.argtypes = (c_mongodb_config_p,)
lib.alloc_mongodb.restype = c_mongodb_p

lib.free_mongodb.argtypes = (c_mongodb_p,)
lib.free_mongodb.restype = None

lib.mongodb_load_graph.argtypes = (c_tvg_p, c_mongodb_p, c_objectid_p, c_uint)
lib.mongodb_load_graph.restype = c_graph_p

lib.tvg_load_graphs_from_mongodb.argtypes = (c_tvg_p, c_mongodb_p)
lib.tvg_load_graphs_from_mongodb.restype = c_int

# libc functions

libc.free.argtypes = (c_void_p,)
libc.free.restype = None

# The 'cacheable' decorator can be used on Vector and Graph objects to cache the result
# of a function call as long as the underlying vector/graph has not changed. This is
# ensured by storing and comparing the revision number embedded in the object header.
def cacheable(func):
    cache = weakref.WeakKeyDictionary()
    @functools.wraps(func)
    def wrapper(self, drop_cache=False):
        try:
            old_revision, result = cache[self]
        except KeyError:
            old_revision, result = (None, None)
        new_revision = self._obj.contents.revision
        if old_revision != new_revision or drop_cache:
            result = func(self) # FIXME: Support *args, **kwargs.
            cache[self] = (new_revision, result)
        return result
    return wrapper

# The 'libtvgobject' decorator should be used for classes corresponding to objects in
# the libtvg library. It ensures that there is at most one Python object corresponding
# to each libtvg object.
def libtvgobject(klass):
    cache = weakref.WeakValueDictionary()
    @functools.wraps(klass, assigned=('__name__', '__module__'), updated=())
    class wrapper(klass):
        __doc__ = klass.__doc__
        def __new__(cls, *args, obj=None, **kwargs):
            if obj:
                try:
                    return cache[addressof(obj.contents)]._get_obj()
                except KeyError:
                    pass
            result = klass(*args, obj=obj, **kwargs)
            result.__class__ = cls
            cache[addressof(result._obj.contents)] = result
            return result
        def __init__(self, *args, **kwargs):
            pass
    return wrapper

def _convert_values(values):
    if isinstance(values, dict):
        return values

    result = collections.defaultdict(list)
    for step, v in enumerate(values):
        if isinstance(v, Vector):
            indices, weights = v.entries()
            for i, w in zip(indices, weights):
                result[i] += [0.0] * (step - len(result[i]))
                result[i].append(w)

        elif isinstance(v, Graph):
            indices, weights = v.edges()
            for i, w in zip(indices, weights):
                i = tuple(i)
                result[i] += [0.0] * (step - len(result[i]))
                result[i].append(w)

        else:
            for i, w in v.items():
                result[i] += [0.0] * (step - len(result[i]))
                result[i].append(w)

    for i in result.keys():
        result[i] += [0.0] * (len(values) - len(result[i]))

    return result

def metric_entropy(values, num_bins=50):
    """
    Rate the importance / interestingness of individual nodes/edges by their entropy.

    # Arguments
    values: Values for each node or edge.
    num_bins: Number of bins used to create the entropy model.

    # Returns
    Dictionary containing the metric for each node or edge.
    """

    values = _convert_values(values)
    if len(values) == 0:
        return {}

    data = []
    for i in values.keys():
        data += values[i]

    prob, bin_edges = np.histogram(data, bins=num_bins)
    prob = np.array(prob, dtype=float) / np.sum(prob)
    entropy = (- np.ma.log(prob) * prob).filled(0)

    @np.vectorize
    def to_entropy(x):
        index = np.searchsorted(bin_edges, x)
        if index >= 1: index -= 1
        return entropy[index]

    result = {}
    for i in values.keys():
        result[i] = np.sum(to_entropy(values[i]))

    return result

def metric_entropy_local(values, num_bins=50):
    """
    Like metric_entropy(), but train a separate model for each time step.

    # Arguments
    values: Values for each node or edge.
    num_bins: Number of bins used to create the entropy model.

    # Returns
    Dictionary containing the metric for each node or edge.
    """

    values = _convert_values(values)
    if len(values) == 0:
        return {}

    sample_steps = len(next(iter(values.values())))

    result = collections.defaultdict(float)
    for j in range(sample_steps):

        data = []
        for i in values.keys():
            data.append(values[i][j])

        prob, bin_edges = np.histogram(data, bins=num_bins)
        prob = np.array(prob, dtype=float) / np.sum(prob)
        entropy = (- np.ma.log(prob) * prob).filled(0)

        def to_entropy(x):
            index = np.searchsorted(bin_edges, x)
            if index >= 1: index -= 1
            return entropy[index]

        for i in values.keys():
            result[i] += to_entropy(values[i][j])

    return result

def metric_entropy_2d(values, num_bins=50):
    """
    Like metric_entropy(), but train a 2-dimensional model for entropy estimations.

    # Arguments
    values: Values for each node or edge.
    num_bins: Number of bins used to create the entropy model.

    # Returns
    Dictionary containing the metric for each node or edge.
    """

    values = _convert_values(values)
    if len(values) == 0:
        return {}

    data_x = []
    data_y = []
    for i in values.keys():
        data_x += values[i][:-1]
        data_y += values[i][1:]

    prob, bin_edges_x, bin_edges_y = np.histogram2d(data_x, data_y, bins=num_bins)

    prob = np.array(prob, dtype=float) / np.sum(prob)
    entropy = (- np.ma.log(prob) * prob).filled(0)

    @np.vectorize
    def to_entropy(x, y):
        index_x = np.searchsorted(bin_edges_x, x)
        index_y = np.searchsorted(bin_edges_y, y)
        if index_x >= 1: index_x -= 1
        if index_y >= 1: index_y -= 1
        return entropy[index_x, index_y]

    result = {}
    for i in values.keys():
        result[i] = np.sum(to_entropy(values[i][:-1], values[i][1:]))

    return result

def metric_trend(values):
    """
    Rate the importance / interestingness of individual nodes/edges by their trend.

    # Arguments
    values: Values for each node or edge.

    # Returns
    Dictionary containing the metric for each node or edge.
    """

    values = _convert_values(values)
    if len(values) == 0:
        return {}

    sample_steps = len(next(iter(values.values())))

    A = np.zeros((sample_steps, 2))
    A[:, 0] = 1.0
    A[:, 1] = range(A.shape[0])

    kwargs = {}
    if np.lib.NumpyVersion(np.__version__) >= '1.14.0':
        kwargs['rcond'] = None

    result = {}
    for i in values.keys():
        result[i] = np.linalg.lstsq(A, values[i], **kwargs)[0][1]

    return result

def metric_stability_ratio(values):
    """
    Rate the stability of individual nodes/edges by their inverse relative standard deviation.

    # Arguments
    values: Values for each node or edge.

    # Returns
    Dictionary containing the metric for each node or edge.
    """

    values = _convert_values(values)
    if len(values) == 0:
        return {}

    result = {}
    for i in values.keys():
        std = np.std(values[i])
        if std > 0.0:
            result[i] = np.abs(np.mean(values[i])) / std
        else:
            result[i] = np.inf

    return result

def metric_avg(values):
    """
    Compute the average of individual nodes/edges.

    # Arguments
    values: Values for each node or edge.

    # Returns
    Average for each node or edge.
    """

    if not isinstance(values, dict):
        # Fast-path for list of Graphs and list of Vectors.

        if all([isinstance(v, Graph) for v in values]):
            objs = (c_graph_p * len(values))()
            for i, v in enumerate(values):
                objs[i] = v._obj
            return Graph(obj=lib.metric_graph_avg(objs, len(values)))

        if all([isinstance(v, Vector) for v in values]):
            objs = (c_vector_p * len(values))()
            for i, v in enumerate(values):
                objs[i] = v._obj
            return Vector(obj=lib.metric_vector_avg(objs, len(values)))

    values = _convert_values(values)
    if len(values) == 0:
        return {}

    result = {}
    for i in values.keys():
        result[i] = np.mean(values[i])

    return result

def metric_std(values):
    """
    Compute the standard deviation of individual nodes/edges

    # Arguments
    values: Values for each node or edge.

    # Returns
    Standard deviation for each node or edge.
    """

    if not isinstance(values, dict):
        # Fast-path for list of Graphs and list of Vectors.

        if all([isinstance(v, Graph) for v in values]):
            objs = (c_graph_p * len(values))()
            for i, v in enumerate(values):
                objs[i] = v._obj
            return Graph(obj=lib.metric_graph_std(objs, len(values)))

        if all([isinstance(v, Vector) for v in values]):
            objs = (c_vector_p * len(values))()
            for i, v in enumerate(values):
                objs[i] = v._obj
            return Vector(obj=lib.metric_vector_std(objs, len(values)))

    values = _convert_values(values)
    if len(values) == 0:
        return {}

    result = {}
    for i in values.keys():
        result[i] = np.std(values[i], ddof=1)

    return result

def metric_pareto(values, maximize=True, base=0.0):
    """
    Compute the pareto ranking of two graphs or vectors.

    # Arguments:
    values: Values for each node or edge.
    maximize: Defines which values should be maximized/minimized.
    base: Use `base**(index - 1)` as weight instead of `index`.

    # Returns
    Metric for each node or edge.
    """

    if maximize in [True, False]:
        maximize = [maximize] * len(values)

    if len(maximize) != len(values):
        raise NotImplementedError("Wrong number of maximize parameters")

    if len(values) == 2 and all([isinstance(v, Graph) for v in values]):
        return Graph(obj=lib.metric_graph_pareto(values[0]._obj, values[1]._obj,
                                                 maximize[0], maximize[1], base))

    if len(values) == 2 and all([isinstance(v, Vector) for v in values]):
        return Vector(obj=lib.metric_vector_pareto(values[0]._obj, values[1]._obj,
                                                   maximize[0], maximize[1], base))

    values = _convert_values(values)
    if len(values) == 0:
        return {}

    weights = np.array([(-1 if m else 1) for m in maximize])

    nodes = []
    costs = []
    for i in values.keys():
        nodes.append(i)
        costs.append(np.multiply(values[i], weights))

    nodes = np.array(nodes)
    costs = np.array(costs)
    weight = 1.0

    result = {}
    while len(nodes):
        front = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if front[i]:
                front[front] = np.any(costs[front] <  c, axis=1) | \
                               np.all(costs[front] <= c, axis=1)

        for i in np.where(front)[0]:
            key = nodes[i]
            if isinstance(key, np.ndarray):
                key = tuple(key)
            result[key] = weight

        nodes = nodes[~front]
        costs = costs[~front]
        if base == 0.0:
            weight += 1.0
        else:
            weight *= base

    return result

def metric_stability_pareto(values, base=0.0):
    """
    Rate the stability of individual nodes/edges by ranking their average and standard deviation.

    # Arguments
    values: Values for each node or edge.
    base: Use `base**(index - 1)` as weight instead of `index`.

    # Returns
    Metric for each node or edge.
    """

    avg = metric_avg(values)
    std = metric_std(values)
    return metric_pareto([avg, std], maximize=[True, False], base=base)

@libtvgobject
class Vector(object):
    """
    This object represents a vector of arbitrary / infinite dimension. To achieve that,
    it only stores entries that are explicitly set, and assumes that all other entries
    of the vector are zero. Internally, it uses hashing to map indices to buckets,
    that are stored in contiguous blocks of memory and in sorted order for faster access.

    # Arguments
    nonzero: Enforce that all entries must be non-zero.
    positive: Enforce that all entries must be positive.
    """

    def __init__(self, nonzero=False, positive=False, obj=None):
        if obj is None:
            flags = 0
            flags |= (TVG_FLAGS_NONZERO  if nonzero  else 0)
            flags |= (TVG_FLAGS_POSITIVE if positive else 0)
            obj = lib.alloc_vector(flags)

        self._obj = obj
        if not obj:
            raise MemoryError

    def __del__(self):
        if lib is None:
            return
        if self._obj:
            lib.free_vector(self._obj)
            self._obj = None

    def _get_obj(self):
        assert self._obj.contents.refcount >= 2
        lib.free_vector(self._obj)
        return self

    @cacheable
    def __repr__(self):
        max_entries = 10
        indices = np.empty(shape=(max_entries,), dtype=np.uint64,  order='C')
        weights = np.empty(shape=(max_entries,), dtype=np.float32, order='C')
        num_entries = lib.vector_get_entries(self._obj, indices, weights, max_entries)

        out = []
        for i in range(min(num_entries, max_entries)):
            out.append("%d: %f" % (indices[i], weights[i]))
        if num_entries > max_entries:
            out.append("...")

        return "Vector({%s})" % ", ".join(out)

    @property
    def flags(self):
        return self._obj.contents.flags

    @property
    def revision(self):
        """
        Return the current revision of the vector object. This value is incremented
        whenever the vector is changed. It is also used by the @cacheable decorator
        to check the cache validity.
        """
        return self._obj.contents.revision

    @property
    def eps(self):
        """
        Get/set the current value of epsilon. This is used to determine whether an
        entry is equal to zero. Whenever |x| < eps, it is treated as zero.
        """
        return self._obj.contents.eps

    @eps.setter
    def eps(self, value):
        res = lib.vector_set_eps(self._obj, value)
        if not res:
            raise MemoryError

    @property
    @cacheable
    def memory_usage(self):
        """ Return the memory usage currently associated with the vector. """
        return lib.vector_memory_usage(self._obj)

    @cacheable
    def empty(self):
        """ Check if a vector is empty, i.e., if it does not have any entries. """
        return lib.vector_empty(self._obj)

    def duplicate(self):
        """ Create an independent copy of the vector. """
        return Vector(obj=lib.vector_duplicate(self._obj))

    def clear(self):
        """ Clear all entries of the vector object. """
        res = lib.vector_clear(self._obj)
        if not res:
            raise RuntimeError

    def has_entry(self, index):
        """ Check if a vector has an entry with index `index`. """
        return lib.vector_has_entry(self._obj, index)

    def __getitem__(self, index):
        """ Return entry `index` of the vector, or 0 if it doesn't exist. """
        return lib.vector_get_entry(self._obj, index)

    def entries(self, ret_indices=True, ret_weights=True, as_dict=False):
        """
        Return all indices and/or weights of a vector.

        # Arguments
        ret_indices: Return indices, otherwise None.
        ret_weights: Return weights, otherwise None.
        as_dict: Return result as dictionary instead of tuple.

        # Returns
        `(indices, weights)` or dictionary
        """

        if as_dict and not ret_indices:
            raise ValueError("Invalid parameter combination")

        num_entries = self.num_entries
        while True:
            max_entries = num_entries
            indices = np.empty(shape=(max_entries,), dtype=np.uint64,  order='C') if ret_indices else None
            weights = np.empty(shape=(max_entries,), dtype=np.float32, order='C') if ret_weights else None
            num_entries = lib.vector_get_entries(self._obj, indices, weights, max_entries)
            if num_entries <= max_entries:
                break

        if indices is not None:
            indices.resize((num_entries,), refcheck=False)
        if weights is not None:
            weights.resize((num_entries,), refcheck=False)

        if as_dict:
            if weights is None:
                weights = [None] * num_entries
            return dict(zip(indices, weights))

        return indices, weights

    def keys(self):
        """ Iterate over indices of a vector. """
        indices, _ = self.entries(ret_weights=False)
        return iter(indices)

    def values(self):
        """ Iterate over weights of a vector. """
        _, weights = self.entries(ret_indices=False)
        return iter(weights)

    def items(self):
        """ Iterate over indices and weights of a vector. """
        indices, weights = self.entries()
        return zip(indices, weights)

    @property
    @cacheable
    def num_entries(self):
        """ Return the number of entries of a vector. """
        return lib.vector_num_entries(self._obj)

    def __len__(self):
        """ Return the number of entries of a vector. """
        return self.num_entries

    def __setitem__(self, index, weight):
        """ Set the entry with index `index` of a vector to `weight`. """
        res = lib.vector_set_entry(self._obj, index, weight)
        if not res:
            raise MemoryError

    @staticmethod
    def _convert_indices_weights(indices, weights=None):
        if weights is not None:
            indices = np.asarray(indices, dtype=np.uint64, order='C')
            weights = np.asarray(weights, dtype=np.float32, order='C')

            if indices.size == 0 and weights.size == 0:
                return None, None
            if len(indices.shape) != 1:
                raise ValueError("indices array does not have correct dimensions")
            if len(weights.shape) != 1:
                raise ValueError("weights array does not have correct dimensions")
            if indices.shape[0] != weights.shape[0]:
                raise ValueError("indices/weights arrays have different length")

        elif isinstance(indices, dict):
            entries = indices
            if len(entries) == 0:
                return None, None

            indices = np.empty(shape=(len(entries),), dtype=np.uint64, order='C')
            weights = np.empty(shape=(len(entries),), dtype=np.float32, order='C')
            for j, (i, w) in enumerate(entries.items()):
                indices[j] = i
                weights[j] = w

        else:
            if isinstance(indices, set):
                indices = list(indices)
            indices = np.asarray(indices, dtype=np.uint64, order='C')

            if indices.size == 0:
                return None, None
            if len(indices.shape) != 1:
                raise ValueError("indices array does not have correct dimensions")

        return indices, weights

    def set_entries(self, indices, weights=None):
        """
        Short-cut to set multiple entries of a vector.
        If weights is None the elements are set to 1.

        # Arguments
        indices: List of indices (list or 1d numpy array).
        weights: List of weights to set (list or 1d numpy array).
        """

        indices, weights = self._convert_indices_weights(indices, weights)
        if indices is None:
            return

        res = lib.vector_set_entries(self._obj, indices, weights, indices.shape[0])
        if not res:
            raise MemoryError

    def add_entry(self, index, weight):
        """ Add weight `weight` to the entry with index `index`. """
        res = lib.vector_add_entry(self._obj, index, weight)
        if not res:
            raise MemoryError

    def add_entries(self, indices, weights=None):
        """
        Short-cut to update multiple entries of a vector by adding values.
        If weights is None the elements are set to 1.

        # Arguments
        indices: List of indices (list or 1d numpy array).
        weights: List of weights to add (list or 1d numpy array).
        """

        indices, weights = self._convert_indices_weights(indices, weights)
        if indices is None:
            return

        res = lib.vector_add_entries(self._obj, indices, weights, indices.shape[0])
        if not res:
            raise MemoryError

    def add_vector(self, other, weight=1.0):
        """ Add entries specified by a second vector, optionally multiplied by `weight`. """
        res = lib.vector_add_vector(self._obj, other._obj, weight)
        if not res:
            raise MemoryError

    def __add__(self, other):
        result = self.duplicate()
        result.add_vector(other)
        return result

    def sub_entry(self, index, weight):
        """ Subtract weight `weight` from the entry with index `index`. """
        res = lib.vector_sub_entry(self._obj, index, weight)
        if not res:
            raise MemoryError

    def sub_entries(self, indices, weights=None):
        """
        Short-cut to update multiple entries of a vector by subtracting values.
        If weights is None the elements are set to 1.

        # Arguments
        indices: List of indices (list or 1d numpy array).
        weights: List of weights to subtract (list or 1d numpy array).
        """

        indices, weights = self._convert_indices_weights(indices, weights)
        if indices is None:
            return

        res = lib.vector_sub_entries(self._obj, indices, weights, indices.shape[0])
        if not res:
            raise MemoryError

    def sub_vector(self, other, weight=1.0):
        """ Subtract entries specified by a second vector, optionally multiplied by `weight`. """
        res = lib.vector_sub_vector(self._obj, other._obj, weight)
        if not res:
            raise MemoryError

    def __sub__(self, other):
        result = self.duplicate()
        result.sub_vector(other)
        return result

    def __delitem__(self, index):
        """ Delete entry `index` from the vector or do nothing if it doesn't exist. """
        res = lib.vector_del_entry(self._obj, index)
        if not res:
            raise RuntimeError

    def del_entries(self, indices):
        """
        Short-cut to delete multiple entries from a vector.

        # Arguments
        indices: List of indices (list or 1d numpy array).
        """

        if isinstance(indices, set):
            indices = list(indices)
        indices = np.asarray(indices, dtype=np.uint64, order='C')

        if indices.size == 0:
            return # nothing to do for empty array
        if len(indices.shape) != 1:
            raise ValueError("indices array does not have correct dimensions")

        res = lib.vector_del_entries(self._obj, indices, indices.shape[0])
        if not res:
            raise RuntimeError

    def mul_const(self, constant):
        """ Perform inplace element-wise multiplication of the vector with `constant`. """
        res = lib.vector_mul_const(self._obj, constant)
        if not res:
            raise RuntimeError

    @cacheable
    def sum_weights(self):
        """ Compute the sum of all weights. """
        return lib.vector_sum_weights(self._obj)

    @cacheable
    def norm(self):
        """ Return the L2 norm of the vector. """
        return lib.vector_norm(self._obj)

    def mul_vector(self, other):
        """ Compute the scalar product of the current vector with a second vector `other`. """
        # FIXME: Check type of 'other'.
        return lib.vector_mul_vector(self._obj, other._obj)

    def sub_vector_norm(self, other):
        """ Compute L2 norm of (self - other). """
        return lib.vector_sub_vector_norm(self._obj, other._obj)

    def as_dict(self):
        """ Return a dictionary containing all vector entries. """
        return self.entries(as_dict=True)

    @staticmethod
    def from_dict(entries, *args, **kwargs):
        """ Generate a Vector object from a dictionary. """
        vector = Vector(*args, **kwargs)
        vector.set_entries(entries)
        return vector

@libtvgobject
class Graph(object):
    """
    This object represents a graph of arbitrary / infinite dimension. To achieve that,
    it only stores edges that are explicitly set, and assumes that all other edges
    of the graph have a weight of zero. Internally, it uses hashing to map source and
    target indices to buckets, that are stored in contiguous blocks of memory and in
    sorted order for faster access.

    # Arguments
    nonzero: Enforce that all entries must be non-zero.
    positive: Enforce that all entries must be positive.
    directed: Create a directed graph.
    """

    def __init__(self, nonzero=False, positive=False, directed=False, obj=None):
        if obj is None:
            flags = 0
            flags |= (TVG_FLAGS_NONZERO  if nonzero  else 0)
            flags |= (TVG_FLAGS_POSITIVE if positive else 0)
            flags |= (TVG_FLAGS_DIRECTED if directed else 0)
            obj = lib.alloc_graph(flags)

        self._obj = obj
        if not obj:
            raise MemoryError

    def __del__(self):
        if lib is None:
            return
        if self._obj:
            lib.free_graph(self._obj)
            self._obj = None

    def _get_obj(self):
        assert self._obj.contents.refcount >= 2
        lib.free_graph(self._obj)
        return self

    @cacheable
    def __repr__(self):
        max_edges = 10
        indices = np.empty(shape=(max_edges, 2), dtype=np.uint64,  order='C')
        weights = np.empty(shape=(max_edges,),   dtype=np.float32, order='C')
        num_edges = lib.graph_get_edges(self._obj, indices, weights, max_edges)

        out = []
        for i in range(min(num_edges, max_edges)):
            out.append("(%d, %d): %f" % (indices[i][0], indices[i][1], weights[i]))
        if num_edges > max_edges:
            out.append("...")

        return "Graph({%s})" % ", ".join(out)

    @property
    def flags(self):
        return self._obj.contents.flags

    @property
    def directed(self):
        return (self._obj.contents.flags & TVG_FLAGS_DIRECTED) != 0

    @property
    def revision(self):
        """
        Return the current revision of the graph object. This value is incremented
        whenever the graph is changed. It is also used by the @cacheable decorator
        to check the cache validity.
        """
        return self._obj.contents.revision

    @property
    def eps(self):
        """
        Get/set the current value of epsilon. This is used to determine whether an
        entry is equal to zero. Whenever |x| < eps, it is treated as zero.
        """
        return self._obj.contents.eps

    @eps.setter
    def eps(self, value):
        res = lib.graph_set_eps(self._obj, value)
        if not res:
            raise RuntimeError

    @property
    def ts(self):
        """
        Get the timestamp associated with this graph object. This only applies to
        objects that are part of a time-varying graph.
        """
        return self._obj.contents.ts

    @property
    def id(self):
        """
        Get the ID associated with this graph object. This only applies to objects
        loaded from an external data source, e.g., from a MongoDB.
        """

        objectid = self._obj.contents.objectid
        if objectid.type == OBJECTID_NONE:
            return None
        if objectid.type == OBJECTID_INT:
            return objectid.lo
        if objectid.type == OBJECTID_OID:
            return struct.pack(">IQ", objectid.hi, objectid.lo).hex()

        raise NotImplementedError

    @staticmethod
    def load_from_file(filename, nonzero=False, positive=False, directed=False):
        raise NotImplementedError

    @staticmethod
    def load_from_mongodb(mongodb, id, nonzero=False, positive=False, directed=False):
        """
        Load a single graph from a MongoDB database.

        # Arguments
        id: Identifier (numeric or objectid) of the document to load
        nonzero: Enforce that all entries must be non-zero.
        positive: Enforce that all entries must be positive.
        directed: Create a directed graph.
        """

        objectid = c_objectid()

        if isinstance(id, int):
            objectid.lo = id
            objectid.hi = 0
            objectid.type = OBJECTID_INT

        elif isinstance(id, bytes) and len(id) == 12:
            hi, lo = struct.unpack(">IQ", id)
            objectid.lo = lo
            objectid.hi = hi
            objectid.type = OBJECTID_OID

        elif isinstance(id, str) and len(id) == 24:
            hi, lo = struct.unpack(">IQ", bytes.fromhex(id))
            objectid.lo = lo
            objectid.hi = hi
            objectid.type = OBJECTID_OID

        else:
            raise ValueError("Objectid is not valid")

        flags = 0
        flags |= (TVG_FLAGS_NONZERO  if nonzero  else 0)
        flags |= (TVG_FLAGS_POSITIVE if positive else 0)
        flags |= (TVG_FLAGS_DIRECTED if directed else 0)

        obj = lib.mongodb_load_graph(None, mongodb._obj, objectid, flags)
        return Graph(obj=obj) if obj else None

    def unlink(self):
        """ Unlink a graph from the TVG object. """
        lib.unlink_graph(self._obj)

    @property
    @cacheable
    def memory_usage(self):
        """ Return the memory usage currently associated with the graph. """
        return lib.graph_memory_usage(self._obj)

    @property
    def next(self):
        """ Return the (chronologically) next graph object. """
        obj = lib.next_graph(self._obj)
        return Graph(obj=obj) if obj else None

    @property
    def prev(self):
        """ Return the (chronologically) previous graph object. """
        obj = lib.prev_graph(self._obj)
        return Graph(obj=obj) if obj else None

    @cacheable
    def empty(self):
        """ Check if the graph is empty, i.e., it does not have any edges. """
        return lib.graph_empty(self._obj)

    def duplicate(self):
        """ Create an independent copy of the graph. """
        return Graph(obj=lib.graph_duplicate(self._obj))

    def clear(self):
        """ Clear all edges of the graph object. """
        res = lib.graph_clear(self._obj)
        if not res:
            raise RuntimeError

    def has_edge(self, indices):
        """ Check if the graph has edge `(source, target)`. """
        (source, target) = indices
        return lib.graph_has_edge(self._obj, source, target)

    def __getitem__(self, indices):
        """ Return the weight of edge `(source, target)`. """
        (source, target) = indices
        return lib.graph_get_edge(self._obj, source, target)

    def edges(self, ret_indices=True, ret_weights=True, as_dict=False):
        """
        Return all indices and/or weights of a graph.

        # Arguments
        ret_indices: Return indices consisting of (source, target), otherwise None.
        ret_weights: Return weights, otherwise None.
        as_dict: Return result as dictionary instead of tuple.

        # Returns
        `(indices, weights)` or dictionary
        """

        if as_dict and not ret_indices:
            raise ValueError("Invalid parameter combination")

        num_edges = 100 # FIXME: Arbitrary limit.
        while True:
            max_edges = num_edges
            indices = np.empty(shape=(max_edges, 2), dtype=np.uint64,  order='C') if ret_indices else None
            weights = np.empty(shape=(max_edges,),   dtype=np.float32, order='C') if ret_weights else None
            num_edges = lib.graph_get_edges(self._obj, indices, weights, max_edges)
            if num_edges <= max_edges:
                break

        if indices is not None:
            indices.resize((num_edges, 2), refcheck=False)
        if weights is not None:
            weights.resize((num_edges,), refcheck=False)

        if as_dict:
            if weights is None:
                weights = [None] * num_edges
            return dict([(tuple(i), w) for i, w in zip(indices, weights)])

        return indices, weights

    def keys(self):
        """ Iterate over indices of a graphs. """
        indices, _ = self.edges(ret_weights=False)
        for i in indices:
            yield tuple(i)

    def values(self):
        """ Iterate over weights of a graph. """
        _, weights = self.edges(ret_indices=False)
        return iter(weights)

    def items(self):
        """ Iterate over indices and weights of a graphs. """
        indices, weights = self.edges()
        for i, w in zip(indices, weights):
            yield (tuple(i), w)

    def top_edges(self, max_edges, ret_indices=True, ret_weights=True, as_dict=False):
        """
        Return indices and/or weights of the top edges.

        # Arguments
        num_edges: Limit the number of edges returned.
        ret_indices: Return indices consisting of (source, target), otherwise None.
        ret_weights: Return weights, otherwise None.
        as_dict: Return result as dictionary instead of tuple.

        # Returns
        `(indices, weights)` or dictionary
        """

        if as_dict and not ret_indices:
            raise ValueError("Invalid parameter combination")

        indices = np.empty(shape=(max_edges, 2), dtype=np.uint64,  order='C') if ret_indices else None
        weights = np.empty(shape=(max_edges,),   dtype=np.float32, order='C') if ret_weights else None
        num_edges = lib.graph_get_top_edges(self._obj, indices, weights, max_edges)

        if indices is not None and num_edges < max_edges:
            indices.resize((num_edges, 2), refcheck=False)
        if weights is not None and num_edges < max_edges:
            weights.resize((num_edges,), refcheck=False)

        if as_dict:
            if weights is None:
                weights = [None] * num_edges
            return collections.OrderedDict([(tuple(i), w) for i, w in zip(indices, weights)])

        return indices, weights

    @property
    @cacheable
    def num_edges(self):
        """ Return the number of edges of a graph. """
        return lib.graph_num_edges(self._obj)

    @cacheable
    def nodes(self):
        """
        Return a list of all nodes. A node is considered present, when it is connected
        to at least one other node (either as a source or target).
        """

        # FIXME: Add a C library helper?
        indices, _ = self.edges(ret_weights=False)
        return np.unique(indices)

    @property
    @cacheable
    def num_nodes(self):
        """ Return the number of nodes of a graph. """
        return len(self.nodes())

    def adjacent_edges(self, source, ret_indices=True, ret_weights=True, as_dict=False):
        """
        Return information about all edges adjacent to a given source edge.

        # Arguments
        source: Index of the source node.
        ret_indices: Return target indices, otherwise None.
        ret_weights: Return weights, otherwise None.
        as_dict: Return result as dictionary instead of tuple.

        # Returns
        `(indices, weights)` or dictionary
        """

        if as_dict and not ret_indices:
            raise ValueError("Invalid parameter combination")

        num_edges = 100 # FIXME: Arbitrary limit.
        while True:
            max_edges = num_edges
            indices = np.empty(shape=(max_edges,), dtype=np.uint64,  order='C') if ret_indices else None
            weights = np.empty(shape=(max_edges,), dtype=np.float32, order='C') if ret_weights else None
            num_edges = lib.graph_get_adjacent_edges(self._obj, source, indices, weights, max_edges)
            if num_edges <= max_edges:
                break

        if indices is not None:
            indices.resize((num_edges,), refcheck=False)
        if weights is not None:
            weights.resize((num_edges,), refcheck=False)

        if as_dict:
            if weights is None:
                weights = [None] * num_edges
            return dict(zip(indices, weights))

        return indices, weights

    def num_adjacent_edges(self, source):
        """ Return the number of adjacent edges to a given `source` node, i.e., the node degree. """
        return lib.graph_get_adjacent_edges(self._obj, source, None, None, 0)

    def __len__(self):
        """ Return the number of edges of a graph. """
        return self.num_edges

    def __setitem__(self, indices, weight):
        """ Set edge `(source, target)` of a graph to `weight`."""
        (source, target) = indices
        res = lib.graph_set_edge(self._obj, source, target, weight)
        if not res:
            raise MemoryError

    @staticmethod
    def _convert_indices_weights(indices, weights=None):
        if weights is not None:
            indices = np.asarray(indices, dtype=np.uint64, order='C')
            weights = np.asarray(weights, dtype=np.float32, order='C')

            if indices.size == 0 and weights.size == 0:
                return None, None
            if len(indices.shape) != 2 or indices.shape[1] != 2:
                raise ValueError("indices array does not have correct dimensions")
            if len(weights.shape) != 1:
                raise ValueError("weights array does not have correct dimensions")
            if indices.shape[0] != weights.shape[0]:
                raise ValueError("indices/weights arrays have different length")

        elif isinstance(indices, dict):
            edges = indices
            if len(edges) == 0:
                return None, None

            indices = np.empty(shape=(len(edges), 2), dtype=np.uint64, order='C')
            weights = np.empty(shape=(len(edges),), dtype=np.float32, order='C')
            for j, (i, w) in enumerate(edges.items()):
                indices[j, :] = i
                weights[j] = w

        else:
            if isinstance(indices, set):
                indices = list(indices)
            indices = np.asarray(indices, dtype=np.uint64, order='C')

            if indices.size == 0:
                return None, None
            if len(indices.shape) != 2 or indices.shape[1] != 2:
                raise ValueError("indices array does not have correct dimensions")

        return indices, weights

    def set_edges(self, indices, weights=None):
        """
        Short-cut to set multiple edges in a graph.
        If weights is None the elements are set to 1.

        # Arguments
        indices: List of indices (list of tuples or 2d numpy array).
        weights: List of weights to set (list or 1d numpy array).
        """

        indices, weights = self._convert_indices_weights(indices, weights)
        if indices is None:
            return

        res = lib.graph_set_edges(self._obj, indices, weights, indices.shape[0])
        if not res:
            raise MemoryError

    def add_edge(self, indices, weight):
        """ Add weight `weight` to edge `(source, target)`. """
        (source, target) = indices
        res = lib.graph_add_edge(self._obj, source, target, weight)
        if not res:
            raise MemoryError

    def add_edges(self, indices, weights=None):
        """
        Short-cut to update multiple edges of a graph by adding values.
        If weights is None the elements are set to 1.

        # Arguments
        indices: List of indices (list of tuples or 2d numpy array).
        weights: List of weights to set (list or 1d numpy array).
        """

        indices, weights = self._convert_indices_weights(indices, weights)
        if indices is None:
            return

        res = lib.graph_add_edges(self._obj, indices, weights, indices.shape[0])
        if not res:
            raise MemoryError

    def add_graph(self, other, weight=1.0):
        """ Add edges specified by a second graph, optionally multiplied by `weight`. """
        res = lib.graph_add_graph(self._obj, other._obj, weight)
        if not res:
            raise MemoryError

    def __add__(self, other):
        result = self.duplicate()
        result.add_graph(other)
        return result

    def sub_edge(self, indices, weight):
        """ Subtract weight `weight` from edge `(source, target)`. """
        (source, target) = indices
        res = lib.graph_sub_edge(self._obj, source, target, weight)
        if not res:
            raise MemoryError

    def sub_edges(self, indices, weights=None):
        """
        Short-cut to update multiple edges of a graph by subtracting values.
        If weights is None the elements are set to 1.

        # Arguments
        indices: List of indices (list of tuples or 2d numpy array).
        weights: List of weights to set (list or 1d numpy array).
        """

        indices, weights = self._convert_indices_weights(indices, weights)
        if indices is None:
            return

        res = lib.graph_sub_edges(self._obj, indices, weights, indices.shape[0])
        if not res:
            raise MemoryError

    def sub_graph(self, other, weight=1.0):
        """ Subtract edges specified by a second graph, optionally multiplied by `weight`. """
        res = lib.graph_sub_graph(self._obj, other._obj, weight)
        if not res:
            raise MemoryError

    def __sub__(self, other):
        result = self.duplicate()
        result.sub_graph(other)
        return result

    def __delitem__(self, indices):
        """ Delete edge `(source, target)` from the graph or do nothing if it doesn't exist. """
        (source, target) = indices
        res = lib.graph_del_edge(self._obj, source, target)
        if not res:
            raise RuntimeError

    def del_edges(self, indices):
        """
        Short-cut to delete multiple edges from a graph.

        # Arguments
        indices: List of indices (list of tuples or 2d numpy array).
        """

        if isinstance(indices, set):
            indices = list(indices)
        indices = np.asarray(indices, dtype=np.uint64, order='C')

        if indices.size == 0:
            return # nothing to do for empty array
        if len(indices.shape) != 2 or indices.shape[1] != 2:
            raise ValueError("indices array does not have correct dimensions")

        res = lib.graph_del_edges(self._obj, indices, indices.shape[0])
        if not res:
            raise RuntimeError

    def mul_const(self, constant):
        """ Perform inplace element-wise multiplication of all graph edges with `constant`. """
        res = lib.graph_mul_const(self._obj, constant)
        if not res:
            raise RuntimeError

    def mul_vector(self, other):
        """ Compute the matrix-vector product of the graph with vector `other`. """
        # FIXME: Check type of 'other'.
        return Vector(obj=lib.graph_mul_vector(self._obj, other._obj))

    def in_degrees(self):
        """ Compute and return a vector of in-degrees. """
        return Vector(obj=lib.graph_in_degrees(self._obj))

    def in_weights(self):
        """ Compute and return a vector of in-weights. """
        return Vector(obj=lib.graph_in_weights(self._obj))

    def out_degrees(self):
        """ Compute and return a vector of out-degrees. """
        return Vector(obj=lib.graph_out_degrees(self._obj))

    def out_weights(self):
        """ Compute and return a vector of out-weights. """
        return Vector(obj=lib.graph_out_weights(self._obj))

    def degree_anomalies(self):
        """ Compute and return a vector of degree anomalies. """
        return Vector(obj=lib.graph_degree_anomalies(self._obj))

    def weight_anomalies(self):
        """ Compute and return a vector of weight anomalies. """
        return Vector(obj=lib.graph_weight_anomalies(self._obj))

    @cacheable
    def sum_weights(self):
        """ Compute the sum of all weights. """
        return lib.graph_sum_weights(self._obj)

    def power_iteration(self, initial_guess=None, num_iterations=0, tolerance=None, ret_eigenvalue=True):
        """
        Compute and return the eigenvector (and optionally the eigenvalue).

        # Arguments
        initial_guess: Initial guess for the solver.
        num_iterations: Number of iterations.
        tolerance: Desired tolerance.
        ret_eigenvalue: Also return the eigenvalue. This requires one more iteration.

        # Returns
        `(eigenvector, eigenvalue)`
        """

        if tolerance is None:
            tolerance = 0.0

        eigenvalue = c_double() if ret_eigenvalue else None
        initial_guess_obj = initial_guess._obj if initial_guess is not None else None
        vector = Vector(obj=lib.graph_power_iteration(self._obj, initial_guess_obj, num_iterations, tolerance, eigenvalue))
        if eigenvalue is not None:
            eigenvalue = eigenvalue.value
        return vector, eigenvalue

    def filter_nodes(self, nodes):
        """
        Create a subgraph by only keeping edges, where at least one node is
        part of the subset specified by the `nodes` parameter.

        # Arguments
        nodes: Vector, list or set of nodes to preserve

        # Returns
        Resulting graph.
        """

        if not isinstance(nodes, Vector):
            vector = Vector()
            vector.set_entries(nodes)
            nodes = vector

        return Graph(obj=lib.graph_filter_nodes(self._obj, nodes._obj))

    def normalize(self):
        """
        Normalize a graph based on the in and out-degrees of neighbors.

        # Returns
        Resulting graph.
        """

        return Graph(obj=lib.graph_normalize(self._obj))

    def sparse_subgraph(self, seeds=None, num_seeds=8, num_neighbors=3):
        """
        Create a sparse subgraph by seleting a few seed edges, and then
        using 'triangular growth' to add additional neighbors.

        # Arguments
        seeds: List of seed edges
        num_seeds: Number of seed edges to select
        num_neighbors: Number of neighbors to add per seed node

        # Returns
        Resulting graph.
        """

        if self.directed:
            raise NotImplementedError("Not implemented for directed graphs")

        if seeds is None:
            seeds = self.top_edges(num_seeds, as_dict=True)
        if not isinstance(seeds, dict):
            seeds = dict([(tuple(i), self[i]) for i in seeds])

        edges = copy.deepcopy(seeds)
        for i, j in seeds.keys():
            edges_i = self.adjacent_edges(i, as_dict=True)
            edges_j = self.adjacent_edges(j, as_dict=True)
            del edges_i[j]
            del edges_j[i]

            neighbors = list(set(edges_i.keys()) & set(edges_j.keys()))
            neighbors = sorted(neighbors, key=lambda k: min(edges_i[k], edges_j[k]), reverse=True)
            for k in neighbors[:num_neighbors]:
                edges[i, k] = edges_i[k]
                edges[j, k] = edges_j[k]

        return Graph.from_dict(edges)

    def bfs_count(self, source, max_count=None):
        """
        Perform a breadth-first search in the graph, starting from node `source`.
        In this version, the order is based solely on the number of links.

        # Arguments
        source: Index of the source node.
        max_count: Maximum depth.

        # Returns
        List of tuples `(weight, count, edge_from, edge_to)`.
        """

        result = []

        if max_count is None:
            max_count = 0xffffffffffffffff

        def wrapper(graph, entry, userdata):
            if entry.contents.count > max_count:
                return 1

            entry = entry.contents
            edge_from = entry.edge_from if entry.edge_from != 0xffffffffffffffff else None
            result.append((entry.weight, entry.count, edge_from, entry.edge_to))
            return 0

        res = lib.graph_bfs(self._obj, source, 0, c_bfs_callback_p(wrapper), None)
        if not res:
            raise RuntimeError

        return result

    def bfs_weight(self, source, max_weight=np.inf):
        """
        Perform a breadth-first search in the graph, starting from node `source`.
        In this version, the order is based on the sum of the weights.

        # Arguments
        source: Index of the source node.
        max_weight: Maximum weight.

        # Returns
        List of tuples `(weight, count, edge_from, edge_to)`.
        """

        result = []

        def wrapper(graph, entry, userdata):
            if entry.contents.weight > max_weight:
                return 1

            entry = entry.contents
            edge_from = entry.edge_from if entry.edge_from != 0xffffffffffffffff else None
            result.append((entry.weight, entry.count, edge_from, entry.edge_to))
            return 0

        res = lib.graph_bfs(self._obj, source, 1, c_bfs_callback_p(wrapper), None)
        if not res:
            raise RuntimeError

        return result

    def distance_count(self, source, end):
        count = lib.graph_get_distance_count(self._obj, source, end)
        if count == 0xffffffffffffffff:
            count = np.inf
        return count

    def distance_weight(self, source, end):
        return lib.graph_get_distance_weight(self._obj, source, end)

    def as_dict(self):
        """ Return a dictionary containing all graph edges. """
        return self.edges(as_dict=True)

    @staticmethod
    def from_dict(edges, *args, **kwargs):
        """ Generate a Graph object from a dictionary. """
        graph = Graph(*args, **kwargs)
        graph.set_edges(edges)
        return graph

class GraphIter(object):
    def __init__(self, graph):
        self._graph = graph

    def __next__(self):
        if not self._graph:
            raise StopIteration

        result = self._graph
        self._graph = result.next
        return result

class GraphIterReversed(object):
    def __init__(self, graph):
        self._graph = graph

    def __iter__(self):
        return self

    def __next__(self):
        if not self._graph:
            raise StopIteration

        result = self._graph
        self._graph = result.prev
        return result

@libtvgobject
class Node(object):
    """
    This object represents a node. Since nodes are implicit in our model, they
    should only have static attributes that do not depend on the timestamp.
    For now, both node attribute keys and values are limited to the string type -
    in the future this might be extended to other data types. Attributes related
    to the primary key (that uniquely identify a node in the context of a time-
    varying-graph) must be set before both objects are linked. All other attributes
    can be set at any time.

    # Arguments
    **kwargs: Key-value pairs of type string to assign to the node.
    """

    def __init__(self, obj=None, **kwargs):
        if obj is None:
            obj = lib.alloc_node()

        self._obj = obj
        if not obj:
            raise MemoryError

        for k, v in kwargs.items():
            self[k] = v

    def __del__(self):
        if lib is None:
            return
        if self._obj:
            lib.free_node(self._obj)
            self._obj = None

    def _get_obj(self):
        assert self._obj.contents.refcount >= 2
        lib.free_node(self._obj)
        return self

    def __repr__(self):
        return "Node(%s)" % self.as_dict().__repr__()

    @property
    def index(self):
        """ Return the index of the node. """
        return self._obj.contents.index

    @property
    def text(self):
        """ Short-cut to return the 'text' attribute of a node. """
        return self["text"]

    def unlink(self):
        """
        Unlink the node from the time-varying graph. The node itself stays valid,
        but it is no longer returned for any `node_by_index` or `node_by_primary_key`
        call.
        """

        lib.unlink_node(self._obj)

    def __setitem__(self, key, value):
        """ Set the node attribute `key` to `value`. Both key and value must have the type string. """
        res = lib.node_set_attribute(self._obj, key.encode("utf-8"), value.encode("utf-8"))
        if not res:
            raise KeyError

    def __getitem__(self, key):
        """ Return the node attribute for `key`. """
        value = lib.node_get_attribute(self._obj, key.encode("utf-8"))
        if not value:
            raise KeyError
        return value.decode("utf-8")

    def as_dict(self):
        """ Return a dictionary containing all node attributes. """
        ptr = lib.node_get_attributes(self._obj)
        if not ptr:
            raise MemoryError

        data = {}
        for i in itertools.count(step=2):
            if not ptr[i]: break
            key = ptr[i].decode("utf-8")
            value = ptr[i + 1].decode("utf-8")
            data[key] = value

        libc.free(ptr)
        return data

@libtvgobject
class TVG(object):
    """
    This object represents a time-varying graph.

    # Arguments
    nonzero: Enforce that all entries must be non-zero.
    positive: Enforce that all entries must be positive.
    directed: Create a directed time-varying graph.
    streaming: Support for streaming / differential updates.
    primary_key: List or semicolon separated string of attributes.
    """

    def __init__(self, nonzero=False, positive=False, directed=False, streaming=False, primary_key=None, obj=None):
        if obj is None:
            flags = 0
            flags |= (TVG_FLAGS_NONZERO  if nonzero  else 0)
            flags |= (TVG_FLAGS_POSITIVE if positive else 0)
            flags |= (TVG_FLAGS_DIRECTED if directed else 0)
            flags |= (TVG_FLAGS_STREAMING if streaming else 0)
            obj = lib.alloc_tvg(flags)

        self._obj = obj
        if not obj:
            raise MemoryError

        if primary_key:
            self.set_primary_key(primary_key)

    def __del__(self):
        if lib is None:
            return
        if self._obj:
            lib.free_tvg(self._obj)
            self._obj = None

    def _get_obj(self):
        assert self._obj.contents.refcount >= 2
        lib.free_tvg(self._obj)
        return self

    @property
    def flags(self):
        return self._obj.contents.flags

    @property
    def verbosity(self):
        return self._obj.contents.verbosity

    @verbosity.setter
    def verbosity(self, verbosity):
        lib.tvg_set_verbosity(self._obj, verbosity)

    @property
    def memory_usage(self):
        """ Return the memory usage currently associated with the TVG. """
        return lib.tvg_memory_usage(self._obj)

    def link_graph(self, graph, ts):
        """
        Link a graph to the time-varying-graph object.

        # Arguments
        graph: The graph to link.
        ts: Time-stamp of the graph (as uint64, typically UNIX timestamp in milliseconds).
        """
        res = lib.tvg_link_graph(self._obj, graph._obj, ts)
        if not res:
            raise RuntimeError

    def set_primary_key(self, key):
        """
        Set or update the primary key used to distinguish graph nodes. The key can
        consist of one or multiple attributes, and is used to identify a node
        (especially, when loading from an external source, that does not use integer
        identifiers).

        # Arguments
        key: List or semicolon separated string of attributes.
        """

        if isinstance(key, list):
            key = ";".join(key)
        res = lib.tvg_set_primary_key(self._obj, key.encode("utf-8"))
        if not res:
            raise RuntimeError

    def link_node(self, node, index=None):
        """
        Link a node to the time-varying-graph object.

        # Arguments
        node: The node to link.
        index: Index to assign to the node, or `None` if the next empty index should be used.
        """

        if index is None:
            index = 0xffffffffffffffff

        res = lib.tvg_link_node(self._obj, node._obj, None, index)
        if not res:
            raise RuntimeError

    def Node(self, **kwargs):
        """
        Create a new node assicated with the graph. Note that all primary key attributes
        must be set immediately during construction, it is not possible to change them later.

        # Arguments
        **kwargs: Key-value pairs of type string to assign to the node.
        """

        node = Node(**kwargs)

        obj = c_node_p()
        res = lib.tvg_link_node(self._obj, node._obj, obj, 0xffffffffffffffff)
        if not res:
            del node
            if not obj:
                raise RuntimeError
            node = Node(obj=obj)

        return node

    def node_by_index(self, index):
        """
        Lookup a node by index.

        # Arguments
        index: Index of the node.

        # Returns
        Node object.
        """

        obj = lib.tvg_get_node_by_index(self._obj, index)
        if not obj:
            raise KeyError
        return Node(obj=obj)

    def node_by_primary_key(self, **kwargs):
        """
        Lookup a node by its primary key. This must match the primary key set with
        `set_primary_key` (currently, a time-varying graph can only have one key).

        # Arguments
        **kwargs: Key-value pairs of the primary key.

        # Returns
        Node object.
        """

        primary_key = Node(**kwargs)
        obj = lib.tvg_get_node_by_primary_key(self._obj, primary_key._obj)
        del primary_key
        if not obj:
            raise KeyError
        return Node(obj=obj)

    def node_by_text(self, text):
        """ Lookup a node by its text (assumes that `text` is the primary key). """
        return self.node_by_primary_key(text=text)

    @staticmethod
    def load(source, nodes=None, *args, **kwargs):
        """
        Load a time-varying-graph from an external data source.

        # Arguments
        source: Data source to load (currently either a file path, or a MongoDB object).
        nodes: Secondary data source to load node attributes (must be a file path).
        *args, **kwargs: Arguments passed through to the `TVG()` constructor.
        """

        tvg = TVG(*args, **kwargs)
        if isinstance(source, MongoDB):
            tvg.load_graphs_from_mongodb(source)
        else:
            tvg.load_graphs_from_file(source)
        if nodes:
            tvg.load_nodes_from_file(nodes)
        return tvg

    def load_graphs_from_file(self, filename):
        """ Load a time-varying-graph (i.e., a collection of graphs) from a file. """
        res = lib.tvg_load_graphs_from_file(self._obj, filename.encode("utf-8"))
        if not res:
            raise IOError

    def load_nodes_from_file(self, filename, key=None):
        """ Load node attributes from a file. """

        if key is None:
            key = "text"
        if isinstance(key, list):
            key = ";".join(key)
        res = lib.tvg_load_nodes_from_file(self._obj, filename.encode("utf-8"), key.encode("utf-8"))
        if not res:
            raise IOError

    def load_graphs_from_mongodb(self, mongodb):
        """ Load a time-varying-graph (i.e., multiple graphs) from a MongoDB. """
        res = lib.tvg_load_graphs_from_mongodb(self._obj, mongodb._obj)
        if not res:
            raise IOError

    def enable_mongodb_sync(self, mongodb, batch_size=0, cache_size=0):
        """
        Enable synchronization with a MongoDB server. Whenever more data is needed
        (e.g., querying the previous or next graph, or looking up graphs in a certain
        range), requests are sent to the database. Each request loads up to
        `batch_size` graphs. The maximum amount of data kept in memory can be
        controlled with the `cache_size` parameter.

        # Arguments
        mongodb: MongoDB object.
        batch_size: Maximum number of graphs to load in a single request.
        cache_size: Maximum size of the cache (in bytes).
        """

        res = lib.tvg_enable_mongodb_sync(self._obj, mongodb._obj, batch_size, cache_size)
        if not res:
            raise IOError

    def disable_mongodb_sync(self):
        """ Disable synchronization with a MongoDB server. """
        lib.tvg_disable_mongodb_sync(self._obj)

    def enable_query_cache(self, cache_size=0):
        """
        Enable the query cache. This can be used to speed up query performance, at the
        cost of higher memory usage. The maximum amount of data kept in memory can be
        controlled with the `cache_size` parameter.

        # Arguments
        cache_size: Maximum size of the cache (in bytes).
        """

        res = lib.tvg_enable_query_cache(self._obj, cache_size)
        if not res:
            raise RuntimeError

    def disable_query_cache(self):
        """ Disable the query cache. """
        lib.tvg_disable_query_cache(self._obj)

    def invalidate_queries(self, ts_min, ts_max):
        """ Invalidate queries in a given timeframe [ts_min, ts_max]. """
        lib.tvg_invalidate_queries(self._obj, ts_min, ts_max)

    def __iter__(self):
        """ Iterates through all graphs of a time-varying-graph object. """
        return GraphIter(self.lookup_ge())

    def __reversed__(self):
        """ Iterates (in reverse order) through all graphs of a time-varying graph object. """
        return GraphIterReversed(self.lookup_le())

    def sum_edges(self, ts_min, ts_max, eps=None):
        """
        Add edges in a given timeframe [ts_min, ts_max].

        # Arguments
        ts_min: Left boundary of the interval.
        ts_max: Right boundary of the interval.
        """

        if ts_min < 0:
            ts_min = 0
        if ts_max > 0xffffffffffffffff:
            ts_max = 0xffffffffffffffff
        if eps is None:
            eps = 0.0

        return Graph(obj=lib.tvg_sum_edges(self._obj, ts_min, ts_max, eps))

    def sum_edges_exp(self, ts_min, ts_max, beta=None, log_beta=None, weight=1.0, eps=None):
        """
        Add edges in a given timeframe [ts_min, ts_max], weighted by an exponential
        decay function.

        # Arguments
        ts_min: Left boundary of the interval.
        ts_max: Right boundary of the interval.
        beta: Exponential decay constant.
        """

        if ts_min < 0:
            ts_min = 0
        if ts_max > 0xffffffffffffffff:
            ts_max = 0xffffffffffffffff
        if log_beta is None:
            log_beta = math.log(beta)
        if eps is None:
            eps = 0.0

        return Graph(obj=lib.tvg_sum_edges_exp(self._obj, ts_min, ts_max, weight, log_beta, eps))

    def sum_edges_exp_norm(self, ts_min, ts_max, beta=None, log_beta=None, eps=None):
        """
        Add edges in a given timeframe [ts_min, ts_max], weighted by an exponential
        smoothing function.

        # Arguments
        ts_min: Left boundary of the interval.
        ts_max: Right boundary of the interval.
        beta: Exponential decay constant.
        """

        if log_beta is None:
            log_beta = math.log(beta)

        return self.sum_edges_exp(ts_min, ts_max, log_beta=log_beta, weight=-np.expm1(log_beta), eps=eps)

    def count_edges(self, ts_min, ts_max):
        """
        Count edges in a given timeframe [ts_min, ts_max].

        # Arguments
        ts_min: Left boundary of the interval.
        ts_max: Right boundary of the interval.
        """

        if ts_min < 0:
            ts_min = 0
        if ts_max > 0xffffffffffffffff:
            ts_max = 0xffffffffffffffff

        return Graph(obj=lib.tvg_count_edges(self._obj, ts_min, ts_max))

    def count_nodes(self, ts_min, ts_max):
        """
        Count nodes in a given timeframe [ts_min, ts_max].

        # Arguments
        ts_min: Left boundary of the interval.
        ts_max: Right boundary of the interval.
        """

        if ts_min < 0:
            ts_min = 0
        if ts_max > 0xffffffffffffffff:
            ts_max = 0xffffffffffffffff

        return Vector(obj=lib.tvg_count_nodes(self._obj, ts_min, ts_max))

    def count_graphs(self, ts_min, ts_max):
        """
        Count graphs in a given timeframe [ts_min, ts_max].

        # Arguments
        ts_min: Left boundary of the interval.
        ts_max: Right boundary of the interval.
        """

        if ts_min < 0:
            ts_min = 0
        if ts_max > 0xffffffffffffffff:
            ts_max = 0xffffffffffffffff

        res = lib.tvg_count_graphs(self._obj, ts_min, ts_max)
        if res == 0xffffffffffffffff:
            raise MemoryError

        return res

    def topics(self, ts_min, ts_max, step=0, offset=0):
        """
        Extract network topics in the timeframe [ts_min, ts_max].

        # Arguments
        ts_min: Left boundary of the interval.
        ts_max: Right boundary of the interval.
        """

        if ts_min < 0:
            ts_min = 0
        if ts_max > 0xffffffffffffffff:
            ts_max = 0xffffffffffffffff

        return Graph(obj=lib.tvg_topics(self._obj, ts_min, ts_max, step, offset))

    def sample_graphs(self, ts_min, ts_max, sample_width, sample_steps=9,
                      method=None, *args, **kwargs):
        """
        Sample graphs in the timeframe [ts_min, ts_max].

        # Arguments
        ts_min: Left boundary of the interval.
        ts_max: Right boundary of the interval.
        sample_width: Width of each sample.
        sample_steps: Number of values to collect.
        method: Method to use (default: 'sum_edges').

        # Yields
        Sampled graphs.
        """

        if not callable(method):
            try:
                method = {
                    None:          self.sum_edges,
                    'sum_edges':   self.sum_edges,
                    'count_edges': self.count_edges,
                    'topics':      self.topics
                }[method]
            except KeyError:
                raise NotImplementedError("Method %s not implemented" % method)

        if sample_width < 1:
            raise RuntimeError("sample_width too small")

        result = []

        for ts in np.linspace(ts_min, ts_max - sample_width + 1, sample_steps):
            graph = method(int(ts), int(ts + sample_width - 1), *args, **kwargs)
            result.append(graph)

        return result

    def sample_eigenvectors(self, ts_min, ts_max, sample_width, sample_steps=9,
                            tolerance=None, method=None, *args, **kwargs):
        """
        Iterative power iteration algorithm to track eigenvectors of a graph over time.
        Eigenvectors are collected within the timeframe [ts_min, ts_max]. Each entry
        of the returned dictionary contains sample_steps values collected at equidistant
        time steps.

        # Arguments
        ts_min: Left boundary of the interval.
        ts_max: Right boundary of the interval.
        sample_width: Width of each sample.
        sample_steps: Number of values to collect.
        tolerance: Tolerance for the power_iteration algorithm.
        method: Method to use (default: 'sum_edges').

        # Returns
        Dictionary containing lists of collected values for each node.
        """

        eigenvector = None
        result = []

        for graph in self.sample_graphs(ts_min, ts_max, sample_width, sample_steps=sample_steps,
                                        method=method, *args, **kwargs):
            eigenvector, _ = graph.power_iteration(initial_guess=eigenvector, tolerance=tolerance,
                                                   ret_eigenvalue=False)
            result.append(eigenvector)

        return result

    def lookup_ge(self, ts=0):
        """ Search for the first graph with timestamps `>= ts`. """
        if isinstance(ts, float):
            ts = math.ceil(ts)
        obj = lib.tvg_lookup_graph_ge(self._obj, ts)
        return Graph(obj=obj) if obj else None

    def lookup_le(self, ts=0xffffffffffffffff):
        """ Search for the last graph with timestamps `<= ts`. """
        if isinstance(ts, float):
            ts = int(ts)
        obj = lib.tvg_lookup_graph_le(self._obj, ts)
        return Graph(obj=obj) if obj else None

    def lookup_near(self, ts):
        """ Search for a graph with a timestamp close to `ts`. """
        if isinstance(ts, float):
            ts = int(ts + 0.5)
        obj = lib.tvg_lookup_graph_near(self._obj, ts)
        return Graph(obj=obj) if obj else None

    def compress(self, step, offset=0):
        """ Compress the graph by aggregating timestamps differing by at most `step`. """
        if step > 0 and np.isinf(step):
            step = 0
        res = lib.tvg_compress(self._obj, step, offset)
        if not res:
            raise MemoryError

@libtvgobject
class MongoDB(object):
    """
    This object represents a MongoDB connection.

    # Arguments
    uri: URI to identify the MongoDB server, e.g., mongodb://localhost.
    database: Name of the database.

    col_articles: Name of the articles collection.
    article_id: Name of the article ID key.
    article_time: Name of the article time key.

    filter_key: Filter articles by comparing the value of a given key.
    filter_value: Expected value of filter key.

    col_entities: Name of the entities collection.
    entity_doc: Name of the entity doc key.
    entity_sen: Name of the entity sen key.
    entity_ent: Name(s) of the entity ent key, e.g., attr1;attr2;attr3.

    use_pool: Use a connection pool to access MongoDB.
    load_nodes: Load node attributes.
    sum_weights: Compute edge weights as the sum of co-occurrence weights.
    norm_weights: Normalize weights, such that each graph has a weight of 1.

    max_distance: Maximum distance of mentions.
    """

    def __init__(self, uri, database, col_articles, article_id, article_time,
                 col_entities, entity_doc, entity_sen, entity_ent, use_pool=True,
                 load_nodes=False, sum_weights=True, norm_weights=False, max_distance=None,
                 filter_key=None, filter_value=None, use_objectids=None, obj=None):
        if obj is None:
            config = c_mongodb_config()
            config.uri           = uri.encode("utf-8")
            config.database      = database.encode("utf-8")
            config.col_articles  = col_articles.encode("utf-8")
            config.article_id    = article_id.encode("utf-8")
            config.article_time  = article_time.encode("utf-8")
            config.filter_key    = filter_key.encode("utf-8")   if filter_key   is not None else None
            config.filter_value  = filter_value.encode("utf-8") if filter_value is not None else None
            config.col_entities  = col_entities.encode("utf-8")
            config.entity_doc    = entity_doc.encode("utf-8")
            config.entity_sen    = entity_sen.encode("utf-8")
            config.entity_ent    = entity_ent.encode("utf-8")
            config.use_pool      = use_pool
            config.load_nodes    = load_nodes
            config.sum_weights   = sum_weights
            config.norm_weights  = norm_weights
            config.max_distance  = max_distance if max_distance is not None else 0xffffffffffffffff
            obj = lib.alloc_mongodb(config)

        self._obj = obj
        if not obj:
            raise MemoryError

    def __del__(self):
        if lib is None:
            return
        if self._obj:
            lib.free_mongodb(self._obj)
            self._obj = None

    def _get_obj(self):
        assert self._obj.contents.refcount >= 2
        lib.free_mongodb(self._obj)
        return self

if __name__ == '__main__':
    import datetime
    import unittest
    import mockupdb
    import tempfile
    import bson
    import gc

    class VectorTests(unittest.TestCase):
        def test_add_entry(self):
            v = Vector()
            self.assertTrue(v.empty())
            self.assertTrue(v.empty(drop_cache=True))
            revisions = [v.revision]
            mem = v.memory_usage

            for i in range(10):
                v[i] = i * i

            self.assertFalse(v.empty())
            self.assertNotIn(v.revision, revisions)
            self.assertGreater(v.memory_usage, mem)
            revisions.append(v.revision)

            self.assertEqual(v.sum_weights(), 285.0)
            self.assertEqual(v.sum_weights(drop_cache=True), 285.0)
            self.assertEqual(v.norm(), math.sqrt(15333.0))
            self.assertEqual(v.norm(drop_cache=True), math.sqrt(15333.0))
            self.assertEqual(v.mul_vector(v), 15333.0)

            for i in range(10):
                self.assertTrue(v.has_entry(i))
                self.assertEqual(v[i], i * i)
                v.add_entry(i, 1.0)
                self.assertEqual(v[i], i * i + 1)
                v.sub_entry(i, 1.0)
                self.assertEqual(v[i], i * i)

            self.assertNotIn(v.revision, revisions)
            revisions.append(v.revision)

            v.mul_const(2.0)

            self.assertNotIn(v.revision, revisions)
            revisions.append(v.revision)

            for i in range(10):
                self.assertTrue(v.has_entry(i))
                self.assertEqual(v[i], 2.0 * i * i)
                del v[i]
                self.assertFalse(v.has_entry(i))
                self.assertEqual(v[i], 0.0)

            self.assertTrue(v.empty())
            self.assertNotIn(v.revision, revisions)
            del v

        def test_add_vector(self):
            v1 = Vector()
            v1[0] = 1.0
            v1[1] = 2.0
            v1[2] = 3.0

            v2 = Vector()
            v2[1] = 30.0
            v2[2] = 20.0
            v2[3] = 10.0

            v = v1 + v2
            self.assertEqual(v.as_dict(), {0: 1.0, 1: 32.0, 2: 23.0, 3: 10.0})
            del v

            v = v2 - v1
            self.assertEqual(v.as_dict(), {0: -1.0, 1: 28.0, 2: 17.0, 3: 10.0})
            del v

            del v1
            del v2

        def test_clear(self):
            v = Vector()
            self.assertTrue(v.empty())
            for i in range(10):
                v[i] = i * i
            self.assertFalse(v.empty())
            v.clear()
            self.assertTrue(v.empty())
            for i in range(10):
                v[i] = i * i
            self.assertFalse(v.empty())
            del v

        def test_batch(self):
            test_indices = np.array([0, 1, 2])
            test_weights = np.array([1.0, 2.0, 3.0])

            v = Vector()
            self.assertEqual(v.flags, 0)

            v.add_entries([], [])
            v.add_entries(test_indices, test_weights)
            indices, weights = v.entries()
            self.assertEqual(indices.tolist(), [0, 1, 2])
            self.assertEqual(weights.tolist(), [1.0, 2.0, 3.0])
            self.assertEqual(v.num_entries, 3)
            self.assertEqual(len(v), 3)

            self.assertEqual(list(v.keys()), [0, 1, 2])
            self.assertEqual(list(v.values()), [1.0, 2.0, 3.0])
            self.assertEqual(list(v.items()), [(0, 1.0), (1, 2.0), (2, 3.0)])

            indices, _ = v.entries(ret_weights=False)
            self.assertEqual(indices.tolist(), [0, 1, 2])

            _, weights = v.entries(ret_indices=False)
            self.assertEqual(weights.tolist(), [1.0, 2.0, 3.0])

            v.add_entries([], [])
            v.add_entries(test_indices, test_weights)
            indices, weights = v.entries()
            self.assertEqual(indices.tolist(), [0, 1, 2])
            self.assertEqual(weights.tolist(), [2.0, 4.0, 6.0])

            v.add_entries([])
            v.add_entries(test_indices)
            indices, weights = v.entries()
            self.assertEqual(indices.tolist(), [0, 1, 2])
            self.assertEqual(weights.tolist(), [3.0, 5.0, 7.0])

            v.sub_entries([], [])
            v.sub_entries(test_indices, -test_weights)
            indices, weights = v.entries()
            self.assertEqual(indices.tolist(), [0, 1, 2])
            self.assertEqual(weights.tolist(), [4.0, 7.0, 10.0])

            v.sub_entries([])
            v.sub_entries(test_indices)
            indices, weights = v.entries()
            self.assertEqual(indices.tolist(), [0, 1, 2])
            self.assertEqual(weights.tolist(), [3.0, 6.0, 9.0])

            v.set_entries([], [])
            v.set_entries(test_indices, test_weights)
            indices, weights = v.entries()
            self.assertEqual(indices.tolist(), [0, 1, 2])
            self.assertEqual(weights.tolist(), [1.0, 2.0, 3.0])

            v.set_entries([])
            v.set_entries(test_indices)
            indices, weights = v.entries()
            self.assertEqual(indices.tolist(), [0, 1, 2])
            self.assertEqual(weights.tolist(), [1.0, 1.0, 1.0])

            v.del_entries([])
            v.del_entries(test_indices)
            self.assertEqual(v.entries()[0].tolist(), [])
            self.assertEqual(v.num_entries, 0)
            self.assertEqual(len(v), 0)

            v.set_entries(set())
            v.set_entries(set(test_indices.tolist()))
            indices, weights = v.entries()
            self.assertEqual(indices.tolist(), [0, 1, 2])
            self.assertEqual(weights.tolist(), [1.0, 1.0, 1.0])

            v.del_entries(set())
            v.del_entries(set(test_indices.tolist()))
            self.assertEqual(v.entries()[0].tolist(), [])
            self.assertEqual(v.num_entries, 0)
            self.assertEqual(len(v), 0)

            for i in range(1000):
                v.add_entry(i, 1.0)
            indices, _ = v.entries(ret_weights=False)
            indices = list(indices)
            for i in range(1000):
                self.assertIn(i, indices)
                indices.remove(i)
            self.assertEqual(len(indices), 0)
            self.assertEqual(v.num_entries, 1000)
            self.assertEqual(len(v), 1000)

            del v

        def test_flags(self):
            v = Vector()
            self.assertEqual(v.flags, 0)
            v[0] = 0.0
            self.assertTrue(v.has_entry(0))
            v.add_entry(0, 1.0)
            self.assertEqual(v[0], 1.0)
            v.add_entry(0, -1.0)
            self.assertTrue(v.has_entry(0))
            self.assertEqual(v[0], 0.0)
            v.sub_entry(0, 1.0)
            self.assertEqual(v[0], -1.0)
            v.sub_entry(0, -1.0)
            self.assertTrue(v.has_entry(0))
            self.assertEqual(v[0], 0.0)
            del v

            v = Vector(nonzero=True)
            self.assertEqual(v.flags, TVG_FLAGS_NONZERO)
            self.assertEqual(v.eps, 0.0)
            v[0] = 0.0
            self.assertFalse(v.has_entry(0))
            v.add_entry(0, 1.0)
            self.assertEqual(v[0], 1.0)
            v.add_entry(0, -0.75)
            self.assertEqual(v[0], 0.25)
            v.add_entry(0, -0.25)
            self.assertFalse(v.has_entry(0))
            v.sub_entry(0, 1.0)
            self.assertEqual(v[0], -1.0)
            v.sub_entry(0, -0.75)
            self.assertEqual(v[0], -0.25)
            v.sub_entry(0, -0.25)
            self.assertFalse(v.has_entry(0))
            del v

            v = Vector(nonzero=True)
            v.eps = 0.5
            self.assertEqual(v.flags, TVG_FLAGS_NONZERO)
            self.assertEqual(v.eps, 0.5)
            v[0] = 0.0
            self.assertFalse(v.has_entry(0))
            v.add_entry(0, 1.0)
            self.assertEqual(v[0], 1.0)
            v.add_entry(0, -0.25)
            self.assertEqual(v[0], 0.75)
            v.add_entry(0, -0.25)
            self.assertFalse(v.has_entry(0))
            v.sub_entry(0, 1.0)
            self.assertEqual(v[0], -1.0)
            v.sub_entry(0, -0.25)
            self.assertEqual(v[0], -0.75)
            v.sub_entry(0, -0.25)
            self.assertFalse(v.has_entry(0))
            del v

            v = Vector(positive=True)
            self.assertEqual(v.flags, TVG_FLAGS_NONZERO | TVG_FLAGS_POSITIVE)
            self.assertEqual(v.eps, 0.0)
            v[0] = 0.0
            self.assertFalse(v.has_entry(0))
            v.add_entry(0, 1.0)
            self.assertEqual(v[0], 1.0)
            v.add_entry(0, -0.75)
            self.assertEqual(v[0], 0.25)
            v.add_entry(0, -0.25)
            self.assertFalse(v.has_entry(0))
            v.sub_entry(0, 1.0)
            self.assertFalse(v.has_entry(0))
            v.sub_entry(0, -0.25)
            self.assertEqual(v[0], 0.25)
            del v

            v = Vector(positive=True)
            v.eps = 0.5
            self.assertEqual(v.flags, TVG_FLAGS_NONZERO | TVG_FLAGS_POSITIVE)
            self.assertEqual(v.eps, 0.5)
            v[0] = 0.0
            self.assertFalse(v.has_entry(0))
            v.add_entry(0, 1.0)
            self.assertEqual(v[0], 1.0)
            v.add_entry(0, -0.25)
            self.assertEqual(v[0], 0.75)
            v.add_entry(0, -0.25)
            self.assertFalse(v.has_entry(0))
            v.sub_entry(0, 1.0)
            self.assertFalse(v.has_entry(0))
            v.sub_entry(0, -0.25)
            self.assertFalse(v.has_entry(0))
            v.sub_entry(0, -0.5)
            self.assertFalse(v.has_entry(0))
            v.sub_entry(0, -0.75)
            self.assertEqual(v[0], 0.75)
            del v

        def test_mul_const(self):
            v = Vector()
            v[0] = 1.0
            v.mul_const(-1.0)
            self.assertEqual(v[0], -1.0)
            v.mul_const(0.0)
            self.assertTrue(v.has_entry(0))
            self.assertEqual(v[0], 0.0)
            del v

            v = Vector(nonzero=True)
            v[0] = 1.0
            v.mul_const(-1.0)
            self.assertEqual(v[0], -1.0)
            v.mul_const(0.0)
            self.assertFalse(v.has_entry(0))
            del v

            v = Vector(positive=True)
            v[0] = 1.0
            v.mul_const(-1.0)
            self.assertFalse(v.has_entry(0))
            del v

            v = Vector(positive=True)
            v[0] = 1.0
            for i in range(200):
                v.mul_const(0.5)
                if not v.has_entry(0):
                    break
            else:
                self.assertTrue(False)

        def test_sub_vector_norm(self):
            v = Vector()
            v[0] = 2.0

            w = Vector()
            w[0] = 5.0
            w[1] = 4.0

            self.assertEqual(v.sub_vector_norm(v), 0.0)
            self.assertEqual(w.sub_vector_norm(w), 0.0)
            self.assertEqual(v.sub_vector_norm(w), 5.0)
            self.assertEqual(w.sub_vector_norm(v), 5.0)

            del v
            del w

        def test_repr(self):
            v = Vector()
            self.assertEqual(repr(v), "Vector({})")
            for i in range(10):
                v[i] = 1.0
            expected = "Vector({0: X, 1: X, 2: X, 3: X, 4: X, 5: X, 6: X, 7: X, 8: X, 9: X})"
            self.assertEqual(repr(v).replace("1.000000", "X"), expected)
            v[10] = 2.0
            expected = "Vector({0: X, 1: X, 2: X, 3: X, 4: X, 5: X, 6: X, 7: X, 8: X, 9: X, ...})"
            self.assertEqual(repr(v).replace("1.000000", "X"), expected)
            del v

        def test_as_dict(self):
            v = Vector()
            for i in range(10):
                v[i] = i * i

            with self.assertRaises(ValueError):
                v.entries(ret_indices=False, as_dict=True)

            result = v.entries(ret_weights=False, as_dict=True)
            self.assertEqual(result, {0: None, 1: None, 2: None, 3: None, 4: None,
                                      5: None, 6: None, 7: None, 8: None, 9: None})

            result = v.as_dict()
            self.assertEqual(result, {0: 0.0, 1: 1.0, 2: 4.0, 3: 9.0, 4: 16.0,
                                      5: 25.0, 6: 36.0, 7: 49.0, 8: 64.0, 9: 81.0})

            del v
            v = Vector.from_dict(result)

            result = v.as_dict()
            self.assertEqual(result, {0: 0.0, 1: 1.0, 2: 4.0, 3: 9.0, 4: 16.0,
                                      5: 25.0, 6: 36.0, 7: 49.0, 8: 64.0, 9: 81.0})

            del v

        def test_duplicate(self):
            v = Vector()

            for i in range(10):
                v[i] = i * i

            revision = v.revision
            v2 = v.duplicate()

            for i in range(10):
                v[i] = 1.0

            self.assertNotEqual(v.revision, revision)
            self.assertEqual(v2.revision, revision)

            for i in range(10):
                self.assertEqual(v2[i], i * i)

            del v
            del v2

    class GraphTests(unittest.TestCase):
        def test_add_edge(self):
            g = Graph(directed=True)
            self.assertTrue(g.empty())
            self.assertTrue(g.empty(drop_cache=True))
            self.assertEqual(g.sum_weights(), 0.0)
            revisions = [g.revision]
            mem = g.memory_usage

            for i in range(100):
                s, t = i//10, i%10
                g[s, t] = i

            self.assertFalse(g.empty())
            self.assertNotIn(g.revision, revisions)
            self.assertGreater(g.memory_usage, mem)
            self.assertEqual(g.sum_weights(), 4950.0)
            revisions.append(g.revision)

            for i in range(100):
                s, t = i//10, i%10
                self.assertTrue(i == 0 or g.has_edge((s, t)))
                self.assertEqual(g[s, t], i)
                g.add_edge((s, t), 1.0)
                self.assertEqual(g[s, t], i + 1)
                g.sub_edge((s, t), 1.0)
                self.assertEqual(g[s, t], i)

            self.assertNotIn(g.revision, revisions)
            self.assertEqual(g.sum_weights(), 4950.0)
            revisions.append(g.revision)

            g.mul_const(2.0)

            self.assertNotIn(g.revision, revisions)
            self.assertEqual(g.sum_weights(), 9900.0)
            revisions.append(g.revision)

            for i in range(100):
                s, t = i//10, i%10
                self.assertTrue(i == 0 or g.has_edge((s, t)))
                self.assertEqual(g[s, t], 2.0 * i)
                del g[s, t]
                self.assertFalse(g.has_edge((s, t)))
                self.assertEqual(g[s, t], 0.0)

            self.assertTrue(g.empty())
            self.assertEqual(g.sum_weights(), 0.0)
            self.assertNotIn(g.revision, revisions)
            del g

        def test_add_graph(self):
            g1 = Graph()
            g1[0, 0] = 1.0
            g1[0, 1] = 2.0
            g1[0, 2] = 3.0

            g2 = Graph()
            g2[0, 1] = 30.0
            g2[0, 2] = 20.0
            g2[0, 3] = 10.0

            g = g1 + g2
            self.assertEqual(g.as_dict(), {(0, 0): 1.0, (0, 1): 32.0,
                                           (0, 2): 23.0, (0, 3): 10.0})
            del g

            g = g2 - g1
            self.assertEqual(g.as_dict(), {(0, 0): -1.0, (0, 1): 28.0,
                                           (0, 2): 17.0, (0, 3): 10.0})
            del g

            del g1
            del g2

        def test_clear(self):
            g = Graph(directed=True)
            self.assertTrue(g.empty())
            for i in range(100):
                s, t = i//10, i%10
                g[s, t] = i
            self.assertFalse(g.empty())
            g.clear()
            self.assertTrue(g.empty())
            for i in range(100):
                s, t = i//10, i%10
                g[s, t] = i
            self.assertFalse(g.empty())
            del g

        def test_power_iteration(self):
            g = Graph(directed=True)
            g[0, 0] = 0.5
            g[0, 1] = 0.5
            g[1, 0] = 0.2
            g[1, 1] = 0.8
            v, e = g.power_iteration()
            self.assertTrue(abs(e - 1.0) < 1e-7)
            self.assertTrue(abs(v[0] - 1.0 / math.sqrt(2)) < 1e-7)
            self.assertTrue(abs(v[1] - 1.0 / math.sqrt(2)) < 1e-7)
            del v
            v, _ = g.power_iteration(ret_eigenvalue=False)
            self.assertTrue(abs(v[0] - 1.0 / math.sqrt(2)) < 1e-7)
            self.assertTrue(abs(v[1] - 1.0 / math.sqrt(2)) < 1e-7)
            del v
            v, _ = g.power_iteration(tolerance=1e-3)
            self.assertTrue(abs(v[0] - 1.0 / math.sqrt(2)) < 1e-3)
            self.assertTrue(abs(v[1] - 1.0 / math.sqrt(2)) < 1e-3)
            del v
            w = Vector()
            w[0] = 1.0 / math.sqrt(2)
            w[1] = 1.0 / math.sqrt(2)
            v, _ = g.power_iteration(initial_guess=w, num_iterations=1)
            self.assertTrue(abs(v[0] - 1.0 / math.sqrt(2)) < 1e-7)
            self.assertTrue(abs(v[1] - 1.0 / math.sqrt(2)) < 1e-7)
            del v
            del w

            g.mul_const(-1)

            v, e = g.power_iteration()
            self.assertTrue(abs(e + 1.0) < 1e-7)
            self.assertTrue(abs(v[0] - 1.0 / math.sqrt(2)) < 1e-7)
            self.assertTrue(abs(v[1] - 1.0 / math.sqrt(2)) < 1e-7)
            del v
            v, _ = g.power_iteration(ret_eigenvalue=False)
            self.assertTrue(abs(v[0] - 1.0 / math.sqrt(2)) < 1e-7)
            self.assertTrue(abs(v[1] - 1.0 / math.sqrt(2)) < 1e-7)
            del v
            v, _ = g.power_iteration(tolerance=1e-3)
            self.assertTrue(abs(v[0] - 1.0 / math.sqrt(2)) < 1e-3)
            self.assertTrue(abs(v[1] - 1.0 / math.sqrt(2)) < 1e-3)
            del v

            del g

        def test_power_iteration_bug(self):
            g = Graph()
            g[0, 0] = 0.0
            g[1, 1] = 1.0

            v = g.power_iteration()[0].as_dict()
            self.assertEqual(v, {0: 0.0, 1: 1.0})

            del g

        def test_batch(self):
            test_indices = np.array([[0, 1], [1, 2], [2, 0]])
            test_weights = np.array([1.0, 2.0, 3.0])

            g = Graph(directed=True)
            self.assertEqual(g.flags, TVG_FLAGS_DIRECTED)
            self.assertTrue(g.directed)

            g.add_edges([], [])
            g.add_edges(test_indices, test_weights)
            indices, weights = g.edges()
            self.assertEqual(indices.tolist(), [[2, 0], [0, 1], [1, 2]])
            self.assertEqual(weights.tolist(), [3.0, 1.0, 2.0])
            self.assertEqual(g.num_edges, 3)
            self.assertEqual(len(g), 3)
            self.assertEqual(g.nodes().tolist(), [0, 1, 2])
            self.assertEqual(g.num_nodes, 3)

            self.assertEqual(list(g.keys()), [(2, 0), (0, 1), (1, 2)])
            self.assertEqual(list(g.values()), [3.0, 1.0, 2.0])
            self.assertEqual(list(g.items()), [((2, 0), 3.0), ((0, 1), 1.0), ((1, 2), 2.0)])

            indices, _ = g.edges(ret_weights=False)
            self.assertEqual(indices.tolist(), [[2, 0], [0, 1], [1, 2]])

            _, weights = g.edges(ret_indices=False)
            self.assertEqual(weights.tolist(), [3.0, 1.0, 2.0])

            for i in range(3):
                indices, weights = g.adjacent_edges(i)
                self.assertEqual(indices.tolist(), [(i + 1) % 3])
                self.assertEqual(weights.tolist(), [i + 1.0])

                indices, _ = g.adjacent_edges(i, ret_weights=False)
                self.assertEqual(indices.tolist(), [(i + 1) % 3])

                _, weights = g.adjacent_edges(i, ret_indices=False)
                self.assertEqual(weights.tolist(), [i + 1.0])

                with self.assertRaises(ValueError):
                    g.adjacent_edges(i, ret_indices=False, as_dict=True)

                result = g.adjacent_edges(i, ret_weights=False, as_dict=True)
                self.assertEqual(result, {(i + 1) % 3: None})

                result = g.adjacent_edges(i, as_dict=True)
                self.assertEqual(result, {(i + 1) % 3: i + 1})

            g.add_edges([], [])
            g.add_edges(test_indices, test_weights)
            indices, weights = g.edges()
            self.assertEqual(indices.tolist(), [[2, 0], [0, 1], [1, 2]])
            self.assertEqual(weights.tolist(), [6.0, 2.0, 4.0])

            g.add_edges([])
            g.add_edges(test_indices)
            indices, weights = g.edges()
            self.assertEqual(indices.tolist(), [[2, 0], [0, 1], [1, 2]])
            self.assertEqual(weights.tolist(), [7.0, 3.0, 5.0])

            g.sub_edges([], [])
            g.sub_edges(test_indices, -test_weights)
            indices, weights = g.edges()
            self.assertEqual(indices.tolist(), [[2, 0], [0, 1], [1, 2]])
            self.assertEqual(weights.tolist(), [10.0, 4.0, 7.0])

            g.sub_edges([])
            g.sub_edges(test_indices)
            indices, weights = g.edges()
            self.assertEqual(indices.tolist(), [[2, 0], [0, 1], [1, 2]])
            self.assertEqual(weights.tolist(), [9.0, 3.0, 6.0])

            g.set_edges([], [])
            g.set_edges(test_indices, test_weights)
            indices, weights = g.edges()
            self.assertEqual(indices.tolist(), [[2, 0], [0, 1], [1, 2]])
            self.assertEqual(weights.tolist(), [3.0, 1.0, 2.0])

            g.set_edges([])
            g.set_edges(test_indices)
            indices, weights = g.edges()
            self.assertEqual(indices.tolist(), [[2, 0], [0, 1], [1, 2]])
            self.assertEqual(weights.tolist(), [1.0, 1.0, 1.0])

            g.del_edges([])
            g.del_edges(test_indices)
            self.assertEqual(g.edges()[0].tolist(), [])
            self.assertEqual(g.num_edges, 0)
            self.assertEqual(len(g), 0)
            self.assertEqual(g.nodes().tolist(), [])
            self.assertEqual(g.num_nodes, 0)

            g.set_edges(set())
            g.set_edges(set([tuple(i) for i in test_indices]))
            indices, weights = g.edges()
            self.assertEqual(indices.tolist(), [[2, 0], [0, 1], [1, 2]])
            self.assertEqual(weights.tolist(), [1.0, 1.0, 1.0])

            g.del_edges(set())
            g.del_edges(set([tuple(i) for i in test_indices]))
            self.assertEqual(g.edges()[0].tolist(), [])
            self.assertEqual(g.num_edges, 0)
            self.assertEqual(len(g), 0)
            self.assertEqual(g.nodes().tolist(), [])
            self.assertEqual(g.num_nodes, 0)

            for i in range(1000):
                g.add_edge((i, i + 1), 1.0)
            indices, _ = g.edges(ret_weights=False)
            indices = [tuple(x) for x in indices]
            for i in range(1000):
                self.assertIn((i, i + 1), indices)
                indices.remove((i, i + 1))
            self.assertEqual(len(indices), 0)
            self.assertEqual(g.num_edges, 1000)
            self.assertEqual(len(g), 1000)

            del g

        def test_directed(self):
            g = Graph(directed=True)
            g[1, 1] = 2.0
            g[1, 2] = 1.0
            self.assertEqual(g[1, 1], 2.0)
            self.assertEqual(g[1, 2], 1.0)
            self.assertEqual(g[2, 1], 0.0)
            indices, weights = g.edges()
            self.assertEqual(indices.tolist(), [[1, 1], [1, 2]])
            self.assertEqual(weights.tolist(), [2.0, 1.0])
            self.assertEqual(g.num_edges, 2)
            self.assertEqual(g.nodes().tolist(), [1, 2])
            self.assertEqual(g.num_nodes, 2)
            del g

        def test_undirected(self):
            g = Graph(directed=False)
            g[1, 1] = 2.0
            g[1, 2] = 1.0
            self.assertEqual(g[1, 1], 2.0)
            self.assertEqual(g[1, 2], 1.0)
            self.assertEqual(g[2, 1], 1.0)
            indices, weights = g.edges()
            self.assertEqual(indices.tolist(), [[1, 1], [1, 2]])
            self.assertEqual(weights.tolist(), [2.0, 1.0])
            self.assertEqual(g.num_edges, 2)
            self.assertEqual(g.nodes().tolist(), [1, 2])
            self.assertEqual(g.num_nodes, 2)
            del g

        def test_flags(self):
            g = Graph()
            self.assertEqual(g.flags, 0)
            self.assertFalse(g.directed)
            g[0, 0] = 0.0
            self.assertTrue(g.has_edge((0, 0)))
            g.add_edge((0, 0), 1.0)
            self.assertEqual(g[0, 0], 1.0)
            g.add_edge((0, 0), -1.0)
            self.assertTrue(g.has_edge((0, 0)))
            self.assertEqual(g[0, 0], 0.0)
            g.sub_edge((0, 0), 1.0)
            self.assertEqual(g[0, 0], -1.0)
            g.sub_edge((0, 0), -1.0)
            self.assertTrue(g.has_edge((0, 0)))
            self.assertEqual(g[0, 0], 0.0)
            del g

            g = Graph(nonzero=True)
            self.assertEqual(g.flags, TVG_FLAGS_NONZERO)
            self.assertFalse(g.directed)
            self.assertEqual(g.eps, 0.0)
            g[0, 0] = 0.0
            self.assertFalse(g.has_edge((0, 0)))
            g.add_edge((0, 0), 1.0)
            self.assertEqual(g[0, 0], 1.0)
            g.add_edge((0, 0), -0.75)
            self.assertEqual(g[0, 0], 0.25)
            g.add_edge((0, 0), -0.25)
            self.assertFalse(g.has_edge((0, 0)))
            g.sub_edge((0, 0), 1.0)
            self.assertEqual(g[0, 0], -1.0)
            g.sub_edge((0, 0), -0.75)
            self.assertEqual(g[0, 0], -0.25)
            g.sub_edge((0, 0), -0.25)
            self.assertFalse(g.has_edge((0, 0)))
            del g

            g = Graph(nonzero=True)
            g.eps = 0.5
            self.assertEqual(g.flags, TVG_FLAGS_NONZERO)
            self.assertFalse(g.directed)
            self.assertEqual(g.eps, 0.5)
            g[0, 0] = 0.0
            self.assertFalse(g.has_edge((0, 0)))
            g.add_edge((0, 0), 1.0)
            self.assertEqual(g[0, 0], 1.0)
            g.add_edge((0, 0), -0.25)
            self.assertEqual(g[0, 0], 0.75)
            g.add_edge((0, 0), -0.25)
            self.assertFalse(g.has_edge((0, 0)))
            g.sub_edge((0, 0), 1.0)
            self.assertEqual(g[0, 0], -1.0)
            g.sub_edge((0, 0), -0.25)
            self.assertEqual(g[0, 0], -0.75)
            g.sub_edge((0, 0), -0.25)
            self.assertFalse(g.has_edge((0, 0)))
            del g

            g = Graph(positive=True)
            self.assertEqual(g.flags, TVG_FLAGS_NONZERO | TVG_FLAGS_POSITIVE)
            self.assertFalse(g.directed)
            self.assertEqual(g.eps, 0.0)
            g[0, 0] = 0.0
            self.assertFalse(g.has_edge((0, 0)))
            g.add_edge((0, 0), 1.0)
            self.assertEqual(g[0, 0], 1.0)
            g.add_edge((0, 0), -0.75)
            self.assertEqual(g[0, 0], 0.25)
            g.add_edge((0, 0), -0.25)
            self.assertFalse(g.has_edge((0, 0)))
            g.sub_edge((0, 0), 1.0)
            self.assertFalse(g.has_edge((0, 0)))
            g.sub_edge((0, 0), -0.25)
            self.assertEqual(g[0, 0], 0.25)
            del g

            g = Graph(positive=True)
            g.eps = 0.5
            self.assertEqual(g.flags, TVG_FLAGS_NONZERO | TVG_FLAGS_POSITIVE)
            self.assertFalse(g.directed)
            self.assertEqual(g.eps, 0.5)
            g[0, 0] = 0.0
            self.assertFalse(g.has_edge((0, 0)))
            g.add_edge((0, 0), 1.0)
            self.assertEqual(g[0, 0], 1.0)
            g.add_edge((0, 0), -0.25)
            self.assertEqual(g[0, 0], 0.75)
            g.add_edge((0, 0), -0.25)
            self.assertFalse(g.has_edge((0, 0)))
            g.sub_edge((0, 0), 1.0)
            self.assertFalse(g.has_edge((0, 0)))
            g.sub_edge((0, 0), -0.25)
            self.assertFalse(g.has_edge((0, 0)))
            g.sub_edge((0, 0), -0.5)
            self.assertFalse(g.has_edge((0, 0)))
            g.sub_edge((0, 0), -0.75)
            self.assertEqual(g[0, 0], 0.75)
            del g

        def test_bfs(self):
            g = Graph(directed=True)
            g[0, 1] = 1.0
            g[1, 2] = 1.0
            g[2, 3] = 1.0
            g[3, 4] = 1.5
            g[2, 4] = 1.5

            indices, weights = g.edges()
            self.assertEqual(indices.tolist(), [[0, 1], [1, 2], [2, 3], [2, 4], [3, 4]])
            self.assertEqual(weights.tolist(), [1.0, 1.0, 1.0, 1.5, 1.5])

            value = g.distance_count(100, 0)
            self.assertEqual(value, np.inf)
            value = g.distance_weight(100, 0)
            self.assertEqual(value, np.inf)

            value = g.distance_count(0, 0)
            self.assertEqual(value, 0)
            value = g.distance_weight(0, 0)
            self.assertEqual(value, 0.0)

            value = g.distance_count(0, 4)
            self.assertEqual(value, 3)
            value = g.distance_weight(0, 4)
            self.assertEqual(value, 3.5)

            results = g.bfs_count(0, max_count=2)
            self.assertEqual(results, [(0.0, 0, None, 0), (1.0, 1, 0, 1), (2.0, 2, 1, 2)])
            results = g.bfs_count(0)
            self.assertEqual(results, [(0.0, 0, None, 0), (1.0, 1, 0, 1), (2.0, 2, 1, 2), (3.0, 3, 2, 3), (3.5, 3, 2, 4)])

            results = g.bfs_weight(0, max_weight=2.0)
            self.assertEqual(results, [(0.0, 0, None, 0), (1.0, 1, 0, 1), (2.0, 2, 1, 2)])
            results = g.bfs_weight(0)
            self.assertEqual(results, [(0.0, 0, None, 0), (1.0, 1, 0, 1), (2.0, 2, 1, 2), (3.0, 3, 2, 3), (3.5, 3, 2, 4)])

            del g

        def test_mul_const(self):
            g = Graph()
            g[0, 0] = 1.0
            g.mul_const(-1.0)
            self.assertEqual(g[0, 0], -1.0)
            g.mul_const(0.0)
            self.assertTrue(g.has_edge((0, 0)))
            self.assertEqual(g[0, 0], 0.0)
            del g

            g = Graph(nonzero=True)
            g[0, 0] = 1.0
            g.mul_const(-1.0)
            self.assertEqual(g[0, 0], -1.0)
            g.mul_const(0.0)
            self.assertFalse(g.has_edge((0, 0)))
            del g

            g = Graph(positive=True)
            g[0, 0] = 1.0
            g.mul_const(-1.0)
            self.assertFalse(g.has_edge((0, 0)))
            del g

            g = Graph(positive=True)
            g[0, 0] = 1.0
            for i in range(200):
                g.mul_const(0.5)
                if not g.has_edge((0, 0)):
                    break
            else:
                self.assertTrue(False)

        def test_weights(self):
            g = Graph(directed=True)
            g[0, 0] = 1.0
            g[0, 1] = 2.0
            g[1, 0] = 3.0
            g[1, 2] = 0.0
            g[2, 3] = 0.0

            d = g.in_degrees()
            indices, weights = d.entries()
            self.assertEqual(indices.tolist(), [0, 1, 2, 3])
            self.assertEqual(weights.tolist(), [2.0, 1.0, 1.0, 1.0])
            d = g.in_weights()
            indices, weights = d.entries()
            self.assertEqual(indices.tolist(), [0, 1, 2, 3])
            self.assertEqual(weights.tolist(), [4.0, 2.0, 0.0, 0.0])

            d = g.out_degrees()
            indices, weights = d.entries()
            self.assertEqual(indices.tolist(), [0, 1, 2])
            self.assertEqual(weights.tolist(), [2.0, 2.0, 1.0])
            d = g.out_weights()
            indices, weights = d.entries()
            self.assertEqual(indices.tolist(), [0, 1, 2])
            self.assertEqual(weights.tolist(), [3.0, 3.0, 0.0])
            del g

        def test_anomalies(self):
            g = Graph(directed=False)

            a = g.degree_anomalies()
            indices, weights = a.entries()
            self.assertEqual(indices.tolist(), [])
            self.assertEqual(weights.tolist(), [])
            a = g.weight_anomalies()
            indices, weights = a.entries()
            self.assertEqual(indices.tolist(), [])
            self.assertEqual(weights.tolist(), [])

            c = 0
            for i in range(5):
                for j in range(i + 1):
                    if c % 3 != 0: g[i, j] = c
                    c += 1

            a = g.degree_anomalies()
            indices, weights = a.entries()
            self.assertEqual(indices.tolist(), [0, 1, 2, 3, 4])
            self.assertEqual(weights.tolist(), [-2.5, 1.5999999046325684, -0.6666667461395264, -1.0, 0.5])
            a = g.weight_anomalies()
            indices, weights = a.entries()
            self.assertEqual(indices.tolist(), [0, 1, 2, 3, 4])
            self.assertEqual(weights.tolist(), [-34.90909194946289,-9.119998931884766, -7.0588226318359375,
                                                -5.392856597900391, 18.39583396911621])
            del g

        def test_repr(self):
            g = Graph()
            self.assertEqual(repr(g), "Graph({})")
            for i in range(10):
                g[1, i] = 1.0
            expected = "Graph({(0, 1): X, (1, 1): X, (1, 2): X, (1, 3): X, (1, 4): X, (1, 5): X, (1, 6): X, (1, 7): X, (1, 8): X, (1, 9): X})"
            self.assertEqual(repr(g).replace("1.000000", "X"), expected)
            g[1, 10] = 2.0
            expected = "Graph({(0, 1): X, (1, 1): X, (1, 2): X, (1, 3): X, (1, 4): X, (1, 5): X, (1, 6): X, (1, 7): X, (1, 8): X, (1, 9): X, ...})"
            self.assertEqual(repr(g).replace("1.000000", "X"), expected)
            del g

            l = Node()
            self.assertEqual(repr(l), "Node({})")
            l["attr1"] = "sample attr1"
            l["attr2"] = "sample attr2"
            l["attr3"] = "sample attr3"
            self.assertEqual(repr(l), "Node({'attr1': 'sample attr1', 'attr2': 'sample attr2', 'attr3': 'sample attr3'})")
            del l

        def test_filter_nodes(self):
            g = Graph(directed=True)
            g[0, 1] = 1.0
            g[1, 2] = 2.0
            g[2, 3] = 3.0
            g[3, 4] = 4.0
            g[2, 4] = 5.0

            h = g.filter_nodes([1, 2, 3])
            indices, weights = h.edges()
            self.assertEqual(indices.tolist(), [[1, 2], [2, 3]])
            self.assertEqual(weights.tolist(), [2.0, 3.0])

            del g
            del h

        def test_normalize(self):
            g = Graph(directed=True)
            g[0, 1] = 1.0
            g[0, 2] = 0.5
            g[0, 3] = 0.5
            g2 = g.normalize()
            self.assertEqual(g2.as_dict(), {(0, 1): 0.5, (0, 2): 0.5, (0, 3): 0.5})
            del g2

            g[2, 1] = 3.0
            g2 = g.normalize()
            self.assertEqual(g2.as_dict(), {(0, 1): 0.125, (0, 2): 0.5, (0, 3): 0.5, (2, 1): 0.25})
            del g2
            del g

            g = Graph(directed=False)
            g[0, 1] = 1.0
            g[0, 2] = 0.5
            g[0, 3] = 0.5
            g2 = g.normalize()
            self.assertEqual(g2.as_dict(), {(0, 1): 0.5, (0, 2): 0.5, (0, 3): 0.5})
            del g2

            g[2, 1] = 3.0
            g2 = g.normalize()
            self.assertEqual(g2.num_edges, 4)
            self.assertEqual(g2[0, 1], 0.125)
            self.assertTrue(abs(g2[0, 2] - 0.07142858) < 1e-7)
            self.assertEqual(g2[0, 3], 0.5)
            self.assertTrue(abs(g2[1, 2] - 0.21428572) < 1e-7)
            del g2
            del g

        def test_as_dict(self):
            g = Graph(directed=True)
            for i in range(100):
                s, t = i//10, i%10
                g[s, t] = i

            with self.assertRaises(ValueError):
                g.edges(ret_indices=False, as_dict=True)

            result = g.edges(ret_weights=False, as_dict=True)
            self.assertEqual(len(result), 100)
            for i in range(100):
                s, t = i//10, i%10
                self.assertEqual(result[s, t], None)

            result = g.as_dict()
            self.assertEqual(len(result), 100)
            for i in range(100):
                s, t = i//10, i%10
                self.assertEqual(result[s, t], i)

            del g
            g = Graph.from_dict(result, directed=True)

            result = g.as_dict()
            self.assertEqual(len(result), 100)
            for i in range(100):
                s, t = i//10, i%10
                self.assertEqual(result[s, t], i)

            del g

        def test_top_edges(self):
            g = Graph(directed=True)

            for i in range(100):
                s, t = i//10, i%10
                g[s, t] = (i * 13) % 100

            indices, weights = g.top_edges(5)
            self.assertEqual(indices.tolist(), [[2, 3], [4, 6], [6, 9], [9, 2], [1, 5]])
            self.assertEqual(weights.tolist(), [99.0, 98.0, 97.0, 96.0, 95.0])

            indices, _ = g.top_edges(5, ret_weights=False)
            self.assertEqual(indices.tolist(), [[2, 3], [4, 6], [6, 9], [9, 2], [1, 5]])

            _, weights = g.top_edges(5, ret_indices=False)
            self.assertEqual(weights.tolist(), [99.0, 98.0, 97.0, 96.0, 95.0])

            with self.assertRaises(ValueError):
                g.top_edges(5, ret_indices=False, as_dict=True)

            result = g.top_edges(5, ret_weights=False, as_dict=True)
            self.assertEqual(result, {(2, 3): None, (4, 6): None, (6, 9): None, (9, 2): None, (1, 5): None})

            result = g.top_edges(5, as_dict=True)
            self.assertEqual(result, {(2, 3): 99.0, (4, 6): 98.0, (6, 9): 97.0, (9, 2): 96.0, (1, 5): 95.0})

            del g

        def test_duplicate(self):
            g = Graph(directed=True)

            for i in range(100):
                s, t = i//10, i%10
                g[s, t] = i

            revision = g.revision
            g2 = g.duplicate()

            for i in range(100):
                s, t = i//10, i%10
                g[s, t] = 1.0

            self.assertNotEqual(g.revision, revision)
            self.assertEqual(g2.revision, revision)

            for i in range(100):
                s, t = i//10, i%10
                self.assertEqual(g2[s, t], i)

            del g
            del g2

    class TVGTests(unittest.TestCase):
        def test_lookup(self):
            tvg = TVG(positive=True)
            self.assertEqual(tvg.flags, TVG_FLAGS_POSITIVE)
            mem = tvg.memory_usage

            g1 = Graph(positive=True)
            tvg.link_graph(g1, 100)
            self.assertEqual(g1.flags, TVG_FLAGS_NONZERO | TVG_FLAGS_POSITIVE)
            self.assertEqual(g1.ts, 100)
            self.assertEqual(g1.id, None)

            g2 = Graph(positive=True)
            tvg.link_graph(g2, 200)
            self.assertEqual(g2.flags, TVG_FLAGS_NONZERO | TVG_FLAGS_POSITIVE)
            self.assertEqual(g2.ts, 200)
            self.assertEqual(g2.id, None)

            g3 = Graph(positive=True)
            tvg.link_graph(g3, 300)
            self.assertEqual(g3.flags, TVG_FLAGS_NONZERO | TVG_FLAGS_POSITIVE)
            self.assertEqual(g3.ts, 300)
            self.assertEqual(g3.id, None)
            self.assertGreater(tvg.memory_usage, mem)

            g = tvg.lookup_le(50)
            self.assertEqual(g, None)
            g = tvg.lookup_ge(50)
            self.assertEqual(g, g1)

            g = tvg.lookup_le(150)
            self.assertEqual(g, g1)
            g = tvg.lookup_ge(150)
            self.assertEqual(g, g2)

            g = tvg.lookup_le(250)
            self.assertEqual(g, g2)
            g = tvg.lookup_ge(250)
            self.assertEqual(g, g3)

            g = tvg.lookup_le(350)
            self.assertEqual(g, g3)
            g = tvg.lookup_ge(350)
            self.assertEqual(g, None)

            g = tvg.lookup_near(149)
            self.assertEqual(g, g1)
            g = tvg.lookup_near(151)
            self.assertEqual(g, g2)

            # For backwards compatibility, we still allow passing float values.

            g = tvg.lookup_ge(100.0)
            self.assertEqual(g, g1)
            g = tvg.lookup_ge(100.01)
            self.assertEqual(g, g2)

            g = tvg.lookup_le(200.0)
            self.assertEqual(g, g2)
            g = tvg.lookup_le(199.99)
            self.assertEqual(g, g1)

            g = tvg.lookup_near(149.49)
            self.assertEqual(g, g1)
            g = tvg.lookup_near(150.51)
            self.assertEqual(g, g2)

            del tvg

        def test_link(self):
            tvg = TVG()
            g1 = Graph()
            g2 = Graph(directed=True)

            tvg.link_graph(g1, 10)
            with self.assertRaises(RuntimeError):
                tvg.link_graph(g1, 20)
            with self.assertRaises(RuntimeError):
                tvg.link_graph(g2, 20)

            g = tvg.lookup_near(10)
            self.assertEqual(g.ts, 10)
            self.assertEqual(addressof(g._obj.contents), addressof(g1._obj.contents))
            del tvg

        def test_compress(self):
            source = np.random.rand(100)

            tvg = TVG(positive=True)

            for t, s in enumerate(source):
                g = Graph()
                g[0, 0] = s
                tvg.link_graph(g, t)

            tvg.compress(step=5, offset=100)

            t = 0
            for g in tvg:
                self.assertEqual(g.ts, t)
                self.assertTrue(abs(g[0, 0] - np.sum(source[t:t+5])) < 1e-6)
                t += 5

            del tvg

        def test_load(self):
            filename_graphs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../datasets/example/example-tvg.graph")
            filename_nodes = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../datasets/example/example-tvg.nodes")
            tvg = TVG.load(filename_graphs, nodes=filename_nodes)
            tvg.set_primary_key(["a", "b", "c"])
            tvg.set_primary_key(["text"])

            l = tvg.node_by_index(1)
            self.assertEqual(l.text, "polic")
            l = tvg.node_by_index(362462)
            self.assertEqual(l.text, "Jay Wright (basketball)")
            with self.assertRaises(KeyError):
                tvg.node_by_index(5)

            l = tvg.node_by_text("polic")
            self.assertEqual(l.index, 1)
            l = tvg.node_by_text("Jay Wright (basketball)")
            self.assertEqual(l.index, 362462)
            with self.assertRaises(KeyError):
                tvg.node_by_text("should-not-exist")

            timestamps = []
            edges = []
            for g in tvg:
                self.assertEqual(g.revision, 0)
                timestamps.append(g.ts)
                edges.append(g.num_edges)

            self.assertEqual(timestamps, [      0,  130000,  141000,  164000,  176000,  272000,  376000,  465000,  666000,  682000,  696000,
                                           770000,  848000, 1217000, 1236000, 1257000, 1266000, 1431000, 1515000, 1539000, 1579000, 1626000,
                                          1763000, 1803000, 1834000, 1920000, 1967000, 2021000, 2188000, 2405000, 2482000, 2542000, 2551000,
                                          2583000, 2591000, 2604000, 2620000, 2830000, 2852000, 2957000, 3008000])

            self.assertEqual(edges, [155, 45, 1250, 90, 178, 85, 367, 98, 18, 528, 158, 201, 267, 214, 613, 567, 1, 137, 532, 59, 184,
                                     40, 99, 285, 326, 140, 173, 315, 211, 120, 19, 137, 170, 42, 135, 348, 168, 132, 147, 218, 321])

            g = tvg.lookup_near(141000)
            self.assertTrue(abs(g[6842, 249977] - 0.367879) < 1e-7)

            g = tvg.lookup_near(1257000)
            self.assertTrue(abs(g[1291, 3529] - 1.013476) < 1e-7)

            g = tvg.lookup_near(2604000)
            self.assertTrue(abs(g[121, 1154] - 3.000000) < 1e-7)

            tvg.compress(step=600000)

            timestamps = []
            edges = []
            for g in tvg:
                timestamps.append(g.ts)
                edges.append(g.num_edges)

            self.assertEqual(timestamps, [0, 600000, 1200000, 1800000, 2400000, 3000000])
            self.assertEqual(edges, [2226, 1172, 2446, 1448, 1632, 321])

            timestamps = []
            edges = []
            for g in reversed(tvg):
                timestamps.append(g.ts)
                edges.append(g.num_edges)

            self.assertEqual(timestamps, [3000000, 2400000, 1800000, 1200000, 600000, 0])
            self.assertEqual(edges, [321, 1632, 1448, 2446, 1172, 2226])

            tvg.compress(step=np.inf, offset=100000)

            timestamps = []
            edges = []
            for g in tvg:
                timestamps.append(g.ts)
                edges.append(g.num_edges)

            self.assertEqual(timestamps, [100000])
            self.assertEqual(edges, [9097])

            del tvg

        def test_load_crlf(self):
            filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../datasets/example/example-tvg.graph")
            with open(filename, "rb") as fp:
                content = fp.read()

            temp_graph = tempfile.NamedTemporaryFile()
            temp_graph.write(content.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n'))
            temp_graph.flush()

            filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../datasets/example/example-tvg.nodes")
            with open(filename, "rb") as fp:
                content = fp.read()

            temp_nodes = tempfile.NamedTemporaryFile()
            temp_nodes.write(content.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n'))
            temp_nodes.flush()

            tvg = TVG.load(temp_graph.name, nodes=temp_nodes.name, primary_key=["text"])

            temp_graph.close()
            temp_nodes.close()

            l = tvg.node_by_index(1)
            self.assertEqual(l.text, "polic")
            l = tvg.node_by_index(362462)
            self.assertEqual(l.text, "Jay Wright (basketball)")
            with self.assertRaises(KeyError):
                tvg.node_by_index(5)

            l = tvg.node_by_text("polic")
            self.assertEqual(l.index, 1)
            l = tvg.node_by_text("Jay Wright (basketball)")
            self.assertEqual(l.index, 362462)
            with self.assertRaises(KeyError):
                tvg.node_by_text("should-not-exist")

            timestamps = []
            edges = []
            for g in tvg:
                self.assertEqual(g.revision, 0)
                timestamps.append(g.ts)
                edges.append(g.num_edges)

            self.assertEqual(timestamps, [      0,  130000,  141000,  164000,  176000,  272000,  376000,  465000,  666000,  682000,  696000,
                                           770000,  848000, 1217000, 1236000, 1257000, 1266000, 1431000, 1515000, 1539000, 1579000, 1626000,
                                          1763000, 1803000, 1834000, 1920000, 1967000, 2021000, 2188000, 2405000, 2482000, 2542000, 2551000,
                                          2583000, 2591000, 2604000, 2620000, 2830000, 2852000, 2957000, 3008000])

            self.assertEqual(edges, [155, 45, 1250, 90, 178, 85, 367, 98, 18, 528, 158, 201, 267, 214, 613, 567, 1, 137, 532, 59, 184,
                                     40, 99, 285, 326, 140, 173, 315, 211, 120, 19, 137, 170, 42, 135, 348, 168, 132, 147, 218, 321])

            del tvg

        def test_nodes(self):
            tvg = TVG()
            tvg.set_primary_key("text")

            l = tvg.Node(text="A", other="sample text")
            self.assertEqual(l.index, 0)
            self.assertEqual(l.text, "A")
            l = tvg.Node(text="B")
            self.assertEqual(l.index, 1)
            self.assertEqual(l.text, "B")
            l = tvg.Node(text="C")
            self.assertEqual(l.index, 2)
            self.assertEqual(l.text, "C")
            l = tvg.Node(text="D")
            self.assertEqual(l.index, 3)
            self.assertEqual(l.text, "D")

            l = tvg.Node(text="A")
            self.assertEqual(l.index, 0)
            self.assertEqual(l.text, "A")
            self.assertEqual(l['other'], "sample text")

            l = tvg.node_by_index(1)
            self.assertEqual(l.index, 1)
            self.assertEqual(l.text, "B")

            with self.assertRaises(KeyError):
                tvg.node_by_index(4)

            l = tvg.node_by_text("C")
            self.assertEqual(l.index, 2)
            self.assertEqual(l.text, "C")

            with self.assertRaises(KeyError):
                tvg.node_by_text("E")

            l.unlink()

            with self.assertRaises(KeyError):
                tvg.node_by_index(2)

            with self.assertRaises(KeyError):
                tvg.node_by_text("C")

            l = Node(text="Z")
            tvg.link_node(l, 4)
            self.assertEqual(l.index, 4)
            self.assertEqual(l.text, "Z")

            l = tvg.node_by_index(4)
            self.assertEqual(l.index, 4)
            self.assertEqual(l.text, "Z")

            l = tvg.node_by_text("Z")
            self.assertEqual(l.index, 4)
            self.assertEqual(l.text, "Z")

            l = Node(text="Z")
            self.assertEqual(l.index, 0xffffffffffffffff)
            with self.assertRaises(RuntimeError):
                tvg.link_node(l)
            self.assertEqual(l.index, 0xffffffffffffffff)

            del tvg

        def test_node_attrs(self):
            tvg = TVG()
            tvg.set_primary_key("text")

            l = tvg.Node(text="sample text")
            self.assertEqual(l.as_dict(), {'text': "sample text"})

            with self.assertRaises(KeyError):
                l["text"] = "other text"

            with self.assertRaises(KeyError):
                l[""] = "empty key"

            l["attr1"] = "sample attr1"
            l["attr2"] = "sample attr2"
            l["attr3"] = "sample attr3"
            self.assertEqual(l.as_dict(), {'text': "sample text",
                                           'attr1': "sample attr1",
                                           'attr2': "sample attr2",
                                           'attr3': "sample attr3"})

            l["attr1"] = "other attr1"
            self.assertEqual(l.as_dict(), {'text': "sample text",
                                           'attr1': "other attr1",
                                           'attr2': "sample attr2",
                                           'attr3': "sample attr3"})

            tvg.set_primary_key(["text", "attr1"])
            with self.assertRaises(KeyError):
                l["attr1"] = "sample attr1"

            del tvg

        def test_readonly(self):
            tvg = TVG(positive=True)
            g = Graph(positive=True)
            tvg.link_graph(g, 0)
            self.assertEqual(g.ts, 0)

            with self.assertRaises(RuntimeError):
                g.clear()
            with self.assertRaises(MemoryError):
                g[0, 0] = 1.0
            with self.assertRaises(MemoryError):
                g.add_edge((0, 0), 1.0)
            with self.assertRaises(RuntimeError):
                del g[0, 0]
            with self.assertRaises(RuntimeError):
                g.mul_const(2.0)
            with self.assertRaises(RuntimeError):
                g.eps = 2.0

            del tvg
            del g

        def test_sum_edges(self):
            tvg = TVG(positive=True)
            tvg.verbosity = True
            self.assertEqual(tvg.verbosity, True)

            g = Graph()
            g[0, 0] = 1.0
            tvg.link_graph(g, 100)

            g = Graph()
            g[0, 1] = 2.0
            tvg.link_graph(g, 200)

            g = Graph()
            g[0, 2] = 3.0
            tvg.link_graph(g, 300)

            with self.assertRaises(MemoryError):
                tvg.sum_edges(1, 0)

            g = tvg.sum_edges(51, 150, eps=0.5)
            self.assertEqual(g[0, 0], 1.0)
            self.assertEqual(g[0, 1], 0.0)
            self.assertEqual(g[0, 2], 0.0)

            g = tvg.sum_edges(151, 250, eps=0.5)
            self.assertEqual(g[0, 0], 0.0)
            self.assertEqual(g[0, 1], 2.0)
            self.assertEqual(g[0, 2], 0.0)

            g = tvg.sum_edges(251, 350, eps=0.5)
            self.assertEqual(g[0, 0], 0.0)
            self.assertEqual(g[0, 1], 0.0)
            self.assertEqual(g[0, 2], 3.0)

            g = tvg.sum_edges(51, 150, eps=0.5)
            self.assertEqual(g[0, 0], 1.0)
            self.assertEqual(g[0, 1], 0.0)
            self.assertEqual(g[0, 2], 0.0)

            del tvg

        def test_sum_edges_exp_precision(self):
            tvg = TVG(positive=True)
            beta = 0.3

            g = Graph()
            g[0, 0] = 1.0
            tvg.link_graph(g, 0)

            with self.assertRaises(MemoryError):
                tvg.sum_edges_exp(1, 0, beta=beta)

            g = tvg.sum_edges_exp(0, 100, beta=beta)
            self.assertTrue(abs(g[0, 0] - math.pow(beta, 100.0)) < 1e-7)

            g = tvg.sum_edges_exp(0, 0, beta=beta)
            self.assertTrue(abs(g[0, 0] - 1.0) < 1e-7)

            del g

        def test_sum_edges_exp_norm(self):
            source = np.random.rand(100)
            beta = 0.3

            tvg = TVG(positive=True)

            for t, s in enumerate(source):
                g = Graph()
                g[0, 0] = s
                tvg.link_graph(g, t)

            with self.assertRaises(MemoryError):
                tvg.sum_edges_exp_norm(1, 0, beta=beta)

            expected = 0.0
            for t, s in enumerate(source):
                g = tvg.sum_edges_exp_norm(0, t, beta=beta)
                expected = beta * expected + (1.0 - beta) * s
                self.assertTrue(abs(g[0, 0] - expected) < 1e-6)

            expected = 0.0
            for t, s in enumerate(source):
                g = tvg.sum_edges_exp_norm(0, t, log_beta=math.log(beta))
                expected = beta * expected + (1.0 - beta) * s
                self.assertTrue(abs(g[0, 0] - expected) < 1e-6)

            del tvg

        def test_count_edges(self):
            tvg = TVG(positive=True)

            g = Graph()
            g[0, 0] = 1.0
            tvg.link_graph(g, 100)

            g = Graph()
            g[0, 1] = 2.0
            tvg.link_graph(g, 200)

            g = Graph()
            g[0, 2] = 3.0
            tvg.link_graph(g, 300)

            with self.assertRaises(MemoryError):
                tvg.count_edges(1, 0)

            g = tvg.count_edges(51, 150)
            self.assertEqual(g[0, 0], 1.0)
            self.assertEqual(g[0, 1], 0.0)
            self.assertEqual(g[0, 2], 0.0)

            g = tvg.count_edges(151, 250)
            self.assertEqual(g[0, 0], 0.0)
            self.assertEqual(g[0, 1], 1.0)
            self.assertEqual(g[0, 2], 0.0)

            g = tvg.count_edges(251, 350)
            self.assertEqual(g[0, 0], 0.0)
            self.assertEqual(g[0, 1], 0.0)
            self.assertEqual(g[0, 2], 1.0)

            g = tvg.count_edges(51, 150)
            self.assertEqual(g[0, 0], 1.0)
            self.assertEqual(g[0, 1], 0.0)
            self.assertEqual(g[0, 2], 0.0)

            del tvg

        def test_count_nodes(self):
            tvg = TVG(positive=True)

            g = Graph()
            g[0, 0] = 1.0
            tvg.link_graph(g, 100)

            g = Graph()
            g[0, 1] = 2.0
            g[1, 2] = 2.0
            tvg.link_graph(g, 200)

            g = Graph()
            g[0, 2] = 3.0
            tvg.link_graph(g, 300)

            with self.assertRaises(MemoryError):
                tvg.count_nodes(1, 0)

            v = tvg.count_nodes(51, 150)
            self.assertEqual(v[0], 1.0)
            self.assertEqual(v[1], 0.0)
            self.assertEqual(v[2], 0.0)

            v = tvg.count_nodes(151, 250)
            self.assertEqual(v[0], 1.0)
            self.assertEqual(v[1], 1.0)
            self.assertEqual(v[2], 1.0)

            v = tvg.count_nodes(251, 350)
            self.assertEqual(v[0], 1.0)
            self.assertEqual(v[1], 0.0)
            self.assertEqual(v[2], 1.0)

            v = tvg.count_nodes(51, 150)
            self.assertEqual(v[0], 1.0)
            self.assertEqual(v[1], 0.0)
            self.assertEqual(v[2], 0.0)

            del tvg

        def test_count_graphs(self):
            tvg = TVG(positive=True)
            tvg.enable_query_cache(cache_size=0x8000) # 32 kB cache

            g = Graph()
            tvg.link_graph(g, 100)

            g = Graph()
            tvg.link_graph(g, 200)
            g = Graph()
            tvg.link_graph(g, 200)

            g = Graph()
            tvg.link_graph(g, 300)
            g = Graph()
            tvg.link_graph(g, 300)
            g = Graph()
            tvg.link_graph(g, 300)

            with self.assertRaises(MemoryError):
                tvg.count_graphs(1, 0)

            c = tvg.count_graphs(51, 150)
            self.assertEqual(c, 1)

            c = tvg.count_graphs(151, 250)
            self.assertEqual(c, 2)

            c = tvg.count_graphs(251, 350)
            self.assertEqual(c, 3)

            c = tvg.count_graphs(51, 150)
            self.assertEqual(c, 1)

            c = tvg.count_graphs(51, 350)
            self.assertEqual(c, 6)

            del tvg

        def test_topics(self):
            tvg = TVG(positive=True)

            g = Graph()
            g[0, 1] = 1.0
            tvg.link_graph(g, 100)

            g = Graph()
            g[0, 1] = 1.0
            tvg.link_graph(g, 200)

            g = Graph()
            g[1, 2] = 0.5
            tvg.link_graph(g, 200)

            g = Graph()
            g[0, 1] = 0.5
            tvg.link_graph(g, 300)

            g = Graph()
            g[0, 1] = 1.0
            tvg.link_graph(g, 300)

            g = Graph()
            g[1, 2] = 1.0
            tvg.link_graph(g, 300)

            with self.assertRaises(MemoryError):
                tvg.topics(1, 0)

            # |D(0) \cup D(1)| = 1.0
            # |D((0, 1))| = 1.0
            # |L((0, 1))| = 1.0
            # \sum exp(-\delta) = 1.0

            g = tvg.topics(51, 150)
            self.assertTrue(abs(g[0, 1] - 1.0) < 1e-7)

            # |D(0) \cup D(1)| = 2.0
            # |D((0, 1))| = 1.0
            # |L((0, 1))| = 1.0
            # \sum exp(-\delta) = 1.0

            g = tvg.topics(151, 250)
            self.assertTrue(abs(g[0, 1] - 2.0 / 3.0) < 1e-7)

            # |D(0) \cup D(1)| = 3.0
            # |D((0, 1))| = 2.0
            # |L((0, 1))| = 2.0
            # \sum exp(-\delta) = 1.5

            g = tvg.topics(251, 350)
            self.assertTrue(abs(g[0, 1] - 12.0 / 17.0) < 1e-7)

            # |D(0) \cup D(1)| = 6.0
            # |D((0, 1))| = 4.0
            # |L((0, 1))| = 4.0
            # \sum exp(-\delta) = 3.5

            g = tvg.topics(51, 350)
            self.assertTrue(abs(g[0, 1] - 28.0 / 37.0) < 1e-7)

            g = tvg.topics(0, 350)
            self.assertTrue(abs(g[0, 1] - 28.0 / 37.0) < 1e-7)

            g = tvg.topics(0, 400)
            self.assertTrue(abs(g[0, 1] - 28.0 / 37.0) < 1e-7)

            # |D(0) \cup D(1)| = 1.0
            # |D((0, 1))| = 1.0
            # |L((0, 1))| = 1.0
            # \sum exp(-\delta) = 1.0
            # \delta T = 1.0
            # |T(e)| = 1.0

            g = tvg.topics(51, 150, step=100, offset=51)
            self.assertTrue(abs(g[0, 1] - 1.0) < 1e-7)

            # |D(0) \cup D(1)| = 2.0
            # |D((0, 1))| = 1.0
            # |L((0, 1))| = 1.0
            # \sum exp(-\delta) = 1.0
            # \delta T = 1.0
            # |T(e)| = 1.0

            g = tvg.topics(151, 250, step=100, offset=51)
            self.assertTrue(abs(g[0, 1] - 3.0 / 4.0) < 1e-7)

            # |D(0) \cup D(1)| = 3.0
            # |D((0, 1))| = 2.0
            # |L((0, 1))| = 2.0
            # \sum exp(-\delta) = 1.5
            # \delta T = 1.0
            # |T(e)| = 1

            g = tvg.topics(251, 350, step=100, offset=51)
            self.assertTrue(abs(g[0, 1] - 18.0 / 23.0) < 1e-7)

            # |D(0) \cup D(1)| = 6.0
            # |D((0, 1))| = 4.0
            # |L((0, 1))| = 4.0
            # \sum exp(-\delta) = 3.5
            # \delta T = 3.0
            # |T(e)| = 3.0

            g = tvg.topics(51, 350, step=100, offset=51)
            self.assertTrue(abs(g[0, 1] - 14.0 / 17.0) < 1e-7)

            # |D(0) \cup D(1)| = 6.0
            # |D((0, 1))| = 4.0
            # |L((0, 1))| = 4.0
            # \sum exp(-\delta) = 3.5
            # \delta T = 4.0
            # |T(e)| = 3.0

            g = tvg.topics(0, 350, step=100, offset=51)
            self.assertTrue(abs(g[0, 1] - 126.0 / 167.0) < 1e-7)

            # |D(0) \cup D(1)| = 6.0
            # |D((0, 1))| = 4.0
            # |L((0, 1))| = 4.0
            # \sum exp(-\delta) = 3.5
            # \delta T = 5.0
            # |T(e)| = 3.0

            g = tvg.topics(0, 400, step=100, offset=51)
            self.assertTrue(abs(g[0, 1] - 126.0 / 181.0) < 1e-7)

            del tvg

        def test_topics_sum_weights(self):
            tvg = TVG(positive=True)

            # In the original publication, the metric is only defined
            # for sum_weights=False. In our implementation, however,
            # we also allow sum_weights=True, i.e., weights greater than 1.

            g = Graph()
            g[0, 1] = 50.0
            tvg.link_graph(g, 100)

            g = Graph()
            g[0, 1] = 50.0
            tvg.link_graph(g, 100)

            g = Graph()
            g[1, 2] = 1.0
            tvg.link_graph(g, 100)

            # |D(0) \cup D(1)| = 3.0
            # |D((0, 1))| = 2.0
            # |L((0, 1))| = 2.0
            # \sum exp(-\delta) = 100.0

            g = tvg.topics(51, 150)
            self.assertTrue(abs(g[0, 1] - 25.0 / 19.0) < 1e-7)

            # |D(0) \cup D(1)| = 3.0
            # |D((0, 1))| = 2.0
            # |L((0, 1))| = 2.0
            # \sum exp(-\delta) = 100.0
            # \delta T = 1.0
            # |T(e)| = 1

            g = tvg.topics(51, 150, step=100, offset=51)
            self.assertTrue(abs(g[0, 1] - 25.0 / 21.0) < 1e-7)

            del tvg

        def test_query_cache(self):
            tvg = TVG(positive=True)
            tvg.verbosity = True

            g = Graph()
            g[0, 0] = 1.0
            tvg.link_graph(g, 100)

            g = Graph()
            g[0, 1] = 2.0
            tvg.link_graph(g, 200)

            g = Graph()
            g[0, 2] = 3.0
            tvg.link_graph(g, 300)

            # test 1
            g = tvg.sum_edges(51, 150, eps=0.5)
            self.assertEqual(g[0, 0], 1.0)
            self.assertEqual(g[0, 1], 0.0)
            self.assertEqual(g[0, 2], 0.0)
            del g
            g = tvg.sum_edges(51, 150, eps=0.5)
            self.assertEqual(g[0, 0], 1.0)
            self.assertEqual(g[0, 1], 0.0)
            self.assertEqual(g[0, 2], 0.0)
            del g

            # test 2
            g1 = tvg.sum_edges(51, 150, eps=0.5)
            g2 = tvg.sum_edges(51, 150, eps=0.5)
            self.assertEqual(g2[0, 0], 1.0)
            self.assertEqual(g2[0, 1], 0.0)
            self.assertEqual(g2[0, 2], 0.0)
            del g2
            tvg.invalidate_queries(100, 100)
            g2 = tvg.sum_edges(51, 150, eps=0.5)
            self.assertEqual(g2[0, 0], 1.0)
            self.assertEqual(g2[0, 1], 0.0)
            self.assertEqual(g2[0, 2], 0.0)
            del g2
            del g1

            tvg.enable_query_cache(cache_size=0x8000) # 32 kB cache

            # test 3
            g = tvg.sum_edges(51, 150, eps=0.5)
            self.assertEqual(g[0, 0], 1.0)
            self.assertEqual(g[0, 1], 0.0)
            self.assertEqual(g[0, 2], 0.0)
            del g
            g = tvg.sum_edges(51, 150, eps=0.5)
            self.assertEqual(g[0, 0], 1.0)
            self.assertEqual(g[0, 1], 0.0)
            self.assertEqual(g[0, 2], 0.0)
            del g
            tvg.invalidate_queries(100, 100)
            g = tvg.sum_edges(51, 150, eps=0.5)
            self.assertEqual(g[0, 0], 1.0)
            self.assertEqual(g[0, 1], 0.0)
            self.assertEqual(g[0, 2], 0.0)
            del g

            tvg.disable_query_cache()
            tvg.enable_query_cache(cache_size=0x8000) # 32 kB cache

            # test 4
            g1 = tvg.sum_edges(51, 150, eps=0.5)
            g2 = tvg.sum_edges(51, 150, eps=0.5)
            self.assertEqual(g2[0, 0], 1.0)
            self.assertEqual(g2[0, 1], 0.0)
            self.assertEqual(g2[0, 2], 0.0)
            del g2
            tvg.invalidate_queries(100, 100)
            g2 = tvg.sum_edges(51, 150, eps=0.5)
            self.assertEqual(g2[0, 0], 1.0)
            self.assertEqual(g2[0, 1], 0.0)
            self.assertEqual(g2[0, 2], 0.0)
            del g2
            del g1

            del tvg

        def test_sparse_topics(self):
            tvg = TVG(positive=True)

            g = Graph()
            g[0, 1] = 1.0
            g[0, 2] = 0.5
            g[1, 2] = 0.5
            g[0, 3] = 0.25
            g[1, 3] = 0.25
            tvg.link_graph(g, 100)

            g = tvg.topics(51, 150)
            h = g.sparse_subgraph(num_seeds=1, num_neighbors=1)
            self.assertEqual(h.num_edges, 3)
            self.assertTrue(abs(h[0, 1] - 1.0) < 1e-7)
            self.assertTrue(abs(h[0, 2] - 2.0 / 3.0) < 1e-7)
            self.assertTrue(abs(h[1, 2] - 2.0 / 3.0) < 1e-7)

            del tvg

        def test_sample_eigenvectors(self):
            tvg = TVG(positive=True)

            g = Graph()
            g[0, 0] = 1.0
            tvg.link_graph(g, 100)

            g = Graph()
            g[1, 1] = 2.0
            tvg.link_graph(g, 200)

            g = Graph()
            g[2, 2] = 3.0
            tvg.link_graph(g, 300)

            values = tvg.sample_eigenvectors(50, 350, sample_width=101, sample_steps=3)
            values = _convert_values(values)
            self.assertEqual(len(values), 3)
            self.assertEqual(values[0], [1.0, 0.0, 0.0])
            self.assertEqual(values[1], [0.0, 1.0, 0.0])
            self.assertEqual(values[2][0], 0.0)
            self.assertEqual(values[2][1], 0.0)
            self.assertTrue(abs(values[2][2] - 1.0) < 1e-6)

            values = tvg.sample_eigenvectors(50, 350, sample_width=101, sample_steps=3, method='sum_edges')
            values = _convert_values(values)
            self.assertEqual(len(values), 3)
            self.assertEqual(values[0], [1.0, 0.0, 0.0])
            self.assertEqual(values[1], [0.0, 1.0, 0.0])
            self.assertEqual(values[2][0], 0.0)
            self.assertEqual(values[2][1], 0.0)
            self.assertTrue(abs(values[2][2] - 1.0) < 1e-6)

            values = tvg.sample_eigenvectors(50, 350, sample_width=101, sample_steps=3, method='count_edges')
            values = _convert_values(values)
            self.assertEqual(len(values), 3)
            self.assertEqual(values[0], [1.0, 0.0, 0.0])
            self.assertEqual(values[1], [0.0, 1.0, 0.0])
            self.assertEqual(values[2][0], 0.0)
            self.assertEqual(values[2][1], 0.0)
            self.assertTrue(abs(values[2][2] - 1.0) < 1e-6)

            values = tvg.sample_eigenvectors(50, 350, sample_width=101, sample_steps=3, method='topics')
            values = _convert_values(values)
            self.assertEqual(len(values), 3)
            self.assertEqual(values[0], [1.0, 0.0, 0.0])
            self.assertEqual(values[1], [0.0, 1.0, 0.0])
            self.assertEqual(values[2][0], 0.0)
            self.assertEqual(values[2][1], 0.0)
            self.assertTrue(abs(values[2][2] - 1.0) < 1e-6)

            del tvg

        def test_sample_graphs(self):
            tvg = TVG(positive=True)

            g = Graph()
            g[0, 0] = 1.0
            tvg.link_graph(g, 100)

            g = Graph()
            g[1, 1] = 2.0
            tvg.link_graph(g, 200)

            g = Graph()
            g[2, 2] = 3.0
            tvg.link_graph(g, 300)

            values = tvg.sample_graphs(50, 350, sample_width=101, sample_steps=3)
            values = _convert_values(values)
            self.assertEqual(len(values), 3)
            self.assertEqual(values[0, 0], [1.0, 0.0, 0.0])
            self.assertEqual(values[1, 1], [0.0, 2.0, 0.0])
            self.assertEqual(values[2, 2], [0.0, 0.0, 3.0])

            values = tvg.sample_graphs(50, 350, sample_width=101, sample_steps=3, method='sum_edges')
            values = _convert_values(values)
            self.assertEqual(len(values), 3)
            self.assertEqual(values[0, 0], [1.0, 0.0, 0.0])
            self.assertEqual(values[1, 1], [0.0, 2.0, 0.0])
            self.assertEqual(values[2, 2], [0.0, 0.0, 3.0])

            values = tvg.sample_graphs(50, 350, sample_width=101, sample_steps=3, method='count_edges')
            values = _convert_values(values)
            self.assertEqual(len(values), 3)
            self.assertEqual(values[0, 0], [1.0, 0.0, 0.0])
            self.assertEqual(values[1, 1], [0.0, 1.0, 0.0])
            self.assertEqual(values[2, 2], [0.0, 0.0, 1.0])

            values = tvg.sample_graphs(50, 350, sample_width=101, sample_steps=3, method='topics')
            values = _convert_values(values)
            self.assertEqual(len(values), 3)
            self.assertEqual(values[0, 0], [1.0, 0.0, 0.0])
            self.assertEqual(values[1, 1][0], 0.0)
            self.assertTrue(abs(values[1, 1][1] - 4.0 / 3.0) < 1e-6)
            self.assertEqual(values[1, 1][2], 0.0)
            self.assertEqual(values[2, 2], [0.0, 0.0, 1.5])

            del tvg

        def test__convert_values(self):
            values = [
                {(0, 0): 1.0, (0, 1): 0.0, (1, 1): 2.0, (2, 2): 2.0},
                {(0, 0): 1.0, (0, 1): 1.0, (1, 1): 1.0, (2, 2): 2.0},
                {(0, 0): 1.0, (0, 1): 2.0, (1, 1): 0.0, (2, 2): 2.0},
            ]

            expected = {
                (0, 0): [1.0, 1.0, 1.0],
                (0, 1): [0.0, 1.0, 2.0],
                (1, 1): [2.0, 1.0, 0.0],
                (2, 2): [2.0, 2.0, 2.0],
            }

            result = _convert_values(values)
            self.assertEqual(result, expected)

        def test_metric_entropy(self):
            tvg = TVG(positive=True)

            g = Graph()
            g[0, 0] = g[0, 1] = g[0, 2] = 1.0
            g[1, 1] = g[1, 2] = 1.0
            g[2, 2] = 1.0
            tvg.link_graph(g, 100)

            g = Graph()
            g[1, 1] = g[1, 2] = 2.0
            g[2, 2] = 2.0
            tvg.link_graph(g, 200)

            g = Graph()
            g[2, 2] = 3.0
            tvg.link_graph(g, 300)

            values = tvg.sample_eigenvectors(50, 350, sample_width=101, sample_steps=3)
            values = metric_entropy(values, num_bins=6)

            P = np.array([3, 0, 0, 3, 2, 1]) / 9.0
            self.assertEqual(len(values), 3)
            self.assertEqual(values[0], - np.log(P[3]) * P[3] - np.log(P[0]) * P[0] - np.log(P[0]) * P[0])
            self.assertEqual(values[1], - np.log(P[3]) * P[3] - np.log(P[4]) * P[4] - np.log(P[0]) * P[0])
            self.assertEqual(values[2], - np.log(P[3]) * P[3] - np.log(P[4]) * P[4] - np.log(P[5]) * P[5])

            del tvg

        def test_metric_entropy_edges(self):
            values = [
                {(0, 0): 1.0, (0, 1): 0.0, (1, 1): 2.0, (2, 2): 2.0},
                {(0, 0): 1.0, (0, 1): 1.0, (1, 1): 1.0, (2, 2): 2.0},
                {(0, 0): 1.0, (0, 1): 2.0, (1, 1): 0.0, (2, 2): 2.0},
            ]

            result = metric_entropy(values, num_bins=2)
            self.assertEqual(len(result), 4)
            self.assertTrue(abs(result[0, 0] - 0.89587973) < 1e-7)
            self.assertTrue(abs(result[0, 1] - 0.74918778) < 1e-7)
            self.assertTrue(abs(result[1, 1] - 0.74918778) < 1e-7)
            self.assertTrue(abs(result[2, 2] - 0.45580389) < 1e-7)

        def test_metric_entropy_local(self):
            tvg = TVG(positive=True)

            g = Graph()
            g[0, 0] = g[0, 1] = g[0, 2] = 1.0
            g[1, 1] = g[1, 2] = 1.0
            g[2, 2] = 1.0
            tvg.link_graph(g, 100)

            g = Graph()
            g[1, 1] = g[1, 2] = 2.0
            g[2, 2] = 2.0
            tvg.link_graph(g, 200)

            g = Graph()
            g[2, 2] = 3.0
            tvg.link_graph(g, 300)

            values = tvg.sample_eigenvectors(50, 350, sample_width=101, sample_steps=3)
            values = metric_entropy_local(values, num_bins=2)

            P0 = 1.0
            P1 = np.array([1, 2]) / 3.0
            P2 = np.array([2, 1]) / 3.0
            self.assertEqual(len(values), 3)
            self.assertEqual(values[0], - np.log(P0) * P0 - np.log(P1[0]) * P1[0] - np.log(P2[0]) * P2[0])
            self.assertEqual(values[1], - np.log(P0) * P0 - np.log(P1[1]) * P1[1] - np.log(P2[0]) * P2[0])
            self.assertEqual(values[2], - np.log(P0) * P0 - np.log(P1[1]) * P1[1] - np.log(P2[1]) * P2[1])

            del tvg

        def test_metric_entropy_local_edges(self):
            values = [
                {(0, 0): 1.0, (0, 1): 0.0, (1, 1): 2.0, (2, 2): 2.0},
                {(0, 0): 1.0, (0, 1): 1.0, (1, 1): 1.0, (2, 2): 2.0},
                {(0, 0): 1.0, (0, 1): 2.0, (1, 1): 0.0, (2, 2): 2.0},
            ]

            result = metric_entropy_local(values, num_bins=2)
            self.assertEqual(len(result), 4)
            self.assertTrue(abs(result[0, 0] - 0.90890873) < 1e-7)
            self.assertTrue(abs(result[0, 1] - 0.77809669) < 1e-7)
            self.assertTrue(abs(result[1, 1] - 0.77809669) < 1e-7)
            self.assertTrue(abs(result[2, 2] - 0.77809669) < 1e-7)

        def test_metric_entropy_2d(self):
            tvg = TVG(positive=True)

            g = Graph()
            g[0, 0] = g[0, 1] = g[0, 2] = 1.0
            g[1, 1] = g[1, 2] = 1.0
            g[2, 2] = 1.0
            tvg.link_graph(g, 100)

            g = Graph()
            g[1, 1] = g[1, 2] = 2.0
            g[2, 2] = 2.0
            tvg.link_graph(g, 200)

            g = Graph()
            g[2, 2] = 3.0
            tvg.link_graph(g, 300)

            values = tvg.sample_eigenvectors(50, 350, sample_width=101, sample_steps=3)
            values = metric_entropy_2d(values, num_bins=2)

            P = np.array([[1, 0], [2, 3]]) / 6.0
            self.assertEqual(len(values), 3)
            self.assertEqual(values[0], - np.log(P[1, 0]) * P[1, 0] - np.log(P[0, 0]) * P[0, 0])
            self.assertEqual(values[1], - np.log(P[1, 1]) * P[1, 1] - np.log(P[1, 0]) * P[1, 0])
            self.assertEqual(values[2], - np.log(P[1, 1]) * P[1, 1] - np.log(P[1, 1]) * P[1, 1])

            del tvg

        def test_metric_entropy_2d_edges(self):
            values = [
                {(0, 0): 1.0, (0, 1): 0.0, (1, 1): 2.0, (2, 2): 2.0},
                {(0, 0): 1.0, (0, 1): 1.0, (1, 1): 1.0, (2, 2): 2.0},
                {(0, 0): 1.0, (0, 1): 2.0, (1, 1): 0.0, (2, 2): 2.0},
            ]

            result = metric_entropy_2d(values, num_bins=2)
            self.assertEqual(len(result), 4)
            self.assertTrue(abs(result[0, 0] - 0.0) < 1e-7)
            self.assertTrue(abs(result[0, 1] - 0.25993019) < 1e-7)
            self.assertTrue(abs(result[1, 1] - 0.25993019) < 1e-7)
            self.assertTrue(abs(result[2, 2] - 0.43152310) < 1e-7)

        def test_metric_trend(self):
            tvg = TVG(positive=True)

            g = Graph()
            g[0, 0] = g[0, 1] = g[0, 2] = 1.0
            g[1, 1] = g[1, 2] = 1.0
            g[2, 2] = 1.0
            tvg.link_graph(g, 100)

            g = Graph()
            g[1, 1] = g[1, 2] = 2.0
            g[2, 2] = 2.0
            tvg.link_graph(g, 200)

            g = Graph()
            g[2, 2] = 3.0
            tvg.link_graph(g, 300)

            values = tvg.sample_eigenvectors(50, 350, sample_width=101, sample_steps=3)
            values = metric_trend(values)

            self.assertEqual(len(values), 3)
            self.assertTrue(abs(values[0] + 0.288675129) < 1e-7)
            self.assertTrue(abs(values[1] + 0.288675129) < 1e-7)
            self.assertTrue(abs(values[2] - 0.211324870) < 1e-7)

            del tvg

        def test_metric_trend_edges(self):
            values = [
                {(0, 0): 1.0, (0, 1): 0.0, (1, 1): 2.0, (2, 2): 2.0},
                {(0, 0): 1.0, (0, 1): 1.0, (1, 1): 1.0, (2, 2): 2.0},
                {(0, 0): 1.0, (0, 1): 2.0, (1, 1): 0.0, (2, 2): 2.0},
            ]

            result = metric_trend(values)
            self.assertEqual(len(result), 4)
            self.assertTrue(abs(result[0, 0] - 0.0) < 1e-7)
            self.assertTrue(abs(result[0, 1] - 1.0) < 1e-7)
            self.assertTrue(abs(result[1, 1] + 1.0) < 1e-7)
            self.assertTrue(abs(result[2, 2] - 0.0) < 1e-7)

        def test_metric_stability_ratio_edges(self):
            values = [
                {(0, 0): 1.0, (0, 1): 0.0, (1, 1): 2.0, (2, 2): 2.0, (3, 3): 0.0},
                {(0, 0): 1.0, (0, 1): 1.0, (1, 1): 1.0, (2, 2): 2.0, (3, 3): 0.0},
                {(0, 0): 1.0, (0, 1): 2.0, (1, 1): 0.0, (2, 2): 2.0, (3, 3): 0.0},
            ]

            result = metric_stability_ratio(values)
            self.assertEqual(len(result), 5)
            self.assertEqual(result[0, 0], np.inf)
            self.assertTrue(abs(result[0, 1] - 1.22474487) < 1e-7)
            self.assertTrue(abs(result[1, 1] - 1.22474487) < 1e-7)
            self.assertEqual(result[2, 2], np.inf)
            self.assertEqual(result[3, 3], np.inf)

        def test_metric_stability_pareto(self):
            tvg = TVG(positive=True)

            g = Graph()
            g[0, 0] = g[0, 1] = g[0, 2] = 1.0
            g[1, 1] = g[1, 2] = 1.0
            g[2, 2] = 1.0
            tvg.link_graph(g, 100)

            g = Graph()
            g[1, 1] = g[1, 2] = 2.0
            g[2, 2] = 2.0
            tvg.link_graph(g, 200)

            g = Graph()
            g[2, 2] = 3.0
            tvg.link_graph(g, 300)

            values = tvg.sample_eigenvectors(50, 350, sample_width=101, sample_steps=3)
            values = metric_stability_pareto(values)

            self.assertEqual(len(values), 3)
            self.assertEqual(values[2], 1.0)
            self.assertEqual(values[0], 2.0)
            self.assertEqual(values[1], 2.0)

            del tvg

        def test_metric_stability_pareto_edges(self):
            values = [
                {(0, 0): 1.0, (0, 1): 0.0, (1, 1): 2.0, (2, 2): 2.0},
                {(0, 0): 1.0, (0, 1): 1.0, (1, 1): 1.0, (2, 2): 2.0},
                {(0, 0): 1.0, (0, 1): 2.0, (1, 1): 0.0, (2, 2): 2.0},
            ]

            result = metric_stability_pareto(values)
            self.assertEqual(len(result), 4)
            self.assertEqual(result[0, 0], 2.0)
            self.assertEqual(result[0, 1], 3.0)
            self.assertEqual(result[1, 1], 3.0)
            self.assertEqual(result[2, 2], 1.0)

            # Fast-path for list of Graphs
            graphs = [Graph.from_dict(v) for v in values]
            result = metric_stability_pareto(graphs)
            self.assertTrue(isinstance(result, Graph))
            self.assertEqual(result.as_dict(), {(0, 0): 2.0, (0, 1): 3.0,
                                                (1, 1): 3.0, (2, 2): 1.0})
            del graphs

            values = [
                {0: 1.0, 1: 0.0, 2: 2.0, 3: 2.0},
                {0: 1.0, 1: 1.0, 2: 1.0, 3: 2.0},
                {0: 1.0, 1: 2.0, 2: 0.0, 3: 2.0},
            ]

            # Fast-path for list of Vectors
            vectors = [Vector.from_dict(v) for v in values]
            result = metric_stability_pareto(vectors)
            self.assertTrue(isinstance(result, Vector))
            self.assertEqual(result.as_dict(), {0: 2.0, 1: 3.0, 2: 3.0, 3: 1.0})
            del vectors

        def test_metric_stability_pareto_compare(self):
            values = []
            for i in range(20):
                values.append(dict(enumerate(np.random.random(100))))
            result1 = metric_stability_pareto(values)

            graphs = [Graph.from_dict(dict([((0, i), w) for i, w in v.items()])) for v in values]
            result2 = metric_stability_pareto(graphs)
            self.assertTrue(isinstance(result2, Graph))
            self.assertEqual(result2.num_edges, 100)
            for i in range(100):
                self.assertEqual(result2[0, i], result1[i])
            del result2
            del graphs

            vectors = [Vector.from_dict(v) for v in values]
            result2 = metric_stability_pareto(vectors)
            self.assertTrue(isinstance(result2, Vector))
            self.assertEqual(result2.num_entries, 100)
            for i in range(100):
                self.assertEqual(result2[i], result1[i])
            del result2
            del vectors

        def test_metric_avg(self):
            values = [
                {(0, 0): 1.0, (0, 1): 0.0, (1, 1): 2.0, (2, 2): 2.0},
                {(0, 0): 1.0, (0, 1): 1.0, (1, 1): 1.0, (2, 2): 2.0},
                {(0, 0): 1.0, (0, 1): 2.0, (1, 1): 0.0, (2, 2): 2.0},
            ]

            result = metric_avg(values)
            self.assertEqual(result, {(0, 0): 1.0, (0, 1): 1.0,
                                      (1, 1): 1.0, (2, 2): 2.0})

            # Fast-path for list of Graphs
            graphs = [Graph.from_dict(v) for v in values]
            result = metric_avg(graphs)
            self.assertTrue(isinstance(result, Graph))
            self.assertEqual(result.as_dict(), {(0, 0): 1.0, (0, 1): 1.0,
                                                (1, 1): 1.0, (2, 2): 2.0})
            del graphs

            values = [
                {0: 1.0, 1: 0.0, 2: 2.0, 3: 2.0},
                {0: 1.0, 1: 1.0, 2: 1.0, 3: 2.0},
                {0: 1.0, 1: 2.0, 2: 0.0, 3: 2.0},
            ]

            # Fast-path for list of Vectors
            vectors = [Vector.from_dict(v) for v in values]
            result = metric_avg(vectors)
            self.assertTrue(isinstance(result, Vector))
            self.assertEqual(result.as_dict(), {0: 1.0, 1: 1.0, 2: 1.0, 3: 2.0})
            del vectors

        def test_metric_std(self):
            values = [
                {(0, 0): 1.0, (0, 1): 0.0, (1, 1): 2.0, (2, 2): 2.0},
                {(0, 0): 1.0, (0, 1): 1.0, (1, 1): 1.0, (2, 2): 2.0},
                {(0, 0): 1.0, (0, 1): 2.0, (1, 1): 0.0, (2, 2): 2.0},
            ]

            result = metric_std(values)
            self.assertEqual(result, {(0, 0): 0.0, (0, 1): 1.0,
                                      (1, 1): 1.0, (2, 2): 0.0})

            # Fast-path for list of Graphs
            graphs = [Graph.from_dict(v) for v in values]
            result = metric_std(graphs)
            self.assertTrue(isinstance(result, Graph))
            self.assertEqual(result.as_dict(), {(0, 0): 0.0, (0, 1): 1.0,
                                                (1, 1): 1.0, (2, 2): 0.0})
            del graphs

            values = [
                {0: 1.0, 1: 0.0, 2: 2.0, 3: 2.0},
                {0: 1.0, 1: 1.0, 2: 1.0, 3: 2.0},
                {0: 1.0, 1: 2.0, 2: 0.0, 3: 2.0},
            ]

            # Fast-path for list of Vectors
            vectors = [Vector.from_dict(v) for v in values]
            result = metric_std(vectors)
            self.assertTrue(isinstance(result, Vector))
            self.assertEqual(result.as_dict(), {0: 0.0, 1: 1.0, 2: 1.0, 3: 0.0})
            del vectors

        def test_metric_pareto(self):
            values = [
                {(0, 0): 1.0, (0, 1): 1.0, (1, 1): 1.0, (2, 2): 2.0},
                {(0, 0): 0.0, (0, 1): 1.0, (1, 1): 1.0, (2, 2): 0.0},
            ]

            result = metric_pareto(values, maximize=[True, False])
            self.assertEqual(result, {(0, 0): 2.0, (0, 1): 3.0,
                                      (1, 1): 3.0, (2, 2): 1.0})

            result = metric_pareto(list(reversed(values)), maximize=[False, True])
            self.assertEqual(result, {(0, 0): 2.0, (0, 1): 3.0,
                                      (1, 1): 3.0, (2, 2): 1.0})

            # Fast-path for list of Graphs
            graphs = [Graph.from_dict(v) for v in values]
            result = metric_pareto(graphs, maximize=[True, False])
            self.assertTrue(isinstance(result, Graph))
            self.assertEqual(result.as_dict(), {(0, 0): 2.0, (0, 1): 3.0,
                                                (1, 1): 3.0, (2, 2): 1.0})

            result = metric_pareto(list(reversed(graphs)), maximize=[False, True])
            self.assertTrue(isinstance(result, Graph))
            self.assertEqual(result.as_dict(), {(0, 0): 2.0, (0, 1): 3.0,
                                                (1, 1): 3.0, (2, 2): 1.0})
            del graphs

            values = [
                {0: 1.0, 1: 1.0, 2: 1.0, 3: 2.0},
                {0: 0.0, 1: 1.0, 2: 1.0, 3: 0.0},
            ]

            # Fast-path for list of Vectors
            vectors = [Vector.from_dict(v) for v in values]
            result = metric_pareto(vectors, maximize=[True, False])
            self.assertTrue(isinstance(result, Vector))
            self.assertEqual(result.as_dict(), {0: 2.0, 1: 3.0, 2: 3.0, 3: 1.0})

            result = metric_pareto(list(reversed(vectors)), maximize=[False, True])
            self.assertTrue(isinstance(result, Vector))
            self.assertEqual(result.as_dict(), {0: 2.0, 1: 3.0, 2: 3.0, 3: 1.0})
            del vectors

    class MongoDBTests(unittest.TestCase):
        def MongoDB(self, *args, **kwargs):
            future = mockupdb.go(MongoDB, *args, **kwargs)

            request = self.s.receives("isMaster")
            request.replies({'ok': 1, 'maxWireVersion': 5})

            request = self.s.receives("ping")
            request.replies({'ok': 1})

            return future()

        def setUp(self):
            self.s = mockupdb.MockupDB()
            self.s.run()

            self.db = self.MongoDB(self.s.uri, "database", "col_articles",
                                   "_id", "time", "col_entities", "doc", "sen", "ent",
                                   use_pool=False, max_distance=5, filter_key="fkey",
                                   filter_value="fvalue")

        def tearDown(self):
            self.s.stop()

        def load_from_occurrences(self, occurrences):
            future = mockupdb.go(Graph.load_from_mongodb, self.db, 1337)

            request = self.s.receives()
            self.assertEqual(request["find"], "col_entities")
            self.assertEqual(request["filter"], {'doc': 1337})
            self.assertEqual(request["sort"], {'sen': 1})
            request.replies({'cursor': {'id': 0, 'firstBatch': occurrences}})

            return future()

        def test_invalid(self):
            with self.assertRaises(MemoryError):
                MongoDB("http://localhost", "database", "col_articles",
                        "_id", "time", "col_entities", "doc", "sen", "ent")

        def test_selfloop(self):
            occurrences = []
            for i in range(10):
                occurrences.append({'sen': i, 'ent': 1})
                occurrences.append({'sen': i, 'ent': 1})
            g = self.load_from_occurrences(occurrences)
            self.assertEqual(g.num_edges, 0)

        def test_max_distance(self):
            for i in range(10):
                occurrences = [{'sen': 1,     'ent': 1},
                               {'sen': 1 + i, 'ent': 2},
                               {              'ent': 1}, # no sen
                               {'sen': 1              }] # no ent
                g = self.load_from_occurrences(occurrences)
                if i <= 5:
                    self.assertEqual(g.num_edges, 1)
                    self.assertTrue(abs(g[1, 2]/math.exp(-i) - 1.0) < 1e-7)
                else:
                    self.assertEqual(g.num_edges, 0)

        def test_weight_sum(self):
            for i in range(10):
                occurrences = [{'sen': 1,     'ent': 1    },
                               {'sen': 1 + i, 'ent': 2    },
                               {              'ent': 1    }, # no sen
                               {'sen': 1                  }] # no ent
                g = self.load_from_occurrences(occurrences)
                if i <= 5:
                    self.assertEqual(g.num_edges, 1)
                    self.assertTrue(abs(g[1, 2]/math.exp(-i) - 1.0) < 1e-7)
                else:
                    self.assertEqual(g.num_edges, 0)

        def test_load(self):
            future = mockupdb.go(TVG.load, self.db, primary_key="dummy")

            request = self.s.receives()
            self.assertEqual(request["find"], "col_articles")
            self.assertEqual(request["filter"], {'fkey': 'fvalue'})
            self.assertEqual(request["sort"], collections.OrderedDict([('time', 1), ('_id', 1)]))
            documents = [{'_id': 10, 'time': datetime.datetime.utcfromtimestamp(1546300800)},
                         {'_id': 11, 'time': datetime.datetime.utcfromtimestamp(1546387200)},
                         {'_id': 12, 'time': datetime.datetime.utcfromtimestamp(1546473600)},
                         {           'time': datetime.datetime.utcfromtimestamp(1546560000)}, # no id
                         {'_id': 14                                                        }] # no time
            request.replies({'cursor': {'id': 0, 'firstBatch': documents}})

            for i in range(3):
                request = self.s.receives()
                self.assertEqual(request["find"], "col_entities")
                self.assertEqual(request["filter"], {'doc': 10 + i})
                self.assertEqual(request["sort"], {'sen': 1})
                occurrences = [{'sen': 1, 'ent': 1}, {'sen': 1, 'ent': 2 + i}]
                request.replies({'cursor': {'id': 0, 'firstBatch': occurrences}})

            tvg = future()
            for i, g in enumerate(tvg):
                self.assertEqual(g.revision, 0)
                self.assertEqual(g.ts, 1546300800000 + i * 86400000)
                self.assertEqual(g.id, 10 + i)
                self.assertEqual(g[1, 2 + i], 1.0)
            del tvg

        def test_sync(self):
            tvg = TVG()
            tvg.enable_mongodb_sync(self.db, batch_size=2, cache_size=0x8000) # 32 kB cache
            tvg.verbosity = True
            self.assertEqual(tvg.verbosity, True)

            future = mockupdb.go(tvg.lookup_ge, 0)

            request = self.s.receives()
            self.assertEqual(request["find"], "col_articles")
            self.assertEqual(request["filter"], {'time': {'$gte': datetime.datetime.utcfromtimestamp(0)},
                                                 'fkey': 'fvalue'})
            self.assertEqual(request["sort"], collections.OrderedDict([('time', 1), ('_id', 1)]))
            self.assertEqual(request["limit"], 2)
            documents = [{'_id': 10, 'time': datetime.datetime.utcfromtimestamp(1546300800)},
                         {'_id': 11, 'time': datetime.datetime.utcfromtimestamp(1546387200)}]
            request.replies({'cursor': {'id': 0, 'firstBatch': documents}})

            for i in range(2):
                request = self.s.receives()
                self.assertEqual(request["find"], "col_entities")
                self.assertEqual(request["filter"], {'doc': 10 + i})
                self.assertEqual(request["sort"], {'sen': 1})
                occurrences = [{'sen': 1, 'ent': 1}, {'sen': 1, 'ent': 2 + i}]
                request.replies({'cursor': {'id': 0, 'firstBatch': occurrences}})

            g = future()
            self.assertEqual(g.revision, 0)
            self.assertEqual(g.flags, 0)
            self.assertEqual(g.ts, 1546300800000)
            self.assertEqual(g.id, 10)
            self.assertEqual(g[1, 2], 1.0)

            g = g.next
            self.assertEqual(g.revision, 0)
            self.assertEqual(g.flags, TVG_FLAGS_LOAD_NEXT)
            self.assertEqual(g.ts, 1546387200000)
            self.assertEqual(g.id, 11)
            self.assertEqual(g[1, 3], 1.0)

            future = mockupdb.go(getattr, g, 'next')

            request = self.s.receives()
            self.assertEqual(request["find"], "col_articles")
            self.assertEqual(request["filter"], {"$or": [{"time": {"$gt": datetime.datetime.utcfromtimestamp(1546387200)}},
                                                         {"time": datetime.datetime.utcfromtimestamp(1546387200), "_id": {"$gt": 11}}],
                                                 'fkey': 'fvalue'})
            self.assertEqual(request["sort"], collections.OrderedDict([('time', 1), ('_id', 1)]))
            self.assertEqual(request["limit"], 2)
            documents = [{'_id': 12, 'time': datetime.datetime.utcfromtimestamp(1546473600)},
                         {'_id': 13, 'time': datetime.datetime.utcfromtimestamp(1546560000)}]
            request.replies({'cursor': {'id': 0, 'firstBatch': documents}})

            for i in range(2):
                request = self.s.receives()
                self.assertEqual(request["find"], "col_entities")
                self.assertEqual(request["filter"], {'doc': 12 + i})
                self.assertEqual(request["sort"], {'sen': 1})
                occurrences = [{'sen': 1, 'ent': 1}, {'sen': 1, 'ent': 4 + i}]
                request.replies({'cursor': {'id': 0, 'firstBatch': occurrences}})

            g = future()
            self.assertEqual(g.revision, 0)
            self.assertEqual(g.flags, 0)
            self.assertEqual(g.ts, 1546473600000)
            self.assertEqual(g.id, 12)
            self.assertEqual(g[1, 4], 1.0)

            g = g.next
            self.assertEqual(g.revision, 0)
            self.assertEqual(g.flags, TVG_FLAGS_LOAD_NEXT)
            self.assertEqual(g.ts, 1546560000000)
            self.assertEqual(g.id, 13)
            self.assertEqual(g[1, 5], 1.0)

            future = mockupdb.go(tvg.lookup_le, 1546732800000)

            request = self.s.receives()
            self.assertEqual(request["find"], "col_articles")
            self.assertEqual(request["filter"], {'time': {'$lte': datetime.datetime.utcfromtimestamp(1546732800)},
                                                 'fkey': 'fvalue'})
            self.assertEqual(request["sort"], collections.OrderedDict([('time', -1), ('_id', -1)]))
            self.assertEqual(request["limit"], 2)
            documents = [{'_id': 15, 'time': datetime.datetime.utcfromtimestamp(1546732800)},
                         {'_id': 14, 'time': datetime.datetime.utcfromtimestamp(1546646400)}]
            request.replies({'cursor': {'id': 0, 'firstBatch': documents}})

            for i in range(2):
                request = self.s.receives()
                self.assertEqual(request["find"], "col_entities")
                self.assertEqual(request["filter"], {'doc': 15 - i})
                self.assertEqual(request["sort"], {'sen': 1})
                occurrences = [{'sen': 1, 'ent': 1}, {'sen': 1, 'ent': 7 - i}]
                request.replies({'cursor': {'id': 0, 'firstBatch': occurrences}})

            g = future()
            self.assertEqual(g.revision, 0)
            self.assertEqual(g.flags, TVG_FLAGS_LOAD_NEXT)
            self.assertEqual(g.ts, 1546732800000)
            self.assertEqual(g.id, 15)
            self.assertEqual(g[1, 7], 1.0)

            g = g.prev
            self.assertEqual(g.revision, 0)
            self.assertEqual(g.flags, TVG_FLAGS_LOAD_PREV)
            self.assertEqual(g.ts, 1546646400000)
            self.assertEqual(g.id, 14)
            self.assertEqual(g[1, 6], 1.0)

            future = mockupdb.go(getattr, g, 'prev')

            request = self.s.receives()
            self.assertEqual(request["find"], "col_articles")
            self.assertEqual(request["filter"], {"$or": [{"time": {"$lt": datetime.datetime.utcfromtimestamp(1546646400)}},
                                                         {"time": datetime.datetime.utcfromtimestamp(1546646400), "_id": {"$lt": 14}}],
                                                 'fkey': 'fvalue'})
            self.assertEqual(request["sort"], collections.OrderedDict([('time', -1), ('_id', -1)]))
            self.assertEqual(request["limit"], 2)
            documents = [{'_id': 13, 'time': datetime.datetime.utcfromtimestamp(1546560000)},
                         {'_id': 12, 'time': datetime.datetime.utcfromtimestamp(1546473600)}]
            request.replies({'cursor': {'id': 0, 'firstBatch': documents}})

            g = future()
            self.assertEqual(g.revision, 0)
            self.assertEqual(g.flags, 0)
            self.assertEqual(g.ts, 1546560000000)
            self.assertEqual(g.id, 13)
            self.assertEqual(g[1, 5], 1.0)

            g = tvg.lookup_ge(1546732800000)
            self.assertEqual(g.revision, 0)
            self.assertEqual(g.flags, TVG_FLAGS_LOAD_NEXT)
            self.assertEqual(g.ts, 1546732800000)
            self.assertEqual(g.id, 15)
            self.assertEqual(g[1, 7], 1.0)

            future = mockupdb.go(getattr, g, 'next')

            request = self.s.receives()
            self.assertEqual(request["find"], "col_articles")
            self.assertEqual(request["filter"], {"$or": [{"time": {"$gt": datetime.datetime.utcfromtimestamp(1546732800)}},
                                                         {"time": datetime.datetime.utcfromtimestamp(1546732800), "_id": {"$gt": 15}}],
                                                 'fkey': 'fvalue'})
            self.assertEqual(request["sort"], collections.OrderedDict([('time', 1), ('_id', 1)]))
            self.assertEqual(request["limit"], 2)
            documents = []
            request.replies({'cursor': {'id': 0, 'firstBatch': documents}})

            g = future()
            self.assertEqual(g, None)

            for i, g in enumerate(tvg):
                self.assertEqual(g.revision, 0)
                self.assertEqual(g.flags, 0)
                self.assertEqual(g.ts, 1546300800000 + i * 86400000)
                self.assertEqual(g.id, 10 + i)
                self.assertEqual(g[1, 2 + i], 1.0)

            tvg.disable_mongodb_sync()
            del tvg

        def test_streaming(self):
            tvg = TVG(streaming=True)
            tvg.enable_mongodb_sync(self.db, batch_size=2, cache_size=0x8000) # 32 kB cache

            future = mockupdb.go(tvg.lookup_le)

            request = self.s.receives()
            self.assertEqual(request["find"], "col_articles")
            self.assertEqual(request["filter"], {'fkey': 'fvalue'})
            self.assertEqual(request["sort"], collections.OrderedDict([('time', -1), ('_id', -1)]))
            self.assertEqual(request["limit"], 2)
            documents = [{'_id': 11, 'time': datetime.datetime.utcfromtimestamp(1546387200)},
                         {'_id': 10, 'time': datetime.datetime.utcfromtimestamp(1546300800)}]
            request.replies({'cursor': {'id': 0, 'firstBatch': documents}})

            for i in range(2):
                request = self.s.receives()
                self.assertEqual(request["find"], "col_entities")
                self.assertEqual(request["filter"], {'doc': 11 - i})
                self.assertEqual(request["sort"], {'sen': 1})
                occurrences = [{'sen': 1, 'ent': 1}, {'sen': 1, 'ent': 3 - i}]
                request.replies({'cursor': {'id': 0, 'firstBatch': occurrences}})

            g = future()
            self.assertEqual(g.revision, 0)
            self.assertEqual(g.flags, TVG_FLAGS_LOAD_NEXT)
            self.assertEqual(g.ts, 1546387200000)
            self.assertEqual(g[1, 3], 1.0)

            g = g.prev
            self.assertEqual(g.revision, 0)
            self.assertEqual(g.flags, TVG_FLAGS_LOAD_PREV)
            self.assertEqual(g.ts, 1546300800000)
            self.assertEqual(g[1, 2], 1.0)

            future = mockupdb.go(tvg.lookup_le)

            request = self.s.receives()
            self.assertEqual(request["find"], "col_articles")
            self.assertEqual(request["filter"], {'fkey': 'fvalue'})
            self.assertEqual(request["sort"], collections.OrderedDict([('time', -1), ('_id', -1)]))
            self.assertEqual(request["limit"], 2)
            documents = [{'_id': 12, 'time': datetime.datetime.utcfromtimestamp(1546473600)},
                         {'_id': 11, 'time': datetime.datetime.utcfromtimestamp(1546387200)}]
            request.replies({'cursor': {'id': 0, 'firstBatch': documents}})

            request = self.s.receives()
            self.assertEqual(request["find"], "col_entities")
            self.assertEqual(request["filter"], {'doc': 12})
            self.assertEqual(request["sort"], {'sen': 1})
            occurrences = [{'sen': 1, 'ent': 1}, {'sen': 1, 'ent': 4}]
            request.replies({'cursor': {'id': 0, 'firstBatch': occurrences}})

            g = future()
            self.assertEqual(g.revision, 0)
            self.assertEqual(g.flags, TVG_FLAGS_LOAD_NEXT)
            self.assertEqual(g.ts, 1546473600000)
            self.assertEqual(g[1, 4], 1.0)

            g2 = g.prev
            self.assertEqual(g2.revision, 0)
            self.assertEqual(g2.flags, 0)
            self.assertEqual(g2.ts, 1546387200000)
            self.assertEqual(g2[1, 3], 1.0)

            future = mockupdb.go(getattr, g, 'next')

            request = self.s.receives()
            self.assertEqual(request["find"], "col_articles")
            self.assertEqual(request["filter"], {"$or": [{"time": {"$gt": datetime.datetime.utcfromtimestamp(1546473600)}},
                                                         {"time": datetime.datetime.utcfromtimestamp(1546473600), "_id": {"$gt": 12}}],
                                                 'fkey': 'fvalue'})
            self.assertEqual(request["sort"], collections.OrderedDict([('time', 1), ('_id', 1)]))
            self.assertEqual(request["limit"], 2)
            documents = []
            request.replies({'cursor': {'id': 0, 'firstBatch': documents}})

            g2 = future()
            self.assertEqual(g2, None)

            self.assertEqual(g.revision, 0)
            self.assertEqual(g.flags, TVG_FLAGS_LOAD_NEXT)
            self.assertEqual(g.ts, 1546473600000)
            self.assertEqual(g[1, 4], 1.0)

            tvg.disable_mongodb_sync()
            del tvg

        def test_objectid(self):
            self.db = self.MongoDB(self.s.uri, "database", "col_articles",
                                   "_id", "time", "col_entities", "doc", "sen", "ent",
                                   use_pool=False, max_distance=5)

            future = mockupdb.go(TVG.load, self.db, primary_key="dummy")

            request = self.s.receives()
            self.assertEqual(request["find"], "col_articles")
            self.assertEqual(request["filter"], {})
            self.assertEqual(request["sort"], collections.OrderedDict([('time', 1), ('_id', 1)]))
            documents = [{'_id': bson.ObjectId('123456781234567812345678'),
                          'time': datetime.datetime.utcfromtimestamp(1546300800)},
                         {'_id': bson.ObjectId('123456781234567812345679'),
                          'time': datetime.datetime.utcfromtimestamp(1546387200)},
                         {'_id': bson.ObjectId('12345678123456781234567a'),
                          'time': datetime.datetime.utcfromtimestamp(1546473600)},
                         {'time': datetime.datetime.utcfromtimestamp(1546560000)}, # no id
                         {'_id': bson.ObjectId('12345678123456781234567c')}]       # no time
            request.replies({'cursor': {'id': 0, 'firstBatch': documents}})

            for i in range(3):
                request = self.s.receives()
                self.assertEqual(request["find"], "col_entities")
                self.assertEqual(request["filter"], {'doc': [bson.ObjectId('123456781234567812345678'),
                                                             bson.ObjectId('123456781234567812345679'),
                                                             bson.ObjectId('12345678123456781234567a')][i]})
                self.assertEqual(request["sort"], {'sen': 1})
                occurrences = [{'sen': 1, 'ent': 1}, {'sen': 1, 'ent': 2 + i}]
                request.replies({'cursor': {'id': 0, 'firstBatch': occurrences}})

            tvg = future()
            for i, g in enumerate(tvg):
                self.assertEqual(g.revision, 0)
                self.assertEqual(g.ts, 1546300800000 + i * 86400000)
                self.assertEqual(g.id, ['123456781234567812345678',
                                        '123456781234567812345679',
                                        '12345678123456781234567a'][i])
                self.assertEqual(g[1, 2 + i], 1.0)
            del tvg

            future = mockupdb.go(Graph.load_from_mongodb, self.db, '112233445566778899aabbcc')

            request = self.s.receives()
            self.assertEqual(request["find"], "col_entities")
            self.assertEqual(request["filter"], {'doc': bson.ObjectId('112233445566778899aabbcc')})
            self.assertEqual(request["sort"], {'sen': 1})
            request.replies({'cursor': {'id': 0, 'firstBatch': []}})

            g = future()
            self.assertTrue(isinstance(g, Graph))
            del g

            future = mockupdb.go(Graph.load_from_mongodb, self.db, b'\x11\x22\x33\x44\x55\x66\x77\x88\x99\xaa\xbb\xcc')

            request = self.s.receives()
            self.assertEqual(request["find"], "col_entities")
            self.assertEqual(request["filter"], {'doc': bson.ObjectId('112233445566778899aabbcc')})
            self.assertEqual(request["sort"], {'sen': 1})
            request.replies({'cursor': {'id': 0, 'firstBatch': []}})

            g = future()
            self.assertTrue(isinstance(g, Graph))
            del g

        def test_max_distance(self):
            occurrences = [{'sen': 0, 'ent': 1 },
                           {'sen': 0x7fffffffffffffff, 'ent': 2 }]

            g = self.load_from_occurrences(occurrences)
            self.assertFalse(g.has_edge((1, 2)))
            del g

            self.db = self.MongoDB(self.s.uri, "database", "col_articles",
                                   "_id", "time", "col_entities", "doc", "sen", "ent",
                                   use_pool=False, max_distance=None)

            g = self.load_from_occurrences(occurrences)
            self.assertTrue(g.has_edge((1, 2)))
            self.assertEqual(g[1, 2], 0.0)
            del g

        def test_sum_weights(self):
            occurrences1 = [{'sen': 1, 'ent': 1},
                            {'sen': 2, 'ent': 2},
                            {'sen': 4, 'ent': 1}]

            occurrences2 = [{'sen': 1, 'ent': 1},
                            {'sen': 3, 'ent': 2},
                            {'sen': 4, 'ent': 1}]

            g = self.load_from_occurrences(occurrences1)
            self.assertTrue(abs(g[1, 2]/(math.exp(-1.0) + np.exp(-2.0)) - 1.0) < 1e-7)
            del g

            g = self.load_from_occurrences(occurrences1)
            self.assertTrue(abs(g[1, 2]/(math.exp(-1.0) + np.exp(-2.0)) - 1.0) < 1e-7)
            del g

            self.db = self.MongoDB(self.s.uri, "database", "col_articles",
                                   "_id", "time", "col_entities", "doc", "sen", "ent",
                                   use_pool=False, sum_weights=False)

            g = self.load_from_occurrences(occurrences1)
            self.assertTrue(abs(g[1, 2]/math.exp(-1.0) - 1.0) < 1e-7)
            del g

            g = self.load_from_occurrences(occurrences2)
            self.assertTrue(abs(g[1, 2]/math.exp(-1.0) - 1.0) < 1e-7)
            del g

            self.db = self.MongoDB(self.s.uri, "database", "col_articles",
                                   "_id", "time", "col_entities", "doc", "sen", "ent",
                                   use_pool=False, norm_weights=True)

            g = self.load_from_occurrences(occurrences1)
            self.assertEqual(g[1, 2], 1.0)
            del g

            g = self.load_from_occurrences(occurrences2)
            self.assertEqual(g[1, 2], 1.0)
            del g

        def test_readonly(self):
            future = mockupdb.go(TVG.load, self.db, primary_key="dummy")

            request = self.s.receives()
            self.assertEqual(request["find"], "col_articles")
            self.assertEqual(request["filter"], {'fkey': 'fvalue'})
            self.assertEqual(request["sort"], collections.OrderedDict([('time', 1), ('_id', 1)]))
            documents = [{'_id': 10, 'time': datetime.datetime.utcfromtimestamp(1546387200)}]
            request.replies({'cursor': {'id': 0, 'firstBatch': documents}})

            request = self.s.receives()
            self.assertEqual(request["find"], "col_entities")
            self.assertEqual(request["filter"], {'doc': 10})
            self.assertEqual(request["sort"], {'sen': 1})
            occurrences = [{'sen': 1, 'ent': 1}, {'sen': 1, 'ent': 2}]
            request.replies({'cursor': {'id': 0, 'firstBatch': occurrences}})

            tvg = future()
            g = tvg.lookup_ge()
            self.assertEqual(g.ts, 1546387200000)

            with self.assertRaises(RuntimeError):
                g.clear()
            with self.assertRaises(MemoryError):
                g[0, 0] = 1.0
            with self.assertRaises(MemoryError):
                g.add_edge((0, 0), 1.0)
            with self.assertRaises(RuntimeError):
                del g[0, 0]
            with self.assertRaises(RuntimeError):
                g.mul_const(2.0)
            with self.assertRaises(RuntimeError):
                g.eps = 2.0

            del tvg
            del g

            tvg = TVG(streaming=True)
            tvg.enable_mongodb_sync(self.db, batch_size=1, cache_size=0x8000) # 32 kB cache

            future = mockupdb.go(tvg.lookup_le)

            request = self.s.receives()
            self.assertEqual(request["find"], "col_articles")
            self.assertEqual(request["filter"], {'fkey': 'fvalue'})
            self.assertEqual(request["sort"], collections.OrderedDict([('time', -1), ('_id', -1)]))
            self.assertEqual(request["limit"], 1)
            documents = [{'_id': 10, 'time': datetime.datetime.utcfromtimestamp(1546387200)}]
            request.replies({'cursor': {'id': 0, 'firstBatch': documents}})

            request = self.s.receives()
            self.assertEqual(request["find"], "col_entities")
            self.assertEqual(request["filter"], {'doc': 10})
            self.assertEqual(request["sort"], {'sen': 1})
            occurrences = [{'sen': 1, 'ent': 1}, {'sen': 1, 'ent': 2}]
            request.replies({'cursor': {'id': 0, 'firstBatch': occurrences}})

            g = future()
            self.assertEqual(g.ts, 1546387200000)

            with self.assertRaises(RuntimeError):
                g.clear()
            with self.assertRaises(MemoryError):
                g[0, 0] = 1.0
            with self.assertRaises(MemoryError):
                g.add_edge((0, 0), 1.0)
            with self.assertRaises(RuntimeError):
                del g[0, 0]
            with self.assertRaises(RuntimeError):
                g.mul_const(2.0)
            with self.assertRaises(RuntimeError):
                g.eps = 2.0

            del g

    # Run the unit tests
    unittest.main()
    gc.collect()
