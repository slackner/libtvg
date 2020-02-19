#!/usr/bin/env python3
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
import traceback
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

TVG_FLAGS_POSITIVE  = 0x00000002
TVG_FLAGS_DIRECTED  = 0x00000004
TVG_FLAGS_STREAMING = 0x00000008

TVG_FLAGS_LOAD_NEXT = 0x00010000
TVG_FLAGS_LOAD_PREV = 0x00020000
TVG_FLAGS_READONLY  = 0x00040000

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
                ("revision", c_uint64)]

class c_graph(Structure):
    _fields_ = [("refcount", c_uint64),
                ("flags",    c_uint),
                ("revision", c_uint64),
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

class c_snapshot_entry(Structure):
    _fields_ = [("ts_min",   c_uint64),
                ("ts_max",   c_uint64)]

# Hacky: we need optional ndpointer parameters at some places.
def or_null(klass):
    class wrapper:
        @classmethod
        def from_param(cls, obj):
            if obj is None: return None
            return klass.from_param(obj)
    return wrapper

class c_uint64_ts:
    @classmethod
    def from_param(cls, obj):
        if obj < 0:
            obj = 0
        elif obj > 0xffffffffffffffff:
            obj = 0xffffffffffffffff
        return c_uint64.from_param(obj)

c_double_p       = POINTER(c_double)
c_objectid_p     = POINTER(c_objectid)
c_vector_p       = POINTER(c_vector)
c_graph_p        = POINTER(c_graph)
c_node_p         = POINTER(c_node)
c_tvg_p          = POINTER(c_tvg)
c_mongodb_config_p = POINTER(c_mongodb_config)
c_mongodb_p      = POINTER(c_mongodb)
c_snapshot_entry_p = POINTER(c_snapshot_entry)
c_bfs_entry_p    = POINTER(c_bfs_entry)
c_bfs_callback_p = CFUNCTYPE(c_int, c_graph_p, c_bfs_entry_p, c_void_p)
c_snapshot_callback_p = CFUNCTYPE(c_int, c_uint64, c_snapshot_entry_p, c_void_p)

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

lib.vector_del_small.argtypes = (c_vector_p, c_float)
lib.vector_del_small.restype = c_int

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

lib.vector_save_binary.argtypes = (c_vector_p, c_char_p)
lib.vector_save_binary.restype = c_int

lib.vector_load_binary.argtypes = (c_char_p,)
lib.vector_load_binary.restype = c_vector_p

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

lib.graph_del_small.argtypes = (c_graph_p, c_float)
lib.graph_del_small.restype = c_int

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

lib.graph_get_nodes.argtypes = (c_graph_p,)
lib.graph_get_nodes.restype = c_vector_p

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

lib.graph_save_binary.argtypes = (c_graph_p, c_char_p)
lib.graph_save_binary.restype = c_int

lib.graph_load_binary.argtypes = (c_char_p,)
lib.graph_load_binary.restype = c_graph_p

lib.graph_bfs.argtypes = (c_graph_p, c_uint64, c_int, c_bfs_callback_p, c_void_p)
lib.graph_bfs.restype = c_int

lib.graph_get_distance_count.argtypes = (c_graph_p, c_uint64, c_uint64)
lib.graph_get_distance_count.restype = c_uint64

lib.graph_get_distance_weight.argtypes = (c_graph_p, c_uint64, c_uint64)
lib.graph_get_distance_weight.restype = c_double

lib.graph_get_all_distances_count.argtypes = (c_graph_p, c_uint64, c_uint64)
lib.graph_get_all_distances_count.restype = c_vector_p

lib.graph_get_all_distances_weight.argtypes = (c_graph_p, c_uint64, c_double)
lib.graph_get_all_distances_weight.restype = c_vector_p

lib.graph_get_all_distances_graph.argtypes = (c_graph_p, c_int)
lib.graph_get_all_distances_graph.restype = c_graph_p

lib.graph_get_connected_components.argtypes = (c_graph_p,)
lib.graph_get_connected_components.restype = c_vector_p

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

lib.tvg_lookup_graph_ge.argtypes = (c_tvg_p, c_uint64_ts)
lib.tvg_lookup_graph_ge.restype = c_graph_p

lib.tvg_lookup_graph_le.argtypes = (c_tvg_p, c_uint64_ts)
lib.tvg_lookup_graph_le.restype = c_graph_p

lib.tvg_lookup_graph_near.argtypes = (c_tvg_p, c_uint64_ts)
lib.tvg_lookup_graph_near.restype = c_graph_p

lib.tvg_compress.argtypes = (c_tvg_p, c_uint64_ts, c_uint64_ts, c_snapshot_callback_p, c_void_p)
lib.tvg_compress.restype = c_int

# Query functions

lib.tvg_sum_edges.argtypes = (c_tvg_p, c_uint64_ts, c_uint64_ts, c_float)
lib.tvg_sum_edges.restype = c_graph_p

lib.tvg_sum_nodes.argtypes = (c_tvg_p, c_uint64_ts, c_uint64_ts)
lib.tvg_sum_nodes.restype = c_vector_p

lib.tvg_sum_edges_exp.argtypes = (c_tvg_p, c_uint64_ts, c_uint64_ts, c_float, c_float, c_float)
lib.tvg_sum_edges_exp.restype = c_graph_p

lib.tvg_count_edges.argtypes = (c_tvg_p, c_uint64_ts, c_uint64_ts)
lib.tvg_count_edges.restype = c_graph_p

lib.tvg_count_nodes.argtypes = (c_tvg_p, c_uint64_ts, c_uint64_ts)
lib.tvg_count_nodes.restype = c_vector_p

lib.tvg_count_graphs.argtypes = (c_tvg_p, c_uint64_ts, c_uint64_ts)
lib.tvg_count_graphs.restype = c_uint64

lib.tvg_topics.argtypes = (c_tvg_p, c_uint64_ts, c_uint64_ts, or_null(c_snapshot_callback_p), c_void_p)
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

class memoized(object):
    def __init__(self, func, typed=True):
        self._func  = func
        self._typed = typed
        self._cache = {}

    def __call__(self, *args, **kwargs):
        key = functools._make_key(args, kwargs, self._typed)
        try:
            return self._cache[key]
        except KeyError:
            value = self._func(*args, **kwargs)
            self._cache[key] = value
            return value

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

class UniformSamples(object):
    def __init__(self, step, offset=0):
        if step > 0 and np.isinf(step):
            step = 0
        if step != 0:
            offset = offset - (offset // step) * step

        self.step = step
        self.offset = offset

    def __call__(self, ts):
        if self.step == 0:
            return (0, 0xffffffffffffffff)
        elif ts < self.offset:
            return (0, self.offset - 1)
        else:
            ts -= (ts - self.offset) % self.step
            return (ts, ts + self.step - 1)

@libtvgobject
class Vector(object):
    """
    This object represents a vector of arbitrary / infinite dimension. To achieve that,
    it only stores entries that are explicitly set, and assumes that all other entries
    of the vector are zero. Internally, it uses hashing to map indices to buckets,
    that are stored in contiguous blocks of memory and in sorted order for faster access.

    # Arguments
    positive: Enforce that all entries must be positive.
    """

    def __init__(self, positive=False, obj=None):
        if obj is None:
            flags = 0
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
    def readonly(self):
        return (self._obj.contents.flags & TVG_FLAGS_READONLY) != 0

    @property
    def revision(self):
        """
        Return the current revision of the vector object. This value is incremented
        whenever the vector is changed. It is also used by the @cacheable decorator
        to check the cache validity.
        """
        return self._obj.contents.revision

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

        if indices is not None and num_entries < max_entries:
            indices.resize((num_entries,), refcheck=False)
        if weights is not None and num_entries < max_entries:
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

    def __iter__(self):
        """ Iterate over indices of a vector. """
        return self.keys()

    def tolist(self):
        """ Return list of indices of a vector. """
        return list(self.keys())

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

    def del_small(self, eps=0.0):
        """ Drop entries smaller than the selected `eps`. """
        res = lib.vector_del_small(self._obj, eps)
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

    def save_binary(self, filename):
        """
        Store a vector in a file using binary format.

        # Arguments
        filename: Path to the file to create
        """

        res = lib.vector_save_binary(self._obj, filename.encode("utf-8"))
        if not res:
            raise IOError

    @staticmethod
    def load_binary(filename):
        """
        Load a vector from a binary file into memory.

        # Arguments
        filename: Path to the file to load
        """

        obj = lib.vector_load_binary(filename.encode("utf-8"))
        if not obj:
            raise IOError
        return Vector(obj=obj)

@libtvgobject
class Graph(object):
    """
    This object represents a graph of arbitrary / infinite dimension. To achieve that,
    it only stores edges that are explicitly set, and assumes that all other edges
    of the graph have a weight of zero. Internally, it uses hashing to map source and
    target indices to buckets, that are stored in contiguous blocks of memory and in
    sorted order for faster access.

    # Arguments
    positive: Enforce that all entries must be positive.
    directed: Create a directed graph.
    """

    def __init__(self, positive=False, directed=False, obj=None):
        if obj is None:
            flags = 0
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
    def readonly(self):
        return (self._obj.contents.flags & TVG_FLAGS_READONLY) != 0

    @property
    def revision(self):
        """
        Return the current revision of the graph object. This value is incremented
        whenever the graph is changed. It is also used by the @cacheable decorator
        to check the cache validity.
        """
        return self._obj.contents.revision

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
    def load_from_file(filename, positive=False, directed=False):
        raise NotImplementedError

    @staticmethod
    def load_from_mongodb(mongodb, id, positive=False, directed=False):
        """
        Load a single graph from a MongoDB database.

        # Arguments
        id: Identifier (numeric or objectid) of the document to load
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

        num_edges = lib.graph_get_edges(self._obj, None, None, 0)
        while True:
            max_edges = num_edges
            indices = np.empty(shape=(max_edges, 2), dtype=np.uint64,  order='C') if ret_indices else None
            weights = np.empty(shape=(max_edges,),   dtype=np.float32, order='C') if ret_weights else None
            num_edges = lib.graph_get_edges(self._obj, indices, weights, max_edges)
            if num_edges <= max_edges:
                break

        if indices is not None and num_edges < max_edges:
            indices.resize((num_edges, 2), refcheck=False)
        if weights is not None and num_edges < max_edges:
            weights.resize((num_edges,), refcheck=False)

        if as_dict:
            if weights is None:
                weights = [None] * num_edges
            return dict((tuple(i), w) for i, w in zip(indices, weights))

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

    def __iter__(self):
        """ Iterate over indices of a graph. """
        return self.keys()

    def top_edges(self, max_edges, ret_indices=True, ret_weights=True, as_dict=False,
                  truncate=False):
        """
        Return indices and/or weights of the top edges.

        # Arguments
        num_edges: Limit the number of edges returned.
        ret_indices: Return indices consisting of (source, target), otherwise None.
        ret_weights: Return weights, otherwise None.
        as_dict: Return result as dictionary instead of tuple.
        truncate: Truncate list of results if too many.

        # Returns
        `(indices, weights)` or dictionary
        """

        if as_dict and not ret_indices:
            raise ValueError("Invalid parameter combination")

        num_edges = max_edges
        while True:
            max_edges = num_edges
            indices = np.empty(shape=(max_edges, 2), dtype=np.uint64,  order='C') if ret_indices else None
            weights = np.empty(shape=(max_edges,),   dtype=np.float32, order='C') if ret_weights else None
            num_edges = lib.graph_get_top_edges(self._obj, indices, weights, max_edges)
            if truncate:
                break
            if num_edges <= max_edges:
                break

        if indices is not None and num_edges < max_edges:
            indices.resize((num_edges, 2), refcheck=False)
        if weights is not None and num_edges < max_edges:
            weights.resize((num_edges,), refcheck=False)

        if as_dict:
            if weights is None:
                weights = [None] * num_edges
            return collections.OrderedDict((tuple(i), w) for i, w in zip(indices, weights))

        return indices, weights

    @property
    @cacheable
    def num_edges(self):
        """ Return the number of edges of a graph. """
        return lib.graph_num_edges(self._obj)

    @cacheable
    def nodes(self):
        """
        Return nodes and their frequencies. A node is considered present, when it is
        connected to at least one other node (either as a source or target). For MongoDB
        graphs, a node is present when it appears at least once in the occurrence list
        (even if it doesn't co-occur with any other node).
        """
        return Vector(obj=lib.graph_get_nodes(self._obj))

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

        if indices is not None and num_edges < max_edges:
            indices.resize((num_edges,), refcheck=False)
        if weights is not None and num_edges < max_edges:
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

    def del_small(self, eps=0.0):
        """ Drop entries smaller than the selected `eps`. """
        res = lib.graph_del_small(self._obj, eps)
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

    def save_binary(self, filename):
        """
        Store a graph in a file using binary format.

        # Arguments
        filename: Path to the file to create
        """

        res = lib.graph_save_binary(self._obj, filename.encode("utf-8"))
        if not res:
            raise IOError

    @staticmethod
    def load_binary(filename):
        """
        Load a graph from a binary file into memory.

        # Arguments
        filename: Path to the file to load
        """

        obj = lib.graph_load_binary(filename.encode("utf-8"))
        if not obj:
            raise IOError
        return Graph(obj=obj)

    def sparse_subgraph(self, seeds=None, num_seeds=8, num_neighbors=3, truncate=False):
        """
        Create a sparse subgraph by seleting a few seed edges, and then
        using 'triangular growth' to add additional neighbors.

        # Arguments
        seeds: List of seed edges
        num_seeds: Number of seed edges to select
        num_neighbors: Number of neighbors to add per seed node
        truncate: Truncate list of results if too many.

        # Returns
        Resulting graph.
        """

        if self.directed:
            raise NotImplementedError("Not implemented for directed graphs")

        if seeds is None:
            seeds = self.top_edges(num_seeds, as_dict=True, truncate=truncate)
        if not isinstance(seeds, dict):
            seeds = dict((tuple(i), self[i]) for i in seeds)

        edges = copy.deepcopy(seeds)
        adjacent_edges = memoized(self.adjacent_edges)

        for i, j in seeds.keys():
            edges_i = adjacent_edges(i, as_dict=True)
            edges_j = adjacent_edges(j, as_dict=True)
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

        @c_bfs_callback_p
        def callback(graph, entry, userdata):
            if entry.contents.count > max_count:
                return 1

            entry = entry.contents
            edge_from = entry.edge_from if entry.edge_from != 0xffffffffffffffff else None
            result.append((entry.weight, entry.count, edge_from, entry.edge_to))
            return 0

        res = lib.graph_bfs(self._obj, source, 0, callback, None)
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

        @c_bfs_callback_p
        def callback(graph, entry, userdata):
            if entry.contents.weight > max_weight:
                return 1

            entry = entry.contents
            edge_from = entry.edge_from if entry.edge_from != 0xffffffffffffffff else None
            result.append((entry.weight, entry.count, edge_from, entry.edge_to))
            return 0

        res = lib.graph_bfs(self._obj, source, 1, callback, None)
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

    def all_distances_count(self, source, max_count=0xffffffffffffffff):
        return Vector(obj=lib.graph_get_all_distances_count(self._obj, source, max_count))

    def all_distances_weight(self, source, max_weight=np.inf):
        return Vector(obj=lib.graph_get_all_distances_weight(self._obj, source, max_weight))

    def all_distances_graph(self, use_weights=False):
        return Graph(obj=lib.graph_get_all_distances_graph(self._obj, use_weights))

    def connected_components(self):
        return Vector(obj=lib.graph_get_connected_components(self._obj))

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
        if self._graph is None:
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
        if self._graph is None:
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
    positive: Enforce that all entries must be positive.
    directed: Create a directed time-varying graph.
    streaming: Support for streaming / differential updates.
    primary_key: List or semicolon separated string of attributes.
    """

    def __init__(self, positive=False, directed=False, streaming=False, primary_key=None, obj=None):
        if obj is None:
            flags = 0
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

    def node_label(self, index):
        """
        Shortcut to get the label of a specific node by index.

        # Arguments
        index: Index of the node.

        # Returns
        Node label.
        """

        try:
            node = self.node_by_index(index)
        except KeyError:
            node = {}

        for key in ['label', 'norm', 'text', 'entity_name']:
            try:
                text = node[key]
            except KeyError:
                pass
            else:
                break
        else:
            text = "Node %d" % (index,)

        return text

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

    def sum_edges(self, ts_min=0, ts_max=0xffffffffffffffff, eps=None):
        """
        Add edges in a given timeframe [ts_min, ts_max].

        # Arguments
        ts_min: Left boundary of the interval.
        ts_max: Right boundary of the interval.
        """

        if eps is None:
            eps = 0.0

        return Graph(obj=lib.tvg_sum_edges(self._obj, ts_min, ts_max, eps))

    def sum_nodes(self, ts_min=0, ts_max=0xffffffffffffffff):
        """
        Add node frequencies in a given timeframe [ts_min, ts_max].

        # Arguments
        ts_min: Left boundary of the interval.
        ts_max: Right boundary of the interval.
        """

        return Vector(obj=lib.tvg_sum_nodes(self._obj, ts_min, ts_max))

    def sum_edges_exp(self, ts_min, ts_max, beta=None, log_beta=None, weight=1.0, eps=None):
        """
        Add edges in a given timeframe [ts_min, ts_max], weighted by an exponential
        decay function.

        # Arguments
        ts_min: Left boundary of the interval.
        ts_max: Right boundary of the interval.
        beta: Exponential decay constant.
        """

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

    def count_edges(self, ts_min=0, ts_max=0xffffffffffffffff):
        """
        Count edges in a given timeframe [ts_min, ts_max].

        # Arguments
        ts_min: Left boundary of the interval.
        ts_max: Right boundary of the interval.
        """

        return Graph(obj=lib.tvg_count_edges(self._obj, ts_min, ts_max))

    def count_nodes(self, ts_min=0, ts_max=0xffffffffffffffff):
        """
        Count nodes in a given timeframe [ts_min, ts_max].

        # Arguments
        ts_min: Left boundary of the interval.
        ts_max: Right boundary of the interval.
        """

        return Vector(obj=lib.tvg_count_nodes(self._obj, ts_min, ts_max))

    def count_graphs(self, ts_min=0, ts_max=0xffffffffffffffff):
        """
        Count graphs in a given timeframe [ts_min, ts_max].

        # Arguments
        ts_min: Left boundary of the interval.
        ts_max: Right boundary of the interval.
        """

        res = lib.tvg_count_graphs(self._obj, ts_min, ts_max)
        if res == 0xffffffffffffffff:
            raise MemoryError

        return res

    def topics(self, ts_min=0, ts_max=0xffffffffffffffff, step=None, offset=0, samples=None):
        """
        Extract network topics in the timeframe [ts_min, ts_max].

        # Arguments
        ts_min: Left boundary of the interval.
        ts_max: Right boundary of the interval.
        """

        if step is not None:
            if samples is not None:
                raise ValueError("Invalid parameter combination")
            samples = UniformSamples(step=step, offset=offset)

        if samples is None:
            return Graph(obj=lib.tvg_topics(self._obj, ts_min, ts_max, None, None))

        @c_snapshot_callback_p
        def callback(ts, entry, userdata):
            try:
                entry.contents.ts_min, entry.contents.ts_max = samples(ts)
            except:
                traceback.print_exc()
                return 0
            else:
                return 1

        return Graph(obj=lib.tvg_topics(self._obj, ts_min, ts_max, callback, None))

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

    def documents(self, ts_min=0, ts_max=0xffffffffffffffff, limit=None):
        """ Iterates through all graphs in the given time frame. """

        graph = self.lookup_ge(ts_min)
        count = 0

        while graph is not None:
            if graph.ts > ts_max:
                break
            if limit is not None and count >= limit:
                break

            yield graph

            graph = graph.next
            count += 1

    def compress(self, ts_min=0, ts_max=0xffffffffffffffff, step=None, offset=0, samples=None):
        """ Compress the graph by aggregating timestamps differing by at most `step`. """

        if step is not None:
            if samples is not None:
                raise ValueError("Invalid parameter combination")
            samples = UniformSamples(step=step, offset=offset)

        if samples is None:
            raise ValueError("Missing step/samples parameter")

        @c_snapshot_callback_p
        def callback(ts, entry, userdata):
            try:
                entry.contents.ts_min, entry.contents.ts_max = samples(ts)
            except:
                traceback.print_exc()
                return 0
            else:
                return 1

        res = lib.tvg_compress(self._obj, ts_min, ts_max, callback, None)
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
