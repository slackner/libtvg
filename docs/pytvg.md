
# pytvg


## c_objectid
```python
c_objectid(*args, **kwargs)
```


### hi
Structure/Union member

### lo
Structure/Union member

### type
Structure/Union member

## c_vector
```python
c_vector(*args, **kwargs)
```


### flags
Structure/Union member

### refcount
Structure/Union member

### revision
Structure/Union member

## c_graph
```python
c_graph(*args, **kwargs)
```


### flags
Structure/Union member

### objectid
Structure/Union member

### refcount
Structure/Union member

### revision
Structure/Union member

### ts
Structure/Union member

## c_node
```python
c_node(*args, **kwargs)
```


### index
Structure/Union member

### refcount
Structure/Union member

## c_tvg
```python
c_tvg(*args, **kwargs)
```


### flags
Structure/Union member

### refcount
Structure/Union member

### verbosity
Structure/Union member

## c_mongodb_config
```python
c_mongodb_config(*args, **kwargs)
```


### article_id
Structure/Union member

### article_time
Structure/Union member

### col_articles
Structure/Union member

### col_entities
Structure/Union member

### database
Structure/Union member

### entity_doc
Structure/Union member

### entity_ent
Structure/Union member

### entity_sen
Structure/Union member

### filter_key
Structure/Union member

### filter_value
Structure/Union member

### load_nodes
Structure/Union member

### max_distance
Structure/Union member

### norm_weights
Structure/Union member

### sum_weights
Structure/Union member

### uri
Structure/Union member

### use_pool
Structure/Union member

## c_mongodb
```python
c_mongodb(*args, **kwargs)
```


### refcount
Structure/Union member

## c_bfs_entry
```python
c_bfs_entry(*args, **kwargs)
```


### count
Structure/Union member

### edge_from
Structure/Union member

### edge_to
Structure/Union member

### weight
Structure/Union member

## c_snapshot_entry
```python
c_snapshot_entry(*args, **kwargs)
```


### ts_max
Structure/Union member

### ts_min
Structure/Union member

## metric_entropy
```python
metric_entropy(values, num_bins=50)
```

Rate the importance / interestingness of individual nodes/edges by their entropy.

__Arguments__

- __values__: Values for each node or edge.
- __num_bins__: Number of bins used to create the entropy model.

__Returns__

Dictionary containing the metric for each node or edge.


## metric_entropy_local
```python
metric_entropy_local(values, num_bins=50)
```

Like metric_entropy(), but train a separate model for each time step.

__Arguments__

- __values__: Values for each node or edge.
- __num_bins__: Number of bins used to create the entropy model.

__Returns__

Dictionary containing the metric for each node or edge.


## metric_entropy_2d
```python
metric_entropy_2d(values, num_bins=50)
```

Like metric_entropy(), but train a 2-dimensional model for entropy estimations.

__Arguments__

- __values__: Values for each node or edge.
- __num_bins__: Number of bins used to create the entropy model.

__Returns__

Dictionary containing the metric for each node or edge.


## metric_trend
```python
metric_trend(values)
```

Rate the importance / interestingness of individual nodes/edges by their trend.

__Arguments__

- __values__: Values for each node or edge.

__Returns__

Dictionary containing the metric for each node or edge.


## metric_stability_ratio
```python
metric_stability_ratio(values)
```

Rate the stability of individual nodes/edges by their inverse relative standard deviation.

__Arguments__

- __values__: Values for each node or edge.

__Returns__

Dictionary containing the metric for each node or edge.


## metric_avg
```python
metric_avg(values)
```

Compute the average of individual nodes/edges.

__Arguments__

- __values__: Values for each node or edge.

__Returns__

Average for each node or edge.


## metric_std
```python
metric_std(values)
```

Compute the standard deviation of individual nodes/edges

__Arguments__

- __values__: Values for each node or edge.

__Returns__

Standard deviation for each node or edge.


## metric_pareto
```python
metric_pareto(values, maximize=True, base=0.0)
```

Compute the pareto ranking of two graphs or vectors.

__Arguments:__

values: Values for each node or edge.
maximize: Defines which values should be maximized/minimized.
base: Use `base**(index - 1)` as weight instead of `index`.

__Returns__

Metric for each node or edge.


## metric_stability_pareto
```python
metric_stability_pareto(values, base=0.0)
```

Rate the stability of individual nodes/edges by ranking their average and standard deviation.

__Arguments__

- __values__: Values for each node or edge.
- __base__: Use `base**(index - 1)` as weight instead of `index`.

__Returns__

Metric for each node or edge.


## Vector
```python
Vector(positive=False, obj=None)
```

This object represents a vector of arbitrary / infinite dimension. To achieve that,
it only stores entries that are explicitly set, and assumes that all other entries
of the vector are zero. Internally, it uses hashing to map indices to buckets,
that are stored in contiguous blocks of memory and in sorted order for faster access.

__Arguments__

- __positive__: Enforce that all entries must be positive.


### memory_usage
Return the memory usage currently associated with the vector.

### num_entries
Return the number of entries of a vector.

### revision

Return the current revision of the vector object. This value is incremented
whenever the vector is changed. It is also used by the @cacheable decorator
to check the cache validity.


### empty
```python
Vector.empty()
```
Check if a vector is empty, i.e., if it does not have any entries.

### duplicate
```python
Vector.duplicate()
```
Create an independent copy of the vector.

### clear
```python
Vector.clear()
```
Clear all entries of the vector object.

### has_entry
```python
Vector.has_entry(index)
```
Check if a vector has an entry with index `index`.

### entries
```python
Vector.entries(ret_indices=True, ret_weights=True, as_dict=False)
```

Return all indices and/or weights of a vector.

__Arguments__

- __ret_indices__: Return indices, otherwise None.
- __ret_weights__: Return weights, otherwise None.
- __as_dict__: Return result as dictionary instead of tuple.

__Returns__

`(indices, weights)` or dictionary


### keys
```python
Vector.keys()
```
Iterate over indices of a vector.

### values
```python
Vector.values()
```
Iterate over weights of a vector.

### items
```python
Vector.items()
```
Iterate over indices and weights of a vector.

### tolist
```python
Vector.tolist()
```
Return list of indices of a vector.

### set_entries
```python
Vector.set_entries(indices, weights=None)
```

Short-cut to set multiple entries of a vector.
If weights is None the elements are set to 1.

__Arguments__

- __indices__: List of indices (list or 1d numpy array).
- __weights__: List of weights to set (list or 1d numpy array).


### add_entry
```python
Vector.add_entry(index, weight)
```
Add weight `weight` to the entry with index `index`.

### add_entries
```python
Vector.add_entries(indices, weights=None)
```

Short-cut to update multiple entries of a vector by adding values.
If weights is None the elements are set to 1.

__Arguments__

- __indices__: List of indices (list or 1d numpy array).
- __weights__: List of weights to add (list or 1d numpy array).


### add_vector
```python
Vector.add_vector(other, weight=1.0)
```
Add entries specified by a second vector, optionally multiplied by `weight`.

### sub_entry
```python
Vector.sub_entry(index, weight)
```
Subtract weight `weight` from the entry with index `index`.

### sub_entries
```python
Vector.sub_entries(indices, weights=None)
```

Short-cut to update multiple entries of a vector by subtracting values.
If weights is None the elements are set to 1.

__Arguments__

- __indices__: List of indices (list or 1d numpy array).
- __weights__: List of weights to subtract (list or 1d numpy array).


### sub_vector
```python
Vector.sub_vector(other, weight=1.0)
```
Subtract entries specified by a second vector, optionally multiplied by `weight`.

### del_entries
```python
Vector.del_entries(indices)
```

Short-cut to delete multiple entries from a vector.

__Arguments__

- __indices__: List of indices (list or 1d numpy array).


### mul_const
```python
Vector.mul_const(constant)
```
Perform inplace element-wise multiplication of the vector with `constant`.

### del_small
```python
Vector.del_small(eps=0.0)
```
Drop entries smaller than the selected `eps`.

### sum_weights
```python
Vector.sum_weights()
```
Compute the sum of all weights.

### norm
```python
Vector.norm()
```
Return the L2 norm of the vector.

### mul_vector
```python
Vector.mul_vector(other)
```
Compute the scalar product of the current vector with a second vector `other`.

### sub_vector_norm
```python
Vector.sub_vector_norm(other)
```
Compute L2 norm of (self - other).

### as_dict
```python
Vector.as_dict()
```
Return a dictionary containing all vector entries.

### from_dict
Generate a Vector object from a dictionary.

### save_binary
```python
Vector.save_binary(filename)
```

Store a vector in a file using binary format.

__Arguments__

- __filename__: Path to the file to create


### load_binary

Load a vector from a binary file into memory.

__Arguments__

- __filename__: Path to the file to load


## Graph
```python
Graph(positive=False, directed=False, obj=None)
```

This object represents a graph of arbitrary / infinite dimension. To achieve that,
it only stores edges that are explicitly set, and assumes that all other edges
of the graph have a weight of zero. Internally, it uses hashing to map source and
target indices to buckets, that are stored in contiguous blocks of memory and in
sorted order for faster access.

__Arguments__

- __positive__: Enforce that all entries must be positive.
- __directed__: Create a directed graph.


### id

Get the ID associated with this graph object. This only applies to objects
loaded from an external data source, e.g., from a MongoDB.


### memory_usage
Return the memory usage currently associated with the graph.

### next
Return the (chronologically) next graph object.

### num_edges
Return the number of edges of a graph.

### num_nodes
Return the number of nodes of a graph.

### prev
Return the (chronologically) previous graph object.

### revision

Return the current revision of the graph object. This value is incremented
whenever the graph is changed. It is also used by the @cacheable decorator
to check the cache validity.


### ts

Get the timestamp associated with this graph object. This only applies to
objects that are part of a time-varying graph.


### load_from_mongodb

Load a single graph from a MongoDB database.

__Arguments__

- __id__: Identifier (numeric or objectid) of the document to load
- __positive__: Enforce that all entries must be positive.
- __directed__: Create a directed graph.


### unlink
```python
Graph.unlink()
```
Unlink a graph from the TVG object.

### empty
```python
Graph.empty()
```
Check if the graph is empty, i.e., it does not have any edges.

### duplicate
```python
Graph.duplicate()
```
Create an independent copy of the graph.

### clear
```python
Graph.clear()
```
Clear all edges of the graph object.

### has_edge
```python
Graph.has_edge(indices)
```
Check if the graph has edge `(source, target)`.

### edges
```python
Graph.edges(ret_indices=True, ret_weights=True, as_dict=False)
```

Return all indices and/or weights of a graph.

__Arguments__

- __ret_indices__: Return indices consisting of (source, target), otherwise None.
- __ret_weights__: Return weights, otherwise None.
- __as_dict__: Return result as dictionary instead of tuple.

__Returns__

`(indices, weights)` or dictionary


### keys
```python
Graph.keys()
```
Iterate over indices of a graphs.

### values
```python
Graph.values()
```
Iterate over weights of a graph.

### items
```python
Graph.items()
```
Iterate over indices and weights of a graphs.

### top_edges
```python
Graph.top_edges(max_edges,
                ret_indices=True,
                ret_weights=True,
                as_dict=False,
                truncate=False)
```

Return indices and/or weights of the top edges.

__Arguments__

- __num_edges__: Limit the number of edges returned.
- __ret_indices__: Return indices consisting of (source, target), otherwise None.
- __ret_weights__: Return weights, otherwise None.
- __as_dict__: Return result as dictionary instead of tuple.
- __truncate__: Truncate list of results if too many.

__Returns__

`(indices, weights)` or dictionary


### nodes
```python
Graph.nodes()
```

Return nodes and their frequencies. A node is considered present, when it is
connected to at least one other node (either as a source or target). For MongoDB
graphs, a node is present when it appears at least once in the occurrence list
(even if it doesn't co-occur with any other node).


### adjacent_edges
```python
Graph.adjacent_edges(source, ret_indices=True, ret_weights=True, as_dict=False)
```

Return information about all edges adjacent to a given source edge.

__Arguments__

- __source__: Index of the source node.
- __ret_indices__: Return target indices, otherwise None.
- __ret_weights__: Return weights, otherwise None.
- __as_dict__: Return result as dictionary instead of tuple.

__Returns__

`(indices, weights)` or dictionary


### num_adjacent_edges
```python
Graph.num_adjacent_edges(source)
```
Return the number of adjacent edges to a given `source` node, i.e., the node degree.

### set_edges
```python
Graph.set_edges(indices, weights=None)
```

Short-cut to set multiple edges in a graph.
If weights is None the elements are set to 1.

__Arguments__

- __indices__: List of indices (list of tuples or 2d numpy array).
- __weights__: List of weights to set (list or 1d numpy array).


### add_edge
```python
Graph.add_edge(indices, weight)
```
Add weight `weight` to edge `(source, target)`.

### add_edges
```python
Graph.add_edges(indices, weights=None)
```

Short-cut to update multiple edges of a graph by adding values.
If weights is None the elements are set to 1.

__Arguments__

- __indices__: List of indices (list of tuples or 2d numpy array).
- __weights__: List of weights to set (list or 1d numpy array).


### add_graph
```python
Graph.add_graph(other, weight=1.0)
```
Add edges specified by a second graph, optionally multiplied by `weight`.

### sub_edge
```python
Graph.sub_edge(indices, weight)
```
Subtract weight `weight` from edge `(source, target)`.

### sub_edges
```python
Graph.sub_edges(indices, weights=None)
```

Short-cut to update multiple edges of a graph by subtracting values.
If weights is None the elements are set to 1.

__Arguments__

- __indices__: List of indices (list of tuples or 2d numpy array).
- __weights__: List of weights to set (list or 1d numpy array).


### sub_graph
```python
Graph.sub_graph(other, weight=1.0)
```
Subtract edges specified by a second graph, optionally multiplied by `weight`.

### del_edges
```python
Graph.del_edges(indices)
```

Short-cut to delete multiple edges from a graph.

__Arguments__

- __indices__: List of indices (list of tuples or 2d numpy array).


### mul_const
```python
Graph.mul_const(constant)
```
Perform inplace element-wise multiplication of all graph edges with `constant`.

### del_small
```python
Graph.del_small(eps=0.0)
```
Drop entries smaller than the selected `eps`.

### mul_vector
```python
Graph.mul_vector(other)
```
Compute the matrix-vector product of the graph with vector `other`.

### in_degrees
```python
Graph.in_degrees()
```
Compute and return a vector of in-degrees.

### in_weights
```python
Graph.in_weights()
```
Compute and return a vector of in-weights.

### out_degrees
```python
Graph.out_degrees()
```
Compute and return a vector of out-degrees.

### out_weights
```python
Graph.out_weights()
```
Compute and return a vector of out-weights.

### degree_anomalies
```python
Graph.degree_anomalies()
```
Compute and return a vector of degree anomalies.

### weight_anomalies
```python
Graph.weight_anomalies()
```
Compute and return a vector of weight anomalies.

### sum_weights
```python
Graph.sum_weights()
```
Compute the sum of all weights.

### power_iteration
```python
Graph.power_iteration(initial_guess=None,
                      num_iterations=0,
                      tolerance=None,
                      ret_eigenvalue=True)
```

Compute and return the eigenvector (and optionally the eigenvalue).

__Arguments__

- __initial_guess__: Initial guess for the solver.
- __num_iterations__: Number of iterations.
- __tolerance__: Desired tolerance.
- __ret_eigenvalue__: Also return the eigenvalue. This requires one more iteration.

__Returns__

`(eigenvector, eigenvalue)`


### filter_nodes
```python
Graph.filter_nodes(nodes)
```

Create a subgraph by only keeping edges, where at least one node is
part of the subset specified by the `nodes` parameter.

__Arguments__

- __nodes__: Vector, list or set of nodes to preserve

__Returns__

Resulting graph.


### normalize
```python
Graph.normalize()
```

Normalize a graph based on the in and out-degrees of neighbors.

__Returns__

Resulting graph.


### save_binary
```python
Graph.save_binary(filename)
```

Store a graph in a file using binary format.

__Arguments__

- __filename__: Path to the file to create


### load_binary

Load a graph from a binary file into memory.

__Arguments__

- __filename__: Path to the file to load


### sparse_subgraph
```python
Graph.sparse_subgraph(seeds=None, num_seeds=8, num_neighbors=3, truncate=False)
```

Create a sparse subgraph by seleting a few seed edges, and then
using 'triangular growth' to add additional neighbors.

__Arguments__

- __seeds__: List of seed edges
- __num_seeds__: Number of seed edges to select
- __num_neighbors__: Number of neighbors to add per seed node
- __truncate__: Truncate list of results if too many.

__Returns__

Resulting graph.


### bfs_count
```python
Graph.bfs_count(source, max_count=None)
```

Perform a breadth-first search in the graph, starting from node `source`.
In this version, the order is based solely on the number of links.

__Arguments__

- __source__: Index of the source node.
- __max_count__: Maximum depth.

__Returns__

List of tuples `(weight, count, edge_from, edge_to)`.


### bfs_weight
```python
Graph.bfs_weight(source, max_weight=inf)
```

Perform a breadth-first search in the graph, starting from node `source`.
In this version, the order is based on the sum of the weights.

__Arguments__

- __source__: Index of the source node.
- __max_weight__: Maximum weight.

__Returns__

List of tuples `(weight, count, edge_from, edge_to)`.


### as_dict
```python
Graph.as_dict()
```
Return a dictionary containing all graph edges.

### from_dict
Generate a Graph object from a dictionary.

## Node
```python
Node(obj=None, **kwargs)
```

This object represents a node. Since nodes are implicit in our model, they
should only have static attributes that do not depend on the timestamp.
For now, both node attribute keys and values are limited to the string type -
in the future this might be extended to other data types. Attributes related
to the primary key (that uniquely identify a node in the context of a time-
varying-graph) must be set before both objects are linked. All other attributes
can be set at any time.

__Arguments__

- __**kwargs__: Key-value pairs of type string to assign to the node.


### index
Return the index of the node.

### text
Short-cut to return the 'text' attribute of a node.

### unlink
```python
Node.unlink()
```

Unlink the node from the time-varying graph. The node itself stays valid,
but it is no longer returned for any `node_by_index` or `node_by_primary_key`
call.


### as_dict
```python
Node.as_dict()
```
Return a dictionary containing all node attributes.

## TVG
```python
TVG(positive=False,
    directed=False,
    streaming=False,
    primary_key=None,
    obj=None)
```

This object represents a time-varying graph.

__Arguments__

- __positive__: Enforce that all entries must be positive.
- __directed__: Create a directed time-varying graph.
- __streaming__: Support for streaming / differential updates.
- __primary_key__: List or semicolon separated string of attributes.


### memory_usage
Return the memory usage currently associated with the TVG.

### link_graph
```python
TVG.link_graph(graph, ts)
```

Link a graph to the time-varying-graph object.

__Arguments__

- __graph__: The graph to link.
- __ts__: Time-stamp of the graph (as uint64, typically UNIX timestamp in milliseconds).


### set_primary_key
```python
TVG.set_primary_key(key)
```

Set or update the primary key used to distinguish graph nodes. The key can
consist of one or multiple attributes, and is used to identify a node
(especially, when loading from an external source, that does not use integer
identifiers).

__Arguments__

- __key__: List or semicolon separated string of attributes.


### link_node
```python
TVG.link_node(node, index=None)
```

Link a node to the time-varying-graph object.

__Arguments__

- __node__: The node to link.
- __index__: Index to assign to the node, or `None` if the next empty index should be used.


### Node
```python
TVG.Node(**kwargs)
```

Create a new node assicated with the graph. Note that all primary key attributes
must be set immediately during construction, it is not possible to change them later.

__Arguments__

- __**kwargs__: Key-value pairs of type string to assign to the node.


### node_by_index
```python
TVG.node_by_index(index)
```

Lookup a node by index.

__Arguments__

- __index__: Index of the node.

__Returns__

Node object.


### node_by_primary_key
```python
TVG.node_by_primary_key(**kwargs)
```

Lookup a node by its primary key. This must match the primary key set with
`set_primary_key` (currently, a time-varying graph can only have one key).

__Arguments__

- __**kwargs__: Key-value pairs of the primary key.

__Returns__

Node object.


### node_by_text
```python
TVG.node_by_text(text)
```
Lookup a node by its text (assumes that `text` is the primary key).

### node_label
```python
TVG.node_label(index)
```

Shortcut to get the label of a specific node by index.

__Arguments__

- __index__: Index of the node.

__Returns__

Node label.


### load

Load a time-varying-graph from an external data source.

__Arguments__

- __source__: Data source to load (currently either a file path, or a MongoDB object).
- __nodes__: Secondary data source to load node attributes (must be a file path).
- __*args, **kwargs__: Arguments passed through to the `TVG()` constructor.


### load_graphs_from_file
```python
TVG.load_graphs_from_file(filename)
```
Load a time-varying-graph (i.e., a collection of graphs) from a file.

### load_nodes_from_file
```python
TVG.load_nodes_from_file(filename, key=None)
```
Load node attributes from a file.

### load_graphs_from_mongodb
```python
TVG.load_graphs_from_mongodb(mongodb)
```
Load a time-varying-graph (i.e., multiple graphs) from a MongoDB.

### enable_mongodb_sync
```python
TVG.enable_mongodb_sync(mongodb, batch_size=0, cache_size=0)
```

Enable synchronization with a MongoDB server. Whenever more data is needed
(e.g., querying the previous or next graph, or looking up graphs in a certain
range), requests are sent to the database. Each request loads up to
`batch_size` graphs. The maximum amount of data kept in memory can be
controlled with the `cache_size` parameter.

__Arguments__

- __mongodb__: MongoDB object.
- __batch_size__: Maximum number of graphs to load in a single request.
- __cache_size__: Maximum size of the cache (in bytes).


### disable_mongodb_sync
```python
TVG.disable_mongodb_sync()
```
Disable synchronization with a MongoDB server.

### enable_query_cache
```python
TVG.enable_query_cache(cache_size=0)
```

Enable the query cache. This can be used to speed up query performance, at the
cost of higher memory usage. The maximum amount of data kept in memory can be
controlled with the `cache_size` parameter.

__Arguments__

- __cache_size__: Maximum size of the cache (in bytes).


### disable_query_cache
```python
TVG.disable_query_cache()
```
Disable the query cache.

### invalidate_queries
```python
TVG.invalidate_queries(ts_min, ts_max)
```
Invalidate queries in a given timeframe [ts_min, ts_max].

### sum_edges
```python
TVG.sum_edges(ts_min, ts_max, eps=None)
```

Add edges in a given timeframe [ts_min, ts_max].

__Arguments__

- __ts_min__: Left boundary of the interval.
- __ts_max__: Right boundary of the interval.


### sum_nodes
```python
TVG.sum_nodes(ts_min, ts_max)
```

Add node frequencies in a given timeframe [ts_min, ts_max].

__Arguments__

- __ts_min__: Left boundary of the interval.
- __ts_max__: Right boundary of the interval.


### sum_edges_exp
```python
TVG.sum_edges_exp(ts_min,
                  ts_max,
                  beta=None,
                  log_beta=None,
                  weight=1.0,
                  eps=None)
```

Add edges in a given timeframe [ts_min, ts_max], weighted by an exponential
decay function.

__Arguments__

- __ts_min__: Left boundary of the interval.
- __ts_max__: Right boundary of the interval.
- __beta__: Exponential decay constant.


### sum_edges_exp_norm
```python
TVG.sum_edges_exp_norm(ts_min, ts_max, beta=None, log_beta=None, eps=None)
```

Add edges in a given timeframe [ts_min, ts_max], weighted by an exponential
smoothing function.

__Arguments__

- __ts_min__: Left boundary of the interval.
- __ts_max__: Right boundary of the interval.
- __beta__: Exponential decay constant.


### count_edges
```python
TVG.count_edges(ts_min, ts_max)
```

Count edges in a given timeframe [ts_min, ts_max].

__Arguments__

- __ts_min__: Left boundary of the interval.
- __ts_max__: Right boundary of the interval.


### count_nodes
```python
TVG.count_nodes(ts_min, ts_max)
```

Count nodes in a given timeframe [ts_min, ts_max].

__Arguments__

- __ts_min__: Left boundary of the interval.
- __ts_max__: Right boundary of the interval.


### count_graphs
```python
TVG.count_graphs(ts_min, ts_max)
```

Count graphs in a given timeframe [ts_min, ts_max].

__Arguments__

- __ts_min__: Left boundary of the interval.
- __ts_max__: Right boundary of the interval.


### topics
```python
TVG.topics(ts_min, ts_max, step=None, offset=0, samples=None)
```

Extract network topics in the timeframe [ts_min, ts_max].

__Arguments__

- __ts_min__: Left boundary of the interval.
- __ts_max__: Right boundary of the interval.


### sample_graphs
```python
TVG.sample_graphs(ts_min,
                  ts_max,
                  sample_width,
                  sample_steps=9,
                  method=None,
                  *args,
                  **kwargs)
```

Sample graphs in the timeframe [ts_min, ts_max].

__Arguments__

- __ts_min__: Left boundary of the interval.
- __ts_max__: Right boundary of the interval.
- __sample_width__: Width of each sample.
- __sample_steps__: Number of values to collect.
- __method__: Method to use (default: 'sum_edges').

__Yields__

Sampled graphs.


### sample_eigenvectors
```python
TVG.sample_eigenvectors(ts_min,
                        ts_max,
                        sample_width,
                        sample_steps=9,
                        tolerance=None,
                        method=None,
                        *args,
                        **kwargs)
```

Iterative power iteration algorithm to track eigenvectors of a graph over time.
Eigenvectors are collected within the timeframe [ts_min, ts_max]. Each entry
of the returned dictionary contains sample_steps values collected at equidistant
time steps.

__Arguments__

- __ts_min__: Left boundary of the interval.
- __ts_max__: Right boundary of the interval.
- __sample_width__: Width of each sample.
- __sample_steps__: Number of values to collect.
- __tolerance__: Tolerance for the power_iteration algorithm.
- __method__: Method to use (default: 'sum_edges').

__Returns__

Dictionary containing lists of collected values for each node.


### lookup_ge
```python
TVG.lookup_ge(ts=0)
```
Search for the first graph with timestamps `>= ts`.

### lookup_le
```python
TVG.lookup_le(ts=18446744073709551615)
```
Search for the last graph with timestamps `<= ts`.

### lookup_near
```python
TVG.lookup_near(ts)
```
Search for a graph with a timestamp close to `ts`.

### documents
```python
TVG.documents(ts_min=0, ts_max=18446744073709551615, limit=None)
```
Iterates through all graphs in the given time frame.

### compress
```python
TVG.compress(ts_min=0,
             ts_max=18446744073709551615,
             step=None,
             offset=0,
             samples=None)
```
Compress the graph by aggregating timestamps differing by at most `step`.

## MongoDB
```python
MongoDB(uri,
        database,
        col_articles,
        article_id,
        article_time,
        col_entities,
        entity_doc,
        entity_sen,
        entity_ent,
        use_pool=True,
        load_nodes=False,
        sum_weights=True,
        norm_weights=False,
        max_distance=None,
        filter_key=None,
        filter_value=None,
        use_objectids=None,
        obj=None)
```

This object represents a MongoDB connection.

__Arguments__

- __uri__: URI to identify the MongoDB server, e.g., mongodb://localhost.
- __database__: Name of the database.

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

