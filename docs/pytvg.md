
# pytvg


## c_vector
```python
c_vector(*args, **kwargs)
```


### eps
Structure/Union member

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


### eps
Structure/Union member

### flags
Structure/Union member

### id
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

## c_window
```python
c_window(*args, **kwargs)
```


### eps
Structure/Union member

### refcount
Structure/Union member

### ts
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

### load_nodes
Structure/Union member

### max_distance
Structure/Union member

### uri
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

## Vector
```python
Vector(nonzero=False, positive=False, obj=None)
```

This object represents a vector of arbitrary / infinite dimension. To achieve that,
it only stores entries that are explicitly set, and assumes that all other entries
of the vector are zero. Internally, it uses hashing to map indices to buckets,
that are stored in contiguous blocks of memory and in sorted order for faster access.

__Arguments__

- __nonzero__: Enforce that all entries must be non-zero.
- __positive__: Enforce that all entries must be positive.


### eps

Get/set the current value of epsilon. This is used to determine whether an
entry is equal to zero. Whenever |x| < eps, it is treated as zero.


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

### has_entry
```python
Vector.has_entry(index)
```
Check if a vector has an entry with index `index`.

### entries
```python
Vector.entries(ret_indices=True, ret_weights=True)
```

Return all indices and/or weights of a vector.

__Arguments__

- __ret_indices__: Return indices, otherwise None.
- __ret_weights__: Return weights, otherwise None.

__Returns__

`(indices, weights)`


### set_entries
```python
Vector.set_entries(indices, weights)
```

Short-cut to set multiple entries of a vector.

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
Vector.add_entries(indices, weights)
```

Short-cut to update multiple entries of a vector by adding values.

__Arguments__

- __indices__: List of indices (list or 1d numpy array).
- __weights__: List of weights to add (list or 1d numpy array).


### sub_entry
```python
Vector.sub_entry(index, weight)
```
Subtract weight `weight` from the entry with index `index`.

### sub_entries
```python
Vector.sub_entries(indices, weights)
```

Short-cut to update multiple entries of a vector by subtracting values.

__Arguments__

- __indices__: List of indices (list or 1d numpy array).
- __weights__: List of weights to subtract (list or 1d numpy array).


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

## Graph
```python
Graph(nonzero=False, positive=False, directed=False, obj=None)
```

This object represents a graph of arbitrary / infinite dimension. To achieve that,
it only stores edges that are explicitly set, and assumes that all other edges
of the graph have a weight of zero. Internally, it uses hashing to map source and
target indices to buckets, that are stored in contiguous blocks of memory and in
sorted order for faster access.

__Arguments__

- __nonzero__: Enforce that all entries must be non-zero.
- __positive__: Enforce that all entries must be positive.
- __directed__: Create a directed graph.


### eps

Get/set the current value of epsilon. This is used to determine whether an
entry is equal to zero. Whenever |x| < eps, it is treated as zero.


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

- __id__: Identifier of the document to load
- __nonzero__: Enforce that all entries must be non-zero.
- __positive__: Enforce that all entries must be positive.
- __directed__: Create a directed graph.


### enable_delta
```python
Graph.enable_delta()
```

Enable tracking of changes in a separate graph object. Whenever an edge of the
original graph is updated, the same change will also be performed on the delta
graph.


### disable_delta
```python
Graph.disable_delta()
```
Disable tracking of changes.

### get_delta
```python
Graph.get_delta()
```

Return a reference to the delta graph, and the current multiplier. Those values
can be used to reconstruct the current graph if the previous state is known.


### empty
```python
Graph.empty()
```
Check if the graph is empty, i.e., it does not have any edges.

### has_edge
```python
Graph.has_edge(indices)
```
Check if the graph has edge `(source, target)`.

### edges
```python
Graph.edges(ret_indices=True, ret_weights=True)
```

Return all indices and/or weights of a graph.

__Arguments__

- __ret_indices__: Return indices consisting of (source, target), otherwise None.
- __ret_weights__: Return weights, otherwise None.

__Returns__

`(indices, weights)`


### nodes
```python
Graph.nodes()
```

Return a list of all nodes. A node is considered present, when it is connected
to at least one other node (either as a source or target).


### adjacent_edges
```python
Graph.adjacent_edges(source, ret_indices=True, ret_weights=True)
```

Return information about all edges adjacent to a given source edge.

__Arguments__

- __source__: Index of the source node.
- __ret_indices__: Return target indices, otherwise None.
- __ret_weights__: Return weights, otherwise None.

__Returns__

`(indices, weights)`


### num_adjacent_edges
```python
Graph.num_adjacent_edges(source)
```
Return the number of adjacent edges to a given `source` node, i.e., the node degree.

### set_edges
```python
Graph.set_edges(indices, weights)
```

Short-cut to set multiple edges in a graph.

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
Graph.add_edges(indices, weights)
```

Short-cut to update multiple edges of a graph by adding values.

__Arguments__

- __indices__: List of indices (list of tuples or 2d numpy array).
- __weights__: List of weights to set (list or 1d numpy array).


### sub_edge
```python
Graph.sub_edge(indices, weight)
```
Subtract weight `weight` from edge `(source, target)`.

### sub_edges
```python
Graph.sub_edges(indices, weights)
```

Short-cut to update multiple edges of a graph by subtracting values.

__Arguments__

- __indices__: List of indices (list of tuples or 2d numpy array).
- __weights__: List of weights to set (list or 1d numpy array).


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

### power_iteration
```python
Graph.power_iteration(num_iterations=0, ret_eigenvalue=True)
```

Compute and return the eigenvector (and optionally the eigenvalue).

__Arguments__

- __num_iterations__: Number of iterations.
- __ret_eigenvalue__: Also return the eigenvalue. This requires one more iteration.

__Returns__

`(eigenvector, eigenvalue)`


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


### encode_visjs
```python
Graph.encode_visjs(node_attributes=None)
```

Encode a graph as a Python dictionary for parsing with visjs.

__Arguments__

- __node_attributes__: Function to query node attributes.

__Returns__

`Dictionary containing the following key-value pairs`:

cmd: Either `"network_set"` for full updates, or `"network_update"` for partial updates.
nodes: List of nodes (with ids and custom attributes).
edges: List of edges (with ids and weights).
deleted_nodes: List of deleted node ids (only for `cmd = "network_update"`).
deleted_edges: List of deleted edge ids (only for `cmd = "network_update"`).


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
TVG(nonzero=False,
    positive=False,
    directed=False,
    streaming=False,
    primary_key=None,
    obj=None)
```

This object represents a time-varying graph.

__Arguments__

- __nonzero__: Enforce that all entries must be non-zero.
- __positive__: Enforce that all entries must be positive.
- __directed__: Create a directed time-varying graph.
- __streaming__: Support for streaming / differential updates.
- __primary_key__: List or semicolon separated string of attributes.


### link_graph
```python
TVG.link_graph(graph, ts)
```

Link a graph to the time-varying-graph object.

__Arguments__

- __graph__: The graph to link.
- __ts__: Time-stamp of the graph (as uint64, typically UNIX timestamp in milliseconds).


### Graph
```python
TVG.Graph(ts)
```
Create a new graph associated with the time-varying-graph object.

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
TVG.load_nodes_from_file(filename)
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
`batch_size` graphs. The total amount of data kept in memory can be controlled
with the `cache_size` parameter.

__Arguments__

- __mongodb__: MongoDB object.
- __batch_size__: Maximum number of graphs to load in a single request.
- __cache_size__: Maximum size of the cache (in bytes).


### disable_mongodb_sync
```python
TVG.disable_mongodb_sync()
```
Disable synchronization with a MongoDB server.

### WindowRect
```python
TVG.WindowRect(window_l, window_r)
```

Create a new rectangular filter window to aggregate data in a specific range
around a fixed timestamp. Only graphs in [ts + window_l, ts + window_r] are
considered.

__Arguments__

- __window_l__: Left boundary of the interval, relative to the timestamp.
- __window_r__: Right boundary of the interval, relative to the timestamp.


### WindowDecay
```python
TVG.WindowDecay(window, beta=None, log_beta=None)
```

Create a new exponential decay window to aggregate data in a specific range
around a fixed timestamp. Only graphs in [ts - window, window] are considered.

__Arguments__

- __window__: Amount of data in the past to consider.
- __beta__: Exponential decay constant.


### WindowSmooth
```python
TVG.WindowSmooth(window, beta=None, log_beta=None)
```

Create a new exponential smoothing window to aggregate data in a specific range
around a fixed timestamp. Only graphs in [ts - window, window] are considered.

__Arguments__

- __window__: Amount of data in the past to consider.
- __beta__: Exponential decay constant.


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

### compress
```python
TVG.compress(step, offset=0)
```
Compress the graph by aggregating timestamps differing by at most `step`.

## Window
```python
Window(obj=None)
```

This object represents a sliding window, which can be used to extract and aggregate
data in a specific timeframe. Once the object has been created, most parameters can
not be changed anymore. Only the timestamp can be changed.


### eps

Get/set the current value of epsilon. This is used to determine whether an
entry is equal to zero. Whenever |x| < eps, it is treated as zero.


### ts
Return the current timestamp of the window.

### clear
```python
Window.clear()
```
"
Clear all additional data associated with the window, and force a full recompute
when the `update` function is used the next time.


### update
```python
Window.update(ts)
```

Move the sliding window to a new timestamp. Whenever possible, the previous state
will be reused to speed up the computation. For rectangular windows, for example,
it is sufficient to add data points that are moved into the interval, and to remove
data points that are now outside of the interval. For exponential windows, it is
also necessary to perform a multiplication of the full graph.

__Arguments__

- __ts__: New timestamp of the window.

__Returns__

Graph object.


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
        load_nodes,
        max_distance,
        obj=None)
```

This object represents a MongoDB connection.

__Arguments__

- __uri__: URI to identify the MongoDB server, e.g., mongodb://localhost.
- __database__: Name of the database.

col_articles: Name of the articles collection.
article_id: Name of the article ID key.
article_time: Name of the article time key.

col_entities: Name of the entities collection.
entity_doc: Name of the entity doc key.
entity_sen: Name of the entity sen key.
entity_ent: Name(s) of the entity ent key, e.g., attr1;attr2;attr3.

load_nodes: Load node attributes.
max_distance: Maximum distance of mentions.

