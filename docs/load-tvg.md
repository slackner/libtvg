## Loading TVGs from external data sources

Time Varying Graph Explorer supports different file formats and data sources.
In the following, we discuss these different options and provide short code
snippets for each of them.

Note that a time varying graph object (`pytvg.TVG` class) acts as a container
for arbitrary many graphs. Each graph (`pytvg.Graph` class) consists of a set
of edges associated with weights and a timestamp. Nodes (`pytvg.Node` class)
are assumed to be always present (i.e., they do not have any dynamic aspects),
and are only used to store additional attributes.

Load a TVG from a file
======================

For small or medium sized datasets, it makes sense to store time varying graphs
explicitly on the hard disk. With this framework, a time varying graph consists
of two files, a `*.graph` file storing the edge information, and a `*.nodes`
file (often much smaller) storing the node information.

For `*.graph` files the format should be as follows (placeholders replaced by
actual values, of course). If the source dataset uses a different file format,
it will be necessary to manually write scripts for conversion. However, this is
outside of the scope of this documentation.

```
# Lines starting with '#' are ignored
StartNodeID EndNodeID Weight Timestamp  \
StartNodeID EndNodeID Weight Timestamp   } Edges
StartNodeID EndNodeID Weight Timestamp  /
[...]
```

Lines starting with "#" are comments and will be ignored. All remaining lines
describe edges. Each edge is defined by a start node ID, an end node ID, a
weight, and a timestamp. The file should be sorted by timestamp. The node IDs
can be arbitrary integers between 0 and 2^64 - 1. An example graph can be seen
in `src/data/example-tvg.graph`.

The node attributes are stored in a separate `*.nodes` file with the following
format:

```
# Lines starting with '#' are ignored
NodeID TextOrOtherDescription
NodeID TextOrOtherDescription
NodeID TextOrOtherDescription
[...]
```

Similar to above, lines starting with "#" are comments and will be ignored. The
remaining lines map node IDs to some text attribute. Note that, at least for now,
such files can only store one attribute per node.

To load the data, it is sufficient to call:

```python
import pytvg

tvg = pytvg.TVG.load("path/to/dataset.graph", nodes="path/to/dataset.nodes")
```

Depending on the use-case, additional attributes like `nonzero=True` or
`positive=True` can be used to restrict the edge weights to nonzero / positive
values, e.g.,

```python
tvg = pytvg.TVG.load("path/to/dataset.graph", nodes="path/to/dataset.nodes",
                     nonzero=True, positive=True)
```

Load TVGs from a MongoDB
========================

This method is currently limited to co-occurrence networks, since raw graph data
is not supported yet. To use this import method, the MongoDB should contain two
collections, storing the following information:

* A collection storing articles. Each article must be identified with a unique
identifier of type integer, and an article timestamp, i.e., tuples of the form
`(article ID, timestamp)`.

* A second collection, storing entities found in each sentence of the article.
The collection should contain tuples `(article ID, sentence index, entity ID)`.

First of all, it is necessary to create a `MongoDB` object.

```python
import pytvg

db = pytvg.MongoDB("mongodb://localhost:27017", # URI
                   "DatabaseName",              # Name of the database

                   "ArticleCollectionName",     # Name of the article collection
                   "ArticleIDColumn",           # Column name / Key for article ID
                   "ArticleTimeColumn",         # Column name / Key for article time

                   "EntityCollectionName",      # Name of the entity collection
                   "EntityArticleIDColumn",     # Column name / Key for article ID
                   "EntitySentenceIndexColumn", # Column name / Key for sentence index
                   "EntityEntityIDColumn",      # Column name / Key for entity ID

                   load_nodes=True,             # Whether to load node attributes
                   max_distance=5)              # Maximum distance of sentences
```

To load the full database content into memory, the following code can be used:

```python
tvg = pytvg.TVG.load(db, primary_key="attr1")
```

If `load_nodes` is set to `False`, the entity ID column must contain integer
values. In this case setting a primary key can be skipped. If it is set to
`True`, nodes will created for entities, whenever they first appear in the
dataset. In this case, the entity ID column can be a semicolon separated list
of attributes to load for each node, e.g., `"attr1;attr2;attr3"`. Besides the
attributes that form the primary key, this can also include additional keys
that should be loaded from the dataset.

Be aware that this method is still only suitable for small to medium sized
collections, since the full dataset is kept in memory. Also, the generated graphs
are typically bigger than the original occurrence data.

For large amounts of data, it is better to enable synchronization, such that
required data is loaded on-demand.

```python
tvg = pytvg.TVG() # optional: nonzero=True, positive=True
tvg.set_primary_key("attr1")
tvg.enable_mongodb_sync(db, batch_size=64, cache_size=0x10000000) # 256 MB
```

The `batch_size` parameter can be used to control how many graphs are loaded per
request. The `cache_size` parameter can be used to limit the amount of memory.
As soon as the total memory consumption of graphs exceeds the size specified by
`cache_size`, those that are no longer needed will be pruned from the cache
(starting from the oldest one).

Creating an in-memory TVG
=========================

Creating graphs in-memory is the most flexible (but also slowest) option to
load time varying graphs. The first step is to create a time-varying graph
object:

```python
import pytvg

tvg = pytvg.TVG(nonzero=True, positive=True)
```

The exact arguments depend on the use-case: If the data has certain restrictions
(e.g., only non-zero and positive edge weights), it makes sense to enforce this
by passing `nonzero=True` and `positive=True` to the constructor. It is also
possible to create a directed time varying graph by passing `directed=True`.

The next step is to create graphs for each record in the source dataset:

```python
# [... load the source dataset 'source' ...]

for record in source:
    g = pytvg.Graph()
    # [... initialize g here ...]
    tvg.link_graph(g, ts=record.timestamp)
```

In this snippet, each graph is first created independently, and then later linked
to the time varying graph object. This has the advantage that objects are never
accessed before they are fully initialized. When linking a graph, it is also
necessary to provide a timestamp - here, we typically use the UNIX timestamp
with millisecond precision, as used by MongoDB, for example.

To initialize each graph, it is sufficient to set the corresponding edge weights.
When constructing co-occurrence networks, these weights could be computed based on
the distance between occurrences of entities. We do not impose any restriction,
and leave it up to the user how the edge weights are computed.

```python
for source, target, weight in record:
    g[source, target] = weight # e.g., exp(-distance) for co-occurrence networks
```

To avoid lots of ctypes function calls, there is also a batch processing function
to set multiple edges at once. The indices argument should be a list of tuples
or 2d numpy array, the weights parameter a list of floats or 1d numpy array.

```python
import numpy as np

indices = record.indices # e.g., np.array([[0, 1], [1, 2], [2, 0]])
weights = record.weights # e.g., np.array([1.0, 2.0, 3.0])
g.set_edges(indices, weights)
```

One step we skipped so far is the mapping of entities to integer values. This
could be done manually - but pytvg also offers functions to simplify this task.
First, it is necessary to tell libtvg which attribute(s) are used to identify
entities. Assume our entities have attributes `type`, `name` and `description`,
while the first two are already sufficient to identify an entity. Then we could
set a primary key as follows:

```python
tvg.set_primary_key(["type", "name"])
```

Then, while loading all records from our data source, we can generate unique
identifiers whenever we need them:

```python
for source_type, source_name, target_type, target_name, weight in record:
    source = tvg.Node(type=source_type, name=source_name) # get or create identifier
    target = tvg.Node(type=target_type, name=target_name) # get or create identifier
    g[source.index, target.index] = weight
```

All attributes can later be accessed similar to a Python dictionary:

```python
node = tvg.node_by_primary_key(type="some type", name="some name")
print (node['type'])
print (node['name'])
print (node['description'])
```
