#!/usr/bin/python3
from SimpleWebSocketServer import SimpleWebSocketServer
from SimpleWebSocketServer import WebSocket
import numpy as np
import traceback
import argparse
import math
import json
import copy
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../libtvg"))
import pytvg

clients = []
default_context = {
# is setup in the config file
#    'nodeTypes': {
#        'LOC': {
#            'title': 'location',
#            'color': '#bf8080',
#            'class': 'mr-3 far fa-compass',
#        },
#    },
#    'defaultColor': '#bf8080',
#    'nodeSize': 'eigenvector',
}

class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)

        if isinstance(obj, np.int64):
            return int(obj)

        if isinstance(obj, np.uint64):
            return int(obj)

        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)

class Client(WebSocket):
    def handleConnected(self):
        try:
            self.event_connected()
            clients.append(self)
        except Exception:
            print("Exception in eventConnected:")
            print("-"*60)
            traceback.print_exc(file=sys.stdout)
            print("-"*60)
            # Pass through exception, SimpleWebSocketServer
            # will then terminate the connection.
            raise

    def handleMessage(self):
        try:
            self.event_message(self.data)
        except Exception:
            print("Exception in eventMessage:")
            print("-"*60)
            traceback.print_exc(file=sys.stdout)
            print("-"*60)
            # Don't pass through exception.

    def handleClose(self):
        clients.remove(self)

        try:
            self.event_close()
        except Exception:
            print("Exception in eventClose:")
            print("-"*60)
            traceback.print_exc(file=sys.stdout)
            print("-"*60)
            # Pass through exception.
            raise

    def send_message_json(self, **kwargs):
        data = json.dumps(kwargs, indent=4, cls=ComplexEncoder)
        self.sendMessage(data)

    def timeline_seek(self, ts_min=None, ts_max=None):
        """
        Update the graph view. If ts is given, seek to the given timestamp.
        Note that the timestamp specifies the right end of the window. If
        width is given, also update the width of the window.
        """

        if ts_min is None:
            ts_min = self.ts_min
        if ts_max is None:
            ts_max = self.ts_max

        if ts_min is None or ts_max is None:
            print('Error: Cannot seek with partial information!')
            return

        log_scale = True
        custom_colors = {}

        if self.context['nodeSize'] == 'in_degrees':
            graph = dataset_tvg.sum_edges(ts_min, ts_max)
            values = graph.in_degrees()

        elif self.context['nodeSize'] == 'in_weights':
            graph = dataset_tvg.sum_edges(ts_min, ts_max)
            values = graph.in_weights()

        elif self.context['nodeSize'] == 'out_degrees':
            graph = dataset_tvg.sum_edges(ts_min, ts_max)
            values = graph.out_degrees()

        elif self.context['nodeSize'] == 'out_weights':
            graph = dataset_tvg.sum_edges(ts_min, ts_max)
            values = graph.out_weights()

        elif self.context['nodeSize'] == 'degree_anomalies':
            graph = dataset_tvg.sum_edges(ts_min, ts_max)
            values = graph.degree_anomalies()

        elif self.context['nodeSize'] == 'weight_anomalies':
            graph = dataset_tvg.sum_edges(ts_min, ts_max)
            values = graph.weight_anomalies()

        elif self.context['nodeSize'] == 'eigenvector':
            graph = dataset_tvg.sum_edges(ts_min, ts_max)
            values, _ = graph.power_iteration(tolerance=1e-3, ret_eigenvalue=False)

        elif self.context['nodeSize'] == 'stable_nodes':
            values = dataset_tvg.sample_eigenvectors(ts_min, ts_max, sample_width=(ts_max - ts_min) / 3, tolerance=1e-3)
            values = pytvg.metric_stability_pareto(values)
            for i in values.keys():
                values[i] = -values[i]
            log_scale = False

        elif self.context['nodeSize'] == 'entropy':
            values = dataset_tvg.sample_eigenvectors(ts_min, ts_max, sample_width=(ts_max - ts_min) / 3, tolerance=1e-3)
            values = pytvg.metric_entropy(values)
            log_scale = False

        elif self.context['nodeSize'] == 'entropy_local':
            values = dataset_tvg.sample_eigenvectors(ts_min, ts_max, sample_width=(ts_max - ts_min) / 3, tolerance=1e-3)
            values = pytvg.metric_entropy_local(values)
            log_scale = False

        elif self.context['nodeSize'] == 'entropy_2d':
            values = dataset_tvg.sample_eigenvectors(ts_min, ts_max, sample_width=(ts_max - ts_min) / 3, tolerance=1e-3)
            values = pytvg.metric_entropy_2d(values)
            log_scale = False

        elif self.context['nodeSize'] == 'trend':
            values = dataset_tvg.sample_eigenvectors(ts_min, ts_max, sample_width=(ts_max - ts_min) / 3, tolerance=1e-3)
            values = pytvg.metric_trend(values)
            for i in values.keys():
                custom_colors[i] = 'green' if values[i] >= 0.0 else 'red'
                values[i] = abs(values[i])
            log_scale = False

        else:
            print('Error: Unimplemented node size "%s"!' % self.context['nodeSize'])
            raise NotImplementedError

        # Showing the full graph is not feasible. Limit the view
        # to a sparse subgraph of about ~40 nodes.

        graph = dataset_tvg.topics(ts_min, ts_max)
        subgraph = graph.sparse_subgraph()

        # convert values to dictionary
        if isinstance(values, pytvg.Vector):
            values = values.as_dict()

        # convert values to logarithmic scale
        if log_scale:
            for i in values.keys():
                values[i] = max(math.log(values[i]) + 10.0, 1.0) if values[i] > 0.0 else 1.0

        if len(values.values()) != 0:
            # rescale the values of selected nodes to [0.0, 1.0]
            min_value = min(values.values())
            max_value = max(values.values())
            if min_value != max_value:
                for i in values.keys():
                    values[i] = (values[i] - min_value) / (max_value - min_value)
            else:
                for i in values.keys():
                    values[i] = 1.0

        nodes = []
        for i in subgraph.nodes():
            value = values[i]

            try:
                node = dataset_tvg.node_by_index(i)
            except KeyError:
                node = {}

            label = "Node %d" % i
            for key in ['label', 'norm', 'text', 'entity_name']:
                try:
                    label = node[key]
                except KeyError:
                    pass
                else:
                    break

            try:
                ne = node['NE']
                color = self.context['nodeTypes'][ne]['color']
            except KeyError:
                color = self.context['defaultColor']

            attrs = {
                'id':    i,
                'value': 0.2 + 0.8 * value,
                'label': label,
                'color': color,
                'nodeType': ne,
                'font':  { 'size': 5 + value * 35 }
            }

            if i in custom_colors:
                attrs['borderWidth'] = 2
                attrs['color'] = {
                    'background': custom_colors[i],
                    'border':     color
                }

            nodes.append(attrs)

        edges = []
        for i, w in zip(*subgraph.edges()):
            edges.append({'id': "%d-%d" % (i[0], i[1]), 'from': i[0], 'to': i[1], 'value': w})

        self.send_message_json(cmd='network_set', nodes=nodes, edges=edges)
        self.ts_min = ts_min
        self.ts_max = ts_max

    def check_for_new_articles(self):
        max_ts = dataset_tvg.lookup_le().ts
        if max_ts > self.latest_displayed_timestamp:
            self.latest_displayed_timestamp = max_ts
            self.send_message_json(cmd='updateTimeline', max=max_ts)

    def event_connected(self):
        print(self.address, 'connected')
        self.context = copy.deepcopy(default_context)
        self.ts_min = None
        self.ts_max = None

        # Set timeline min/max. The client can then seek to any position.
        data_ts_min = dataset_tvg.lookup_ge().ts
        data_ts_max = dataset_tvg.lookup_le().ts
        self.latest_displayed_timestamp = data_ts_max
        self.send_message_json(cmd='timeline_set_options', min=data_ts_min, max=data_ts_max)

        self.send_message_json(cmd='set_context', context=self.context)

    def event_message(self, data):
        msg = json.loads(data)
        context = self.context

        if msg['cmd'] == 'timeline_seek':
            self.timeline_seek(ts_min=int(msg['start']), ts_max=int(msg['end']))
            self.send_message_json(cmd='focus_timeline');
            return

        elif msg['cmd'] == 'save_custom_color':
            if 'flag' in msg:
                context['nodeTypes'][msg['flag']]['color'] = msg['color']
            else:
                context['defaultColor'] = msg['color']
            return

        elif msg['cmd'] == 'change_node_size':
            context['nodeSize'] = msg['value']

            self.timeline_seek()
            return

        elif msg['cmd'] == 'check_for_new_articles':
            self.check_for_new_articles()
            return

        print('Unimplemented command "%s"!' % msg['cmd'])
        raise NotImplementedError

    def event_close(self):
        pass

class PreloadTask(object):
    def __init__(self, tvg, step=86400000):
        self.tvg    = tvg                # dataset to preload
        self.time   = time.time() + 1.0  # next time to run
        self.ts_min = tvg.lookup_ge().ts # minimum timestamp
        self.ts_max = None               # maximum timestamp
        self.ts     = self.ts_min        # current timestamp
        self.step   = step               # step size
        self.refs   = []                 # references to preloaded results

    def run(self):
        if time.time() < self.time:
            return

        if self.ts_max is None:
            self.ts_max = self.tvg.lookup_le().ts

        if self.ts + self.step - 1 >= self.ts_max:
            self.time = time.time() + 600.0
            self.ts_max = None
            return

        print("Preloading interval %u-%u (%f%%)" % (self.ts, self.ts + self.step - 1,
              (self.ts - self.ts_min) * 100.0 / (self.ts_max - self.ts_min)))

        args = (self.ts, self.ts + self.step - 1)
        self.refs.append(self.tvg.sum_edges(*args))
        self.refs.append(self.tvg.count_edges(*args))
        self.refs.append(self.tvg.count_nodes(*args))

        self.time = time.time() + 0.5
        self.ts += self.step

if __name__ == "__main__":
    def cache_size(s):
        if s.endswith("K") or s.endswith("k"):
            s, mul = s[:-1], 1024
        elif s.endswith("M"):
            s, mul = s[:-1], 1024 * 1024
        elif s.endswith("G"):
            s, mul = s[:-1], 1024 * 1024 * 1024
        else:
            mul = 1
        try:
            return int(float(s) * mul)
        except ValueError:
            raise argparse.ArgumentTypeError("%r is not a valid cache size" % s)

    parser = argparse.ArgumentParser(description="TVG Explorer")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print debug information")
    parser.add_argument("--preload", action="store_true", help="Preload query results")
    parser.add_argument("--graph-cache", type=cache_size, help="Set graph cache size", default=0x10000000) # 256 MB
    parser.add_argument("--query-cache", type=cache_size, help="Set query cache size", default=0x10000000) # 256 MB
    parser.add_argument("config", help="Path to a configuration file")
    args = parser.parse_args()

    with open(args.config) as fp:
        config = json.load(fp)

    source = config.get('source', {})
    default_context['nodeTypes'] = config.get('nodeTypes', {})
    default_context['defaultColor'] = config.get('defaultColor', '#bf8080')
    default_context['nodeSize'] = config.get('nodeSize', 'eigenvector')

    if 'uri' in source:
        if 'database' not in source:
            raise RuntimeError("No database specified")
        if 'col_articles' not in source:
            raise RuntimeError("Article collection not specified")
        if 'article_id' not in source:
            raise RuntimeError("Article ID key not specified")
        if 'article_time' not in source:
            raise RuntimeError("Article time key not specified")
        if 'col_entities' not in source:
            raise RuntimeError("Entities collection not specified")
        if 'entity_doc' not in source:
            raise RuntimeError("Entities doc key not specified")
        if 'entity_sen' not in source:
            raise RuntimeError("Entities sen key not specified")
        if 'entity_ent' not in source:
            raise RuntimeError("Entities ent key not specified")
        if 'primary_key' not in source:
            raise RuntimeError("Primary key not specified")

        # Enforce sum_weights = False, otherwise the creation of WindowTopics will fail.
        source['sum_weights'] = False

        primary_key = source.pop('primary_key')
        mongodb = pytvg.MongoDB(**source)

        dataset_tvg = pytvg.TVG(positive=True, streaming=True)
        dataset_tvg.set_primary_key(primary_key)
        dataset_tvg.enable_mongodb_sync(mongodb, batch_size=256, cache_size=args.graph_cache)

    elif 'graph' in source:
        dataset_tvg = pytvg.TVG.load(source['graph'], positive=True, streaming=True)
        if 'nodes' in source:
            dataset_tvg.load_nodes_from_file(source['nodes'], source.get('attributes', None))

    else:
        raise RuntimeError("Config does not have expected format")

    dataset_tvg.enable_query_cache(cache_size=args.query_cache)
    dataset_tvg.verbosity = args.verbose

    server = SimpleWebSocketServer('', 8000, Client)

    tasks = []
    if args.preload:
        tasks.append(PreloadTask(dataset_tvg))

    while True:
        server.serveonce()
        for task in tasks:
            task.run()
