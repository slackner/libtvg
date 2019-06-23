#!/usr/bin/python3
from SimpleWebSocketServer import SimpleWebSocketServer
from SimpleWebSocketServer import WebSocket
import numpy as np
import traceback
import argparse
import math
import json
import sys
import os
import copy

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../libtvg"))
import pytvg

clients = []
default_context = {
    'nodeTypes': {
        'LOC': {
            'title': 'location',
            'color': '#bf8080',
            'class': 'mr-3 far fa-compass',
        },
        'ORG': {
            'title': 'organisation',
            'color': '#b3a6c1',
            'class': 'mr-3 fas fa-globe',
        },
        'ACT': {
            'title': 'actor',
            'color': '#80b2e5',
            'class': 'mr-3 far fa-user-circle',
        },
        'DAT': {
            'title': 'date',
            'color': '#80c7bf',
            'class': 'mr-3 far fa-clock',
        },
        'TER': {
            'title': 'term',
            'color': '#cbcbcb',
            'class': 'mr-3 fas fa-exclamation-circle',
        },
    },
    'defaultColor': '#bf8080',
    'nodeWeight': 'eigenvector',
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

    def timeline_seek(self, ts=None, width=None):
        """
        Update the graph view. If ts is given, seek to the given timestamp.
        Note that the timestamp specifies the right end of the window. If
        width is given, also update the width of the window.
        """
        if self.window is None and (ts is None or width is None):
            print('Error: Cannot seek with partial information!')
            return

        if ts is None:
            ts = self.window.ts

        if self.window is None or (width is not None and self.window.width != width):
            self.window = dataset_tvg.Window(-width, 0)
            self.nodes  = self.window.SumEdges()
            self.edges  = self.window.Topics()

        log_scale = True
        custom_colors = {}

        if self.context['nodeWeight'] == 'in_degrees':
            self.window.update(ts)
            values = self.nodes.result.in_degrees()

        elif self.context['nodeWeight'] == 'in_weights':
            self.window.update(ts)
            values = self.nodes.result.in_weights()

        elif self.context['nodeWeight'] == 'out_degrees':
            self.window.update(ts)
            values = self.nodes.result.out_degrees()

        elif self.context['nodeWeight'] == 'out_weights':
            self.window.update(ts)
            values = self.nodes.result.out_weights()

        elif self.context['nodeWeight'] == 'degree_anomalies':
            self.window.update(ts)
            values = self.nodes.result.degree_anomalies()

        elif self.context['nodeWeight'] == 'weight_anomalies':
            self.window.update(ts)
            values = self.nodes.result.weight_anomalies()

        elif self.context['nodeWeight'] == 'eigenvector':
            self.window.update(ts)
            values, _ = self.nodes.result.power_iteration(tolerance=1e-3, ret_eigenvalue=False)

        elif self.context['nodeWeight'] == 'stable_nodes':
            values = self.nodes.metric_stability(ts, self.window.width * 3)
            for i in values.keys():
                values[i] = -values[i]
            log_scale = False

        elif self.context['nodeWeight'] == 'entropy':
            values = self.nodes.metric_entropy(ts, self.window.width * 3)
            log_scale = False

        elif self.context['nodeWeight'] == 'entropy_local':
            values = self.nodes.metric_entropy_local(ts, self.window.width * 3)
            log_scale = False

        elif self.context['nodeWeight'] == 'entropy_2d':
            values = self.nodes.metric_entropy_2d(ts, self.window.width * 3)
            log_scale = False

        elif self.context['nodeWeight'] == 'trend':
            values = self.nodes.metric_trend(ts, self.window.width * 3)
            for i in values.keys():
                custom_colors[i] = 'green' if values[i] >= 0.0 else 'red'
                values[i] = abs(values[i])
            log_scale = False

        else:
            print('Error: Unimplemented node weight "%s"!' % self.context['nodeWeight'])
            raise NotImplementedError

        # Showing the full graph is not feasible. Limit the view
        # to a sparse subgraph of about ~40 nodes.

        if self.window.ts != ts:
            self.window.update(ts)

        subgraph = self.edges.result.sparse_subgraph()

        # convert values to dictionary
        if isinstance(values, pytvg.Vector):
            values = values.as_dict()

        # convert values to logarithmic scale
        if log_scale:
            for i in values.keys():
                values[i] = max(math.log(values[i]) + 10.0, 1.0) if values[i] > 0.0 else 1.0

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
            for key in ['label', 'norm', 'text']:
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
        self.ts = ts

    def event_connected(self):
        print(self.address, 'connected')
        self.context = copy.deepcopy(default_context)
        self.window  = None
        self.nodes   = None
        self.edges   = None

        # Set timeline min/max. The client can then seek to any position.
        min_ts = dataset_tvg.lookup_ge().ts
        max_ts = dataset_tvg.lookup_le().ts
        self.send_message_json(cmd='timeline_set_options', min=min_ts, max=max_ts)

        self.send_message_json(cmd='set_context', context=self.context)

    def event_message(self, data):
        msg = json.loads(data)
        context = self.context

        if msg['cmd'] == 'timeline_seek':
            self.timeline_seek(ts=int(msg['end']), width=max(int(msg['end'] - msg['start']), 1000))
            self.send_message_json(cmd='focus_timeline');
            return

        elif msg['cmd'] == 'recolor_graph_nodes':
            if 'flag' in msg:
                context['nodeTypes'][msg['flag']]['color'] = msg['color']
            else:
                context['defaultColor'] = msg['color']

            self.timeline_seek()
            return

        elif msg['cmd'] == 'change_node_weight':
            context['nodeWeight'] = msg['value']

            self.timeline_seek()
            return

        print('Unimplemented command "%s"!' % msg['cmd'])
        raise NotImplementedError

    def event_close(self):
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TVG Explorer")
    parser.add_argument("config", help="Path to a configuration file")
    args = parser.parse_args()

    with open(args.config) as fp:
        config = json.load(fp)

    source = config['source']

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
        dataset_tvg.enable_mongodb_sync(mongodb, batch_size=256, cache_size=0x10000000) # 256 MB

    elif 'graph' in source:
        if 'nodes' not in source:
            source['nodes'] = None

        dataset_tvg = pytvg.TVG.load(source['graph'], nodes=source['nodes'], positive=True, streaming=True)

    else:
        raise RuntimeError("Config does not have expected format")

    server = SimpleWebSocketServer('', 8000, Client)
    server.serveforever()
