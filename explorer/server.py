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
    'colorMap': {
        'LOC': '#bf8080', # Location
        'ORG': '#b3a6c1', # Organisation
        'ACT': '#80b2e5', # Actor
        'DAT': '#80c7bf', # Date
        'TER': '#cbcbcb', # Term - everything except LOAD
    },
    'defaultColor': '#bf8080',
    'nodeWeight': 'power_iteration',
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
        context = self.context

        if self.window is None and (ts is None or width is None):
            print('Error: Cannot seek with partial information!')
            return

        if ts is None:
            ts = self.window.ts

        if self.window is None or (width is not None and self.window.width != width):
            self.window = dataset_tvg.WindowTopics(-width, 0)

        graph = self.window.update(ts)

        if context['nodeWeight'] == 'in_degrees':
            values = graph.in_degrees()

        elif context['nodeWeight'] == 'in_weights':
            values = graph.in_weights()

        elif context['nodeWeight'] == 'out_degrees':
            values = graph.out_degrees()

        elif context['nodeWeight'] == 'out_weights':
            values = graph.out_weights()

        elif context['nodeWeight'] == 'degree_anomalies':
            values = graph.degree_anomalies()

        elif context['nodeWeight'] == 'weight_anomalies':
            values = graph.weight_anomalies()

        elif context['nodeWeight'] == 'power_iteration':
            values, _ = graph.power_iteration(ret_eigenvalue=False)

        else:
            print('Error: Unimplemented node weight "%s"!' % context['nodeWeight'])
            raise NotImplementedError

        # Showing the full graph is not feasible. Limit the view
        # to a sparse subgraph of about ~40 nodes.
        subgraph = graph.sparse_subgraph()

        nodes = []
        for i in subgraph.nodes():
            value = values[i]
            value = max(math.log(value) + 10.0, 1.0) if value > 0.0 else 1.0

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
                color = self.context['colorMap'][ne]
            except KeyError:
                color = self.context['defaultColor']

            nodes.append({'id': i, 'value': value, 'label': label, 'color': color})

        edges = []
        for i, w in zip(*subgraph.edges()):
            edges.append({'id': "%d-%d" % (i[0], i[1]), 'from': i[0], 'to': i[1], 'value': w})

        self.send_message_json(cmd='network_set', nodes=nodes, edges=edges)
        self.ts = ts

    def event_connected(self):
        print(self.address, 'connected')
        self.context = copy.deepcopy(default_context)
        self.window  = None

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
                context['colorMap'][msg['flag']] = msg['color']
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

        dataset_tvg = pytvg.TVG.load(source['graph'], nodes=source['nodes'],
                                     positive=True, streaming=True)

    else:
        raise RuntimeError("Config does not have expected format")

    server = SimpleWebSocketServer('', 8000, Client)
    server.serveforever()
