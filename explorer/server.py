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
    'windowWidth': 600000,
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

    def timeline_seek(self, ts): # ts = x-value on timeline
        context = self.context
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
            print('Unimplemented node weight "%s"!' % context['nodeWeight'])
            raise NotImplementedError

        def node_attributes(i):
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
                color = context['colorMap'][ne]
            except KeyError:
                color = context['defaultColor']

            return {'value': value, 'label': label, 'color': color}

        visjs = graph.encode_visjs(node_attributes)
        self.send_message_json(**visjs)
        self.ts = ts

    def event_connected(self):
        print(self.address, 'connected')
        self.context = copy.deepcopy(default_context)

        self.window = dataset_tvg.WindowDecay(self.context['windowWidth'], log_beta=np.log(0.93) / 1000.0)
        self.window.eps = 1e-6
        self.ts = None

        # Set timeline min/max. The client can then seek to any position.
        min_ts = dataset_tvg.lookup_ge().ts
        max_ts = dataset_tvg.lookup_le().ts
        self.send_message_json(cmd='timeline_set_options', min=min_ts, max=max_ts)

        self.send_message_json(cmd='set_context', context=self.context)

    def event_message(self, data):
        msg = json.loads(data)
        context = self.context

        if msg['cmd'] == 'timeline_seek':
            self.timeline_seek(msg['time'])
            self.send_message_json(cmd='focus_timeline');
            return

        elif msg['cmd'] == 'recolor_graph_nodes':
            if 'flag' in msg:
                context['colorMap'][msg['flag']] = msg['color']
            else:
                context['defaultColor'] = msg['color']

            self.timeline_seek(self.ts)
            return

        elif msg['cmd'] == 'change_node_weight':
            context['nodeWeight'] = msg['value']
            self.timeline_seek(self.ts)
            return

        print('Unimplemented command "%s"!' % msg['cmd'])
        raise NotImplementedError

    def event_close(self):
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TVG Explorer")

    parser.add_argument("--database",     default="AmbiverseNewsAnnotated",       help="Name of the database")
    parser.add_argument("--col_articles", default="c02_RawArticles",              help="Name of the articles collection")
    parser.add_argument("--article_id",   default="_id",                          help="Name of the article ID key")
    parser.add_argument("--article_time", default="pub",                          help="Name of the article time key")
    parser.add_argument("--col_entities", default="c11_selectedDocumentEntities", help="Name of the entities collection")
    parser.add_argument("--entity_doc",   default="docID",                        help="Name of the entity doc key")
    parser.add_argument("--entity_sen",   default="senDocID",                     help="Name of the entity sen key")
    parser.add_argument("--entity_ent",   default="NE;norm;label;covText",        help="Name of the entity ent key")
    parser.add_argument("--max_distance", default=5, type=int,                    help="Maximum distance of mentions")
    parser.add_argument("--primary_key",  default="NE;norm",                      help="Nodes primary key")

    parser.add_argument("--nodes",  help="Path to nodes")
    parser.add_argument("source",   help="Path/URI to a dataset")
    args = parser.parse_args()

    if args.source.startswith("mongodb://") or args.source.startswith("mongodb+srv://"):
        mongodb = pytvg.MongoDB(args.source, args.database, args.col_articles,
                                args.article_id, args.article_time, args.col_entities,
                                args.entity_doc, args.entity_sen, args.entity_ent,
                                load_nodes=True, max_distance=args.max_distance)

        dataset_tvg = pytvg.TVG(positive=True, streaming=True)
        dataset_tvg.set_primary_key(args.primary_key)
        dataset_tvg.enable_mongodb_sync(mongodb, batch_size=256, cache_size=0x10000000) # 256 MB

    else:
        dataset_tvg    = pytvg.TVG.load(args.source, nodes=args.nodes, positive=True, streaming=True)

    server = SimpleWebSocketServer('', 8000, Client)
    server.serveforever()
