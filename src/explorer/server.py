#!/usr/bin/python3
from SimpleWebSocketServer import SimpleWebSocketServer
from SimpleWebSocketServer import WebSocket
import numpy as np
import traceback
import argparse
import math
import json
import sys
import re
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../libtvg"))
import pytvg

label_regex = re.compile("^(.*) \(([LOADTSP])\)$")
clients = []

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

    def timeline_seek(self, ts):
        if ts == self.ts:
            return
        graph = self.window.update(ts)

        # FIXME: Let user choose the node size feature.
        # values = graph.in_degrees()
        # values = graph.in_weights()
        # values = graph.out_degrees()
        # values = graph.out_weights()
        # values = graph.degree_anomalies()
        # values = graph.weight_anomalies()
        values, _ = graph.power_iteration(ret_eigenvalue=False)

        def node_attributes(i):
            value = values[i]
            value = max(math.log(value) + 10.0, 1.0) if value > 0.0 else 1.0

            try:
                label = dataset_labels[i]
            except KeyError:
                label = "Node %d" % i

            m = label_regex.match(label)
            if m is not None:
                label = m.group(1)
                color = { 'L': "#bf8080",
                          'O': "#b3a6c1",
                          'A': "#80b2e5",
                          'D': "#80c7bf",
                          'T': "#cbcbcb",
                          'S': "#95cb8f",
                          'P': "#ebb14b" }[m.group(2)]
            else:
                color = "#cccccc"

            return {'value': value, 'label': label, 'color': color}

        visjs = graph.encode_visjs(node_attributes)
        self.send_message_json(**visjs)
        self.ts = ts

    def event_connected(self):
        print(self.address, 'connected')

        self.window = dataset_tvg.WindowDecay(600, 0.93)
        self.window.eps = 1e-6
        self.ts = None

        # Set timeline min/max. The client can then seek to any position.
        min_ts = dataset_tvg.lookup_ge().ts
        max_ts = dataset_tvg.lookup_le().ts
        self.send_message_json(cmd='timeline_set_options', min=min_ts, max=max_ts)

    def event_message(self, data):
        msg = json.loads(data)

        if msg['cmd'] == 'timeline_seek':
            self.timeline_seek(msg['time'])
            return

        print("Unimplemented command '%s'!" % msg['cmd'])
        raise NotImplementedError

    def event_close(self):
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TVG Explorer")
    parser.add_argument('--labels', help="Path to labels")
    parser.add_argument('filename', help="Path to a dataset")
    args = parser.parse_args()

    dataset_tvg    = pytvg.TVG.load(args.filename, positive=True, streaming=True)
    dataset_labels = pytvg.Labels.load(args.labels) if args.labels else {}

    server = SimpleWebSocketServer('', 8000, Client)
    server.serveforever()
