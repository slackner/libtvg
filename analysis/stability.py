#!/usr/bin/python3
import numpy as np
import datetime
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../libtvg"))
import pytvg

def analyze_stability(tvg, width):
    data_ts_min = tvg.lookup_ge().ts
    data_ts_max = tvg.lookup_le().ts
    assert data_ts_min <= data_ts_max

    for ts_min in range(data_ts_min, data_ts_max, width):
        ts_max = ts_min + width

        # Print the current time frame.
        dt_min = datetime.datetime.utcfromtimestamp(ts_min / 1000.0)
        dt_max = datetime.datetime.utcfromtimestamp(ts_max / 1000.0)
        print ("%s - %s" % (dt_min, dt_max))

        # Print the top edges.
        print ("Top edges:")
        graph = tvg.sum_edges(ts_min, ts_max)
        edges = graph.top_edges(10, as_dict=True)
        for (i, j), w in edges.items():
            print ("Edge %s - %s: %f" % (tvg.node_label(i), tvg.node_label(j), w))

        # Print the stable edges (within the timeframe).
        print ("Stable edges:")
        graphs = dataset_tvg.sample_graphs(ts_min, ts_max, sample_width=(ts_max - ts_min) / 3)
        graphs = [g.normalize() for g in graphs]
        graph = pytvg.metric_stability_pareto(graphs, base=0.5)
        edges = graph.top_edges(10, as_dict=True)
        for (i, j), w in edges.items():
            print ("Edge %s - %s: %f" % (tvg.node_label(i), tvg.node_label(j), w))

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

    def sample_width(w):
        if w.endswith("s"):
            w, mul = w[:-1], 1000
        elif w.endswith("m"):
            w, mul = w[:-1], 1000 * 60
        elif w.endswith("h") or w.endswith("H"):
            w, mul = w[:-1], 1000 * 60 * 60
        elif w.endswith("d") or w.endswith("D"):
            w, mul = w[:-1], 1000 * 60 * 60 * 24
        elif w.endswith("w") or w.endswith("W"):
            w, mul = w[:-1], 1000 * 60 * 60 * 24 * 7
        elif w.endswith("M"):
            w, mul = w[:-1], 1000 * 60 * 60 * 24 * 30.5 # FIXME
        else:
            mul = 1
        try:
            return int(float(w) * mul)
        except ValueError:
            raise argparse.ArgumentTypeError("%r is not a valid sample width" % w)

    parser = argparse.ArgumentParser(description="Perform stability analysis on TVG")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print debug information")
    parser.add_argument("--graph-cache", type=cache_size, help="Set graph cache size", default=0x10000000) # 256 MB
    parser.add_argument("--query-cache", type=cache_size, help="Set query cache size", default=0x10000000) # 256 MB
    parser.add_argument("--width", type=sample_width, help="Set the window width", default=1000 * 60 * 60 * 24 * 7) # 1 week
    parser.add_argument("config", help="Path to a configuration file")
    args = parser.parse_args()

    with open(args.config) as fp:
        config = json.load(fp)

    source = config.get('source', {})

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

    analyze_stability(dataset_tvg, width=args.width)
