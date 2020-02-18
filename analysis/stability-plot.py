#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import collections
import datetime
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../libtvg"))
import pytvg

class Analyzer(object):
    def __init__(self, tvg, ranges):
        self.tvg = tvg
        self.ranges = ranges

    def run(self):
        self.times       = []
        self.sums        = collections.defaultdict(list)
        self.stabilities = collections.defaultdict(list)

        self.top_edges = self.tvg.sum_edges().top_edges(10, as_dict=True).keys()

        for (ts_min, ts_max) in self.ranges(self.tvg):
            self.analyze(ts_min, ts_max)

    def analyze(self, ts_min, ts_max):
        # Print the current time frame.
        dt_min = datetime.datetime.utcfromtimestamp(ts_min / 1000.0)
        dt_max = datetime.datetime.utcfromtimestamp(ts_max / 1000.0)
        print ("%s - %s" % (dt_min, dt_max))

        self.times.append(dt_min)

        # Track the importance of edges.
        graph = self.tvg.sum_edges(ts_min, ts_max)
        for edge in self.top_edges:
            self.sums[edge].append(graph[edge])

        # Track the stable edges (within the timeframe).
        graphs = dataset_tvg.sample_graphs(ts_min, ts_max, sample_width=(ts_max - ts_min) / 3)
        graphs = [g.normalize() for g in graphs]
        graph = pytvg.metric_stability_pareto(graphs, base=0.5)
        # alternatively, but much slower:
        # graph = pytvg.Graph.from_dict(pytvg.metric_stability_ratio(graphs))
        for edge in self.top_edges:
            self.stabilities[edge].append(graph[edge])

    def plot(self):
        ticks = range(len(self.times))

        # Plot importance of edges over time.
        plt.title("Importance of edges over time")
        for edge in self.top_edges:
            label = " - ".join(map(self.tvg.node_label, edge))
            plt.plot(range(len(self.times)), self.sums[edge], label=label)
        plt.xticks(ticks, [t.strftime("%d/%m/%y") for t in self.times])
        plt.ylabel("Importance")
        plt.xlabel("Time")
        plt.legend()
        plt.show()

        # Plot stability of edges over time.
        plt.title("Stability of edges over time")
        for edge in self.top_edges:
            label = " - ".join(map(self.tvg.node_label, edge))
            plt.plot(range(len(self.times)), self.stabilities[edge], label=label)
        plt.xticks(ticks, [t.strftime("%d/%m/%y") for t in self.times])
        plt.ylabel("Stability")
        plt.xlabel("Time")
        plt.legend()
        plt.show()

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

    def sample_ranges(w):
        def _datetime_to_unix(date):
            diff = date - datetime.datetime.utcfromtimestamp(0)
            return (diff.days * 86400000) + (diff.seconds * 1000) + (diff.microseconds // 1000)

        def _next_millisecond(ts, mul, off=0):
            return off + ((ts - off + mul) // mul) * mul

        def _next_week(ts, mul):
            return _next_millisecond(ts, mul * 1000 * 60 * 60 * 24 * 7,
                                     off=1000 * 60 * 60 * 24 * 4)

        def _next_month(ts, mul):
            date = datetime.datetime.utcfromtimestamp(ts / 1000.0)
            months = date.year * 12 + date.month - 1 + mul
            date = date.replace(year=months // 12, month=(months % 12) + 1, day=1,
                                hour=0, minute=0, second=0, microsecond=0)
            return _datetime_to_unix(date)

        def _next_quarter(ts, mul):
            date = datetime.datetime.utcfromtimestamp(ts / 1000.0)
            months = ((date.year * 12 + date.month - 1 + 3 * mul) // 3) * 3
            date = date.replace(year=months // 12, month=(months % 12) + 1, day=1,
                                hour=0, minute=0, second=0, microsecond=0)
            return _datetime_to_unix(date)

        def _next_year(ts, mul):
            date = datetime.datetime.utcfromtimestamp(ts / 1000.0)
            date = date.replace(year=date.year + mul, month=1, day=1,
                                hour=0, minute=0, second=0, microsecond=0)
            return _datetime_to_unix(date)

        next_func = _next_millisecond
        if w.endswith("s"):
            w, mul = w[:-1], 1000
        elif w.endswith("m"):
            w, mul = w[:-1], 1000 * 60
        elif w.endswith("h") or w.endswith("H"):
            w, mul = w[:-1], 1000 * 60 * 60
        elif w.endswith("d") or w.endswith("D"):
            w, mul = w[:-1], 1000 * 60 * 60 * 24
        elif w.endswith("w") or w.endswith("W"):
            next_func = _next_week
            w, mul = w[:-1], 1
        elif w.endswith("M"):
            next_func = _next_month
            w, mul = w[:-1], 1
        elif w.endswith("q") or w.endswith("Q"):
            next_func = _next_quarter
            w, mul = w[:-1], 1
        elif w.endswith("y") or w.endswith("Y"):
            next_func = _next_year
            w, mul = w[:-1], 1
        else:
            mul = 1

        try:
            mul = int(float(w) * mul)
        except ValueError:
            raise argparse.ArgumentTypeError("%r is not a valid sample width" % w)

        def _ranges(tvg):
            data_ts_min = tvg.lookup_ge().ts
            data_ts_max = tvg.lookup_le().ts
            assert data_ts_min <= data_ts_max

            ts_min = data_ts_min
            while ts_min <= data_ts_max:
                ts_max = min(next_func(ts_min, mul) - 1, data_ts_max)
                assert ts_max >= ts_min
                yield (ts_min, ts_max)
                ts_min = ts_max + 1

        return _ranges

    parser = argparse.ArgumentParser(description="Perform stability analysis on TVG")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print debug information")
    parser.add_argument("--graph-cache", type=cache_size, help="Set graph cache size", default=0x10000000) # 256 MB
    parser.add_argument("--query-cache", type=cache_size, help="Set query cache size", default=0x10000000) # 256 MB
    parser.add_argument("--width", dest='sample_ranges', type=sample_ranges, help="Set the window width", default=sample_ranges("1w"))
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

    analyzer = Analyzer(dataset_tvg, ranges=args.sample_ranges)
    analyzer.run()
    analyzer.plot()
