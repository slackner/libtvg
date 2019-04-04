#!/usr/bin/python2
from collections import defaultdict
import datetime
import numpy as np
import sys
import re

ts_to_str = {}
edges = {}

# For each line in the source file, determine the (reduced) timestamp.
# Then append the line to the bucket and perform a bucket sort.
with open("news_edgelist.tsv") as f:
    for lineno, line in enumerate(f):
        if lineno % 10000 == 0:
            sys.stderr.write("Processing line %d\n" % lineno)

        line = line.rstrip()
        if line == "": continue
        if line.startswith("id1\t"): continue
        values  = line.split("\t")

        assert len(values) >= 8
        assert values[2] in "LOADTSP"
        assert values[3] in "LOADTSP"
        assert values[4].startswith("2016-")

        id1     = int(values[0])
        id2     = int(values[1])
        # values[2] -> type1
        # values[3] -> type2
        date    = datetime.datetime.strptime(values[4], "%Y-%m-%d %H:%M:%S")
        article = int(values[5])
        outlet  = values[6]
        dist    = int(values[7])

        ts      = int(date.strftime("%s")) * 1000 # sec -> msec
        ts_str  = values[4]

        date = date.timetuple()
        assert date.tm_year == 2016
        if outlet == "WP" and date.tm_yday == 231 and date.tm_hour in [16, 17, 18]:
            sys.stderr.write("Filtered article: %s\n" % line)
            continue # Corrupted
        if outlet == "WP" and date.tm_yday == 241 and date.tm_hour in [12, 13]:
            sys.stderr.write("Filtered article: %s\n" % line)
            continue # Corrupted

        try:
            assert ts_to_str[ts] == ts_str
        except KeyError:
            ts_to_str[ts] = ts_str

        if not edges.has_key(ts):
            edges[ts] = {}
        if not edges[ts].has_key(article):
            edges[ts][article] = defaultdict(float)
        if not edges[ts][article].has_key((id1, id2)):
            edges[ts][article][(id1, id2)] = []
        edges[ts][article][(id1, id2)].append(np.exp(-dist))

with open("result-sum.graph", "w") as g:
    sorted_ts = sorted(edges.keys())

    g.write("# Generated from news_edgelist.tsv\n")
    g.write("# Start: %s\n" % ts_to_str[sorted_ts[0]])
    g.write("# End:   %s\n" % ts_to_str[sorted_ts[-1]])

    for ts in sorted_ts:
        merged = defaultdict(float)

        for article in edges[ts].keys():
            for (id1, id2), weights in edges[ts][article].iteritems():
                merged[(id1, id2)] += np.sum(weights)

        for (id1, id2), weight in sorted(merged.items()):
            g.write("%d\t%d\t%f\t%s\n" % (id1, id2, weight, ts))

with open("result-mean.graph", "w") as g:
    sorted_ts = sorted(edges.keys())

    g.write("# Generated from news_edgelist.tsv\n")
    g.write("# Start: %s\n" % ts_to_str[sorted_ts[0]])
    g.write("# End:   %s\n" % ts_to_str[sorted_ts[-1]])

    for ts in sorted_ts:
        merged = defaultdict(float)

        for article in edges[ts].keys():
            # If the same article contains multiple (id1, id2) - co-occurrences,
            # then use the average instead of the sum.
            for (id1, id2), weights in edges[ts][article].iteritems():
                merged[(id1, id2)] += np.mean(weights)

        for (id1, id2), weight in sorted(merged.items()):
            g.write("%d\t%d\t%f\t%s\n" % (id1, id2, weight, ts))

with open("result-max.graph", "w") as g:
    sorted_ts = sorted(edges.keys())

    g.write("# Generated from news_edgelist.tsv\n")
    g.write("# Start: %s\n" % ts_to_str[sorted_ts[0]])
    g.write("# End:   %s\n" % ts_to_str[sorted_ts[-1]])

    for ts in sorted_ts:
        merged = defaultdict(float)

        for article in edges[ts].keys():
            # If the same article contains multiple (id1, id2) - co-occurrences,
            # use the maximum instead of the sum.
            for (id1, id2), weights in edges[ts][article].iteritems():
                merged[(id1, id2)] += np.max(weights)

        for (id1, id2), weight in sorted(merged.items()):
            g.write("%d\t%d\t%f\t%s\n" % (id1, id2, weight, ts))

with open("result-norm.graph", "w") as g:
    sorted_ts = sorted(edges.keys())

    g.write("# Generated from news_edgelist.tsv\n")
    g.write("# Start: %s\n" % ts_to_str[sorted_ts[0]])
    g.write("# End:   %s\n" % ts_to_str[sorted_ts[-1]])

    for ts in sorted_ts:
        merged = defaultdict(float)

        for article in edges[ts].keys():
            # The weight of a single article is limited to 1.
            norm = np.sum([np.sum(weights) for weights in edges[ts][article].values()])
            for (id1, id2), weights in edges[ts][article].iteritems():
                merged[(id1, id2)] += np.sum(weights) / norm

        for (id1, id2), weight in sorted(merged.items()):
            g.write("%d\t%d\t%f\t%s\n" % (id1, id2, weight, ts))
