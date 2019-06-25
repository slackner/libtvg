#!/usr/bin/python2
from collections import defaultdict
import datetime
import numpy as np
import sys
import re

labels = []

# Load the node labels - this will be useful for debugging or when printing
# the result.
with open("news_nodelist.tsv") as f:
    for lineno, line in enumerate(f):
        if lineno % 10000 == 0:
            sys.stderr.write("Processing line %d\n" % lineno)

        line = line.rstrip()
        if line == "": continue
        if line.startswith("id"): continue
        values  = line.split("\t")

        assert len(values) >= 4
        assert values[1] in "LOADTSP"

        id1 	= int(values[0])
        label 	= values[3]
        type1   = { 'L': 'LOC', 'O': 'ORG', 'A': 'ACT',
                    'D': 'DAT', 'T': 'TER' }[values[1]]
        # values[2] -> wikidata ID (empty if type=T)

        label = label.replace("\t", " ")
        type1 = type1.replace("\t", " ")
        labels.append((id1, label, type1))

labels.sort(key=lambda x: x[0])

with open("result.nodes", "w") as g:
    for id1, label, type1 in labels:
        g.write("%d\t%s\t%s\n" % (id1, label, type1))
