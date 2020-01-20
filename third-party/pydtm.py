#!/usr/bin/python3
import collections
import subprocess
import tempfile
import datetime
import pymongo
import numpy as np
import copy
import sys
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_directory, "../libtvg"))
import pytvg

dtm_filename = os.path.join(current_directory, "dtm/dtm/main")
if not os.path.exists(dtm_filename):
    raise RuntimeError("Helper program not found")

class Topics(object):
    """
    Representation of a dynamic topic model.
    """

    def __init__(self, tvg, log_probs, ts_to_snapshot):
        self._tvg = tvg
        self._log_probs = log_probs
        self._ts_to_snapshot = ts_to_snapshot

    @property
    def num_topics(self):
        return self._log_probs.shape[0]

    @property
    def num_snapshots(self):
        return self._log_probs.shape[1]

    def ranked_list(self, topic, ts=None, snapshot=None, max_words=10):
        if topic < 0 or topic >= self._log_probs.shape[0]:
            raise IndexError("Topic out of range")

        if snapshot is None:
            snapshot = self._ts_to_snapshot(ts)
        if snapshot < 0:
            snapshot = 0
        elif snapshot >= self._log_probs.shape[1]:
            snapshot = self._log_probs.shape[1] - 1

        log_probs = self._log_probs[topic, snapshot, :]
        rank = sorted(range(len(log_probs)), key=lambda i: log_probs[i], reverse=True)
        if max_words is not None and max_words < len(rank):
            rank = rank[:max_words]

        return [self._tvg.node_by_index(i) for i in rank]

def dynamic_topic_model(tvg, ranges, num_topics=10):
    ranges = sorted(list(ranges))
    num_snapshots = len(ranges)

    if num_snapshots == 0:
        raise ValueError("At least one range required")
    if num_topics > 999:
        raise ValueError("More than 999 topics are not supported")

    def ts_to_snapshot(ts):
        if ts is None:
            return 0
        for i, (ts_min, ts_max) in enumerate(ranges):
            if ts_min <= ts and ts <= ts_max:
                return i
        raise RuntimeError("Snapshot not found")

    ts_min = min([t for t, _ in ranges])
    ts_max = max([t for _, t in ranges])

    counts = [0] * num_snapshots
    last_snapshot = 0

    with tempfile.TemporaryDirectory() as tempdir:

        with open(os.path.join(tempdir, "input-mult.dat"), 'w') as fp:
            for doc in tvg.documents(ts_min, ts_max):
                frequency = doc.nodes()
                fp.write("%d %s\n" % (len(frequency), " ".join(["%d:%d" % (k, v) for k, v in frequency.items()])))

                snapshot = ts_to_snapshot(doc.ts)
                assert snapshot >= last_snapshot
                counts[snapshot] += 1
                last_snapshot = snapshot

        with open(os.path.join(tempdir, "input-seq.dat"), 'w') as fp:
            fp.write("%d\n" % (num_snapshots,))
            for count in counts:
                fp.write("%d\n" % (count,))

        subprocess.check_call([dtm_filename,
            "--ntopics=%d" % (num_topics,),
            "--mode=fit",
            "--rng_seed=0",
            "--initialize_lda=true",
            "--corpus_prefix=%s" % (os.path.join(tempdir, "input"),),
            "--outname=%s" % (os.path.join(tempdir, "output"),),
            "--top_chain_var=0.005",
            "--alpha=0.01",
            "--lda_sequence_min_iter=6",
            "--lda_sequence_max_iter=20",
            "--lda_max_em_iter=10",
            ])

        log_probs = []
        for i in range(num_topics):
            data = np.loadtxt(os.path.join(tempdir, "output/lda-seq/topic-%03d-var-e-log-prob.dat" % (i,)))
            assert len(data.shape) == 1
            data = data.reshape((-1, num_snapshots))
            log_probs.append(data.T)

        log_probs = np.stack(log_probs)
        # log_probs[topic][snapshot][term]

        """
        num_documents = sum(counts)
        data = np.loadtxt(os.path.join(tempdir, "output/lda-seq/gam.dat"))
        data = data.reshape((num_documents, num_topics))
        # data[doc][...]
        """

        return Topics(tvg=tvg,
                      log_probs=log_probs,
                      ts_to_snapshot=ts_to_snapshot)

if __name__ == '__main__':
    source = {
        "uri": "mongodb://thabit:27021",
        "database": "AmbiverseNewsAnnotated",
        "col_articles": "c02_RawArticles",
        "article_id": "_id",
        "article_time": "pub",
        "col_entities": "c11_selectedDocumentEntities",
        "entity_doc": "docID",
        "entity_sen": "senDocID",
        "entity_ent": "NE;norm;label;covText",
        "load_nodes": True,
        "max_distance": 5,
        "primary_key": "NE;norm"
    }

    primary_key = source.pop('primary_key')
    mongodb = pytvg.MongoDB(**source)

    tvg = pytvg.TVG(positive=True, streaming=True)
    tvg.set_primary_key(primary_key)
    tvg.enable_mongodb_sync(mongodb, batch_size=256, cache_size=0x10000000)

    ts_min = tvg.lookup_ge().ts
    ts_max = tvg.lookup_le().ts

    for doc in tvg.documents(limit=10):
        print ("ts = ", doc.ts)

    ranges = [
        (ts_min + 86400000 * 0, ts_min + 86400000 * 1 - 1),
        (ts_min + 86400000 * 1, ts_min + 86400000 * 2 - 1),
        (ts_min + 86400000 * 2, ts_min + 86400000 * 3 - 1),
    ]

    t = dynamic_topic_model(tvg, ranges)

    print ("topics = ", t.num_topics)
    print ("snapshots = ", t.num_snapshots)

    for i in range(t.num_topics):
        print ("Topic %d: %s" % (i, t.ranked_list(i)))
