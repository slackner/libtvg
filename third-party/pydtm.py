#!/usr/bin/python3
import collections
import subprocess
import tempfile
import datetime
import pymongo
import numpy as np
import copy
import os

dtm_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dtm/dtm/main")
if not os.path.exists(dtm_filename):
    raise RuntimeError("Helper program not found")

# Convert datetime object to unix timestamp
def datetime_to_unix(date):
    diff = date - datetime.datetime.utcfromtimestamp(0)
    return (diff.days * 86400000) + (diff.seconds * 1000) + (diff.microseconds // 1000)

# Convert unix timestamp to datetime object
def unix_to_datetime(date):
    return datetime.datetime.utcfromtimestamp(date / 1000)

class Entity(object):
    """
    Entity object representation (e.g., term or word from a document).
    """

    def __init__(self, **kwargs):
        self._index = None
        self._data = {}
        for k, v in kwargs.items():
            self._data[k] = v

    def __repr__(self):
        return "Entity(%s)" % self.as_dict().__repr__()

    @property
    def index(self):
        return self._index

    @property
    def text(self):
        return self._data['text']

    def __setitem__(self, key, value):
        self._data[key] = value

    def __getitem__(self, key):
        return self._data[key]

    def as_dict(self):
        return copy.deepcopy(self._data)

class Document(object):
    """
    Single document from a collection.
    """

    def __init__(self, collection, id, ts):
        self._collection = collection
        self._id = id
        self._ts = ts
        self._entities = None

    @property
    def ts(self):
        return self._ts

    @property
    def id(self):
        return self._id

    @property
    def entities(self):
        if self._entities is None:
            self._entities = self._collection.entities(self._id)
        return self._entities

    def nodes(self):
        return np.unique(self.entities)

    @property
    def entity_ids(self):
        return [entity.index for entity in self.entities]

class Topics(object):
    """
    Representation of a dynamic topic model.
    """

    def __init__(self, collection, log_probs, ts_to_snapshot):
        self._collection = collection
        self._log_probs = log_probs
        self._ts_to_snapshot = ts_to_snapshot

    @property
    def num_topics(self):
        return self._log_probs.shape[0]

    @property
    def num_snapshots(self):
        return self._log_probs.shape[1]

    def ranked_list(self, topic, ts=0, snapshot=None, max_words=10):
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

        return [self._collection.entity_by_index(i) for i in rank]

class Collection(object):
    """
    Document collection (streamed from a MongoDB).
    """

    def __init__(self, uri, database, col_articles, article_id, article_time,
                 col_entities, entity_doc, entity_sen, entity_ent,
                 filter_key=None, filter_value=None, primary_key=None, **kwargs):
        self._client = pymongo.MongoClient(uri)
        self._articles      = self._client[database][col_articles]
        self._article_id    = article_id
        self._article_time  = article_time
        self._entities      = self._client[database][col_entities]
        self._entity_doc    = entity_doc
        self._entity_sen    = entity_sen
        self._entity_ent    = entity_ent
        self._filter_key    = filter_key
        self._filter_value  = filter_value

        if primary_key is None:
            self._primary_key = []
        elif isinstance(primary_key, list):
            self._primary_key = copy.deepcopy(primary_key)
        else:
            self._primary_key = primary_key.split(";")

        self._entity_by_key   = {}
        self._entity_by_index = {}
        self._entity_index    = 0

    def Entity(self, **kwargs):
        key = []
        for k in self._primary_key:
            key.append(kwargs.get(k, None))
        key = tuple(key)

        try:
            return self._entity_by_key[key]
        except KeyError:
            pass

        # FIXME: Do we really want to load all attributes?
        entity = Entity(**kwargs)
        entity._index = self._entity_index
        self._entity_by_key[key] = entity
        self._entity_by_index[entity._index] = entity
        self._entity_index += 1
        return entity

    def entity_by_index(self, index):
        return self._entity_by_index[index]

    def entities(self, doc_id):
        result = []
        for kwargs in self._entities.find({self._entity_doc: doc_id}):
            result.append(self.Entity(**kwargs))
        return result

    def documents(self, ts_min=None, ts_max=None, direction=1, limit=None):
        query = {}
        condition = {}
        if ts_min is not None:
            condition['$gte'] = unix_to_datetime(ts_min)
        if ts_max is not None:
            condition['$lte'] = unix_to_datetime(ts_max)
        if len(condition) > 0:
            query[self._article_time] = condition
        if self._filter_key is not None:
            query[self._filter_key] = self._filter_value

        query = self._articles.find(query)
        if direction >= 0:
            query = query.sort([(self._article_time, pymongo.ASCENDING),
                                (self._article_id,   pymongo.ASCENDING)])
        else:
            query = query.sort([(self._article_time, pymongo.DESCENDING),
                                (self._article_id,   pymongo.DESCENDING)])
        if limit is not None:
            query = query.limit(limit)

        for doc in query:
            doc_id = doc[self._article_id]
            doc_ts = datetime_to_unix(doc[self._article_time])
            yield Document(collection=self, id=doc_id, ts=doc_ts)

    def lookup_ge(self, ts=None):
        for doc in self.documents(ts_min=ts, direction=1, limit=1):
            return doc
        return None

    def lookup_le(self, ts=None):
        for doc in self.documents(ts_max=ts, direction=-1, limit=1):
            return doc
        return None

    def topics(self, ts_min=None, ts_max=None, step=86400000, offset=0, num_topics=10):
        if step <= 0:
            raise ValueError("Step should be > 0")
        if num_topics > 999:
            raise ValueError("More than 999 topics are not supported")

        if ts_min is None:
            doc = self.lookup_ge()
            if doc is None:
                raise RuntimeError("Dataset is empty")
            ts_min = doc.ts

        if ts_max is None:
            doc = self.lookup_le()
            if doc is None:
                raise RuntimeError("Dataset is empty")
            ts_max = doc.ts

        offset -= (offset // step) * step

        if ts_max < offset:
            num_snapshots = 1
            ts_to_snapshot = lambda ts: 0
        elif ts_min < offset:
            num_snapshots = (ts_max - offset) // step + 2
            ts_to_snapshot = lambda ts: (ts - offset) // step + 1 if doc.ts >= offset else 0
        else:
            num_snapshots = (ts_max - offset) // step - \
                            (ts_min - offset) // step + 1
            ts_to_snapshot = lambda ts: (ts - offset) // step - (ts_min - offset) // step

        counts = [0] * num_snapshots
        last_snapshot = 0

        tempdir = tempfile.TemporaryDirectory()

        with open(os.path.join(tempdir.name, "input-mult.dat"), 'w') as fp:
            for doc in self.documents(ts_min, ts_max):
                frequency = collections.Counter(doc.entity_ids)
                fp.write("%d %s\n" % (len(frequency), " ".join(["%d:%d" % (k, v) for k, v in frequency.items()])))

                snapshot = ts_to_snapshot(doc.ts)
                assert snapshot >= last_snapshot
                counts[snapshot] += 1
                last_snapshot = snapshot

        with open(os.path.join(tempdir.name, "input-seq.dat"), 'w') as fp:
            fp.write("%d\n" % (num_snapshots,))
            for count in counts:
                fp.write("%d\n" % (count,))

        try:
            subprocess.check_call([dtm_filename,
                "--ntopics=%d" % (num_topics,),
                "--mode=fit",
                "--rng_seed=0",
                "--initialize_lda=true",
                "--corpus_prefix=%s" % (os.path.join(tempdir.name, "input"),),
                "--outname=%s" % (os.path.join(tempdir.name, "output"),),
                "--top_chain_var=0.005",
                "--alpha=0.01",
                "--lda_sequence_min_iter=6",
                "--lda_sequence_max_iter=20",
                "--lda_max_em_iter=10",
                ])

            log_probs = []
            for i in range(num_topics):
                data = np.loadtxt(os.path.join(tempdir.name, "output/lda-seq/topic-%03d-var-e-log-prob.dat" % (i,)))
                assert len(data.shape) == 1
                data = data.reshape((-1, num_snapshots))
                log_probs.append(data.T)

            log_probs = np.stack(log_probs)
            # log_probs[topic][snapshot][term]

            """
            num_documents = sum(counts)
            data = np.loadtxt(os.path.join(tempdir.name, "output/lda-seq/gam.dat"))
            data = data.reshape((num_documents, num_topics))
            # data[doc][...]
            """

            return Topics(collection=self,
                          log_probs=log_probs,
                          ts_to_snapshot=ts_to_snapshot)
        finally:
            tempdir.cleanup()

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

    c = Collection(**source)

    ts_min = c.lookup_ge().ts
    ts_max = c.lookup_le().ts

    for doc in c.documents(limit=10):
        print ("ts = ", doc.ts)

    t = c.topics(ts_min=ts_min, ts_max=ts_min + 86400000 * 30 - 1, offset=ts_min)

    print ("topics = ", t.num_topics)
    print ("snapshots = ", t.num_snapshots)

    for i in range(t.num_topics):
        print ("Topic %d: %s" % (i, t.ranked_list(t)))
