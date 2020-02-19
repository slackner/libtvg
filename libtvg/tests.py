#!/usr/bin/env python3
import collections
import numpy as np
import datetime
import unittest
import mockupdb
import tempfile
import gc
import re
import math
import sys
import os

# Ancient versions of mockupdb ship their own bson library.
try:
    bson = mockupdb._bson
except AttributeError:
    import bson

try:
    from pytvg import *
    from pytvg import _convert_values
except OSError as e:
    if "wrong ELF class" not in str(e):
        raise

    sys.stderr.write("Library has wrong ELF class, skipping tests\n")
    exit(0)

class CaptureStderr(object):
    def __init__(self):
        self.output = None

    def __enter__(self):
        sys.stderr.flush()
        self.orig_fd = os.dup(2)
        self.buffer = tempfile.TemporaryFile()
        os.dup2(self.buffer.fileno(), 2)
        return self

    def __exit__(self, *args):
        sys.stderr.flush()
        os.dup2(self.orig_fd, 2)
        os.close(self.orig_fd)

        # Load stderr messages from file
        self.buffer.seek(0)
        output = self.buffer.read()
        self.buffer.close()

        # Store messages, duplicate to stderr
        self.output = output.decode("utf-8")
        sys.stderr.buffer.write(output)
        return False

class VectorTests(unittest.TestCase):
    def test_add_entry(self):
        v = Vector()
        self.assertTrue(v.empty())
        self.assertTrue(v.empty(drop_cache=True))
        revisions = [v.revision]
        mem = v.memory_usage

        for i in range(10):
            v[i] = i * i

        self.assertFalse(v.empty())
        self.assertNotIn(v.revision, revisions)
        self.assertGreater(v.memory_usage, mem)
        revisions.append(v.revision)

        self.assertEqual(v.sum_weights(), 285.0)
        self.assertEqual(v.sum_weights(drop_cache=True), 285.0)
        self.assertEqual(v.norm(), math.sqrt(15333.0))
        self.assertEqual(v.norm(drop_cache=True), math.sqrt(15333.0))
        self.assertEqual(v.mul_vector(v), 15333.0)

        for i in range(10):
            self.assertTrue(v.has_entry(i))
            self.assertEqual(v[i], i * i)
            v.add_entry(i, 1.0)
            self.assertEqual(v[i], i * i + 1)
            v.sub_entry(i, 1.0)
            self.assertEqual(v[i], i * i)

        self.assertNotIn(v.revision, revisions)
        revisions.append(v.revision)

        v.mul_const(2.0)

        self.assertNotIn(v.revision, revisions)
        revisions.append(v.revision)

        for i in range(10):
            self.assertTrue(v.has_entry(i))
            self.assertEqual(v[i], 2.0 * i * i)
            del v[i]
            self.assertFalse(v.has_entry(i))
            self.assertEqual(v[i], 0.0)

        self.assertTrue(v.empty())
        self.assertNotIn(v.revision, revisions)
        del v

    def test_add_vector(self):
        v1 = Vector()
        v1[0] = 1.0
        v1[1] = 2.0
        v1[2] = 3.0

        v2 = Vector()
        v2[1] = 30.0
        v2[2] = 20.0
        v2[3] = 10.0

        v = v1 + v2
        self.assertEqual(v.as_dict(), {0: 1.0, 1: 32.0, 2: 23.0, 3: 10.0})
        del v

        v = v2 - v1
        self.assertEqual(v.as_dict(), {0: -1.0, 1: 28.0, 2: 17.0, 3: 10.0})
        del v

        del v1
        del v2

    def test_clear(self):
        v = Vector()
        self.assertTrue(v.empty())
        for i in range(10):
            v[i] = i * i
        self.assertFalse(v.empty())
        v.clear()
        self.assertTrue(v.empty())
        for i in range(10):
            v[i] = i * i
        self.assertFalse(v.empty())
        del v

    def test_batch(self):
        test_indices = np.array([0, 1, 2])
        test_weights = np.array([1.0, 2.0, 3.0])

        v = Vector()
        self.assertEqual(v.flags, 0)

        v.add_entries([], [])
        v.add_entries(test_indices, test_weights)
        indices, weights = v.entries()
        self.assertEqual(indices.tolist(), [0, 1, 2])
        self.assertEqual(weights.tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(v.num_entries, 3)
        self.assertEqual(len(v), 3)

        self.assertEqual(list(v.keys()), [0, 1, 2])
        self.assertEqual(list(v.values()), [1.0, 2.0, 3.0])
        self.assertEqual(list(v.items()), [(0, 1.0), (1, 2.0), (2, 3.0)])

        self.assertEqual(list(v), [0, 1, 2])
        for entry1, entry2 in zip(v, [0, 1, 2]):
            self.assertEqual(entry1, entry2)

        indices, _ = v.entries(ret_weights=False)
        self.assertEqual(indices.tolist(), [0, 1, 2])

        _, weights = v.entries(ret_indices=False)
        self.assertEqual(weights.tolist(), [1.0, 2.0, 3.0])

        v.add_entries([], [])
        v.add_entries(test_indices, test_weights)
        indices, weights = v.entries()
        self.assertEqual(indices.tolist(), [0, 1, 2])
        self.assertEqual(weights.tolist(), [2.0, 4.0, 6.0])

        v.add_entries([])
        v.add_entries(test_indices)
        indices, weights = v.entries()
        self.assertEqual(indices.tolist(), [0, 1, 2])
        self.assertEqual(weights.tolist(), [3.0, 5.0, 7.0])

        v.sub_entries([], [])
        v.sub_entries(test_indices, -test_weights)
        indices, weights = v.entries()
        self.assertEqual(indices.tolist(), [0, 1, 2])
        self.assertEqual(weights.tolist(), [4.0, 7.0, 10.0])

        v.sub_entries([])
        v.sub_entries(test_indices)
        indices, weights = v.entries()
        self.assertEqual(indices.tolist(), [0, 1, 2])
        self.assertEqual(weights.tolist(), [3.0, 6.0, 9.0])

        v.set_entries([], [])
        v.set_entries(test_indices, test_weights)
        indices, weights = v.entries()
        self.assertEqual(indices.tolist(), [0, 1, 2])
        self.assertEqual(weights.tolist(), [1.0, 2.0, 3.0])

        v.set_entries([])
        v.set_entries(test_indices)
        indices, weights = v.entries()
        self.assertEqual(indices.tolist(), [0, 1, 2])
        self.assertEqual(weights.tolist(), [1.0, 1.0, 1.0])

        v.del_entries([])
        v.del_entries(test_indices)
        self.assertEqual(v.entries()[0].tolist(), [])
        self.assertEqual(v.num_entries, 0)
        self.assertEqual(len(v), 0)

        v.set_entries(set())
        v.set_entries(set(test_indices.tolist()))
        indices, weights = v.entries()
        self.assertEqual(indices.tolist(), [0, 1, 2])
        self.assertEqual(weights.tolist(), [1.0, 1.0, 1.0])

        v.del_entries(set())
        v.del_entries(set(test_indices.tolist()))
        self.assertEqual(v.entries()[0].tolist(), [])
        self.assertEqual(v.num_entries, 0)
        self.assertEqual(len(v), 0)

        for i in range(1000):
            v.add_entry(i, 1.0)
        indices, _ = v.entries(ret_weights=False)
        indices = list(indices)
        for i in range(1000):
            self.assertIn(i, indices)
            indices.remove(i)
        self.assertEqual(len(indices), 0)
        self.assertEqual(v.num_entries, 1000)
        self.assertEqual(len(v), 1000)

        del v

    def test_flags(self):
        v = Vector()
        self.assertEqual(v.flags, 0)
        v[0] = 0.0
        self.assertTrue(v.has_entry(0))
        v.add_entry(0, 1.0)
        self.assertEqual(v[0], 1.0)
        v.add_entry(0, -1.0)
        self.assertTrue(v.has_entry(0))
        self.assertEqual(v[0], 0.0)
        v.sub_entry(0, 1.0)
        self.assertEqual(v[0], -1.0)
        v.sub_entry(0, -1.0)
        self.assertTrue(v.has_entry(0))
        self.assertEqual(v[0], 0.0)
        del v

        v = Vector()
        self.assertEqual(v.flags, 0)
        v[0] = 0.0
        v.del_small()
        self.assertFalse(v.has_entry(0))
        v.add_entry(0, 1.0)
        self.assertEqual(v[0], 1.0)
        v.add_entry(0, -0.75)
        self.assertEqual(v[0], 0.25)
        v.add_entry(0, -0.25)
        v.del_small()
        self.assertFalse(v.has_entry(0))
        v.sub_entry(0, 1.0)
        self.assertEqual(v[0], -1.0)
        v.sub_entry(0, -0.75)
        self.assertEqual(v[0], -0.25)
        v.sub_entry(0, -0.25)
        v.del_small()
        self.assertFalse(v.has_entry(0))
        del v

        v = Vector()
        self.assertEqual(v.flags, 0)
        v[0] = 0.0
        v.del_small(eps=0.5)
        self.assertFalse(v.has_entry(0))
        v.add_entry(0, 1.0)
        self.assertEqual(v[0], 1.0)
        v.add_entry(0, -0.25)
        self.assertEqual(v[0], 0.75)
        v.add_entry(0, -0.25)
        v.del_small(eps=0.5)
        self.assertFalse(v.has_entry(0))
        v.sub_entry(0, 1.0)
        self.assertEqual(v[0], -1.0)
        v.sub_entry(0, -0.25)
        self.assertEqual(v[0], -0.75)
        v.sub_entry(0, -0.25)
        v.del_small(eps=0.5)
        self.assertFalse(v.has_entry(0))
        del v

        v = Vector(positive=True)
        self.assertEqual(v.flags, TVG_FLAGS_POSITIVE)
        v[0] = 0.0
        v.del_small()
        self.assertFalse(v.has_entry(0))
        v.add_entry(0, 1.0)
        self.assertEqual(v[0], 1.0)
        v.add_entry(0, -0.75)
        self.assertEqual(v[0], 0.25)
        v.add_entry(0, -0.25)
        v.del_small()
        self.assertFalse(v.has_entry(0))
        v.sub_entry(0, 1.0)
        v.del_small()
        self.assertFalse(v.has_entry(0))
        v.sub_entry(0, -0.25)
        self.assertEqual(v[0], 0.25)
        del v

        v = Vector(positive=True)
        self.assertEqual(v.flags, TVG_FLAGS_POSITIVE)
        v[0] = 0.0
        v.del_small(eps=0.5)
        self.assertFalse(v.has_entry(0))
        v.add_entry(0, 1.0)
        self.assertEqual(v[0], 1.0)
        v.add_entry(0, -0.25)
        self.assertEqual(v[0], 0.75)
        v.add_entry(0, -0.25)
        v.del_small(eps=0.5)
        self.assertFalse(v.has_entry(0))
        v.sub_entry(0, 1.0)
        v.del_small(eps=0.5)
        self.assertFalse(v.has_entry(0))
        v.sub_entry(0, -0.25)
        v.del_small(eps=0.5)
        self.assertFalse(v.has_entry(0))
        v.sub_entry(0, -0.5)
        v.del_small(eps=0.5)
        self.assertFalse(v.has_entry(0))
        v.sub_entry(0, -0.75)
        self.assertEqual(v[0], 0.75)
        del v

    def test_mul_const(self):
        v = Vector()
        v[0] = 1.0
        v.mul_const(-1.0)
        self.assertEqual(v[0], -1.0)
        v.mul_const(0.0)
        self.assertTrue(v.has_entry(0))
        self.assertEqual(v[0], 0.0)
        del v

        v = Vector()
        v[0] = 1.0
        v.mul_const(-1.0)
        v.del_small()
        self.assertEqual(v[0], -1.0)
        v.mul_const(0.0)
        v.del_small()
        self.assertFalse(v.has_entry(0))
        del v

        v = Vector(positive=True)
        v[0] = 1.0
        v.mul_const(-1.0)
        v.del_small()
        self.assertFalse(v.has_entry(0))
        del v

        v = Vector(positive=True)
        v[0] = 1.0
        for i in range(200):
            v.mul_const(0.5)
            v.del_small()
            if not v.has_entry(0):
                break
        else:
            self.assertTrue(False)

    def test_sub_vector_norm(self):
        v = Vector()
        v[0] = 2.0

        w = Vector()
        w[0] = 5.0
        w[1] = 4.0

        self.assertEqual(v.sub_vector_norm(v), 0.0)
        self.assertEqual(w.sub_vector_norm(w), 0.0)
        self.assertEqual(v.sub_vector_norm(w), 5.0)
        self.assertEqual(w.sub_vector_norm(v), 5.0)

        del v
        del w

    def test_repr(self):
        v = Vector()
        self.assertEqual(repr(v), "Vector({})")
        for i in range(10):
            v[i] = 1.0
        expected = "Vector({0: X, 1: X, 2: X, 3: X, 4: X, 5: X, 6: X, 7: X, 8: X, 9: X})"
        self.assertEqual(repr(v).replace("1.000000", "X"), expected)
        v[10] = 2.0
        expected = "Vector({0: X, 1: X, 2: X, 3: X, 4: X, 5: X, 6: X, 7: X, 8: X, 9: X, ...})"
        self.assertEqual(repr(v).replace("1.000000", "X"), expected)
        del v

    def test_as_dict(self):
        v = Vector()
        for i in range(10):
            v[i] = i * i

        with self.assertRaises(ValueError):
            v.entries(ret_indices=False, as_dict=True)

        result = v.entries(ret_weights=False, as_dict=True)
        self.assertEqual(result, {0: None, 1: None, 2: None, 3: None, 4: None,
                                  5: None, 6: None, 7: None, 8: None, 9: None})

        result = v.as_dict()
        self.assertEqual(result, {0: 0.0, 1: 1.0, 2: 4.0, 3: 9.0, 4: 16.0,
                                  5: 25.0, 6: 36.0, 7: 49.0, 8: 64.0, 9: 81.0})

        del v
        v = Vector.from_dict(result)

        result = v.as_dict()
        self.assertEqual(result, {0: 0.0, 1: 1.0, 2: 4.0, 3: 9.0, 4: 16.0,
                                  5: 25.0, 6: 36.0, 7: 49.0, 8: 64.0, 9: 81.0})

        del v

    def test_duplicate(self):
        v = Vector()

        for i in range(10):
            v[i] = i * i

        revision = v.revision
        v2 = v.duplicate()

        for i in range(10):
            v[i] = 1.0

        self.assertNotEqual(v.revision, revision)
        self.assertEqual(v2.revision, revision)

        for i in range(10):
            self.assertEqual(v2[i], i * i)

        del v
        del v2

    def test_save_binary(self):
        v = Vector()
        for i in range(5):
            v[i] = i + 1.0

        with tempfile.NamedTemporaryFile() as temp:
            v.save_binary(temp.name)
            data = temp.read()
            with self.assertRaises(IOError):
                Graph.load_binary(temp.name)
            w = Vector.load_binary(temp.name)

        self.assertEqual(data.hex(), "54564756010000000000000000000000" +
                                     "0500000000000000" +
                                     "00000000000000000000803f00000000" +
                                     "01000000000000000000004000000000" +
                                     "02000000000000000000404000000000" +
                                     "03000000000000000000804000000000" +
                                     "04000000000000000000a04000000000")

        self.assertEqual(w.flags, 0)
        self.assertEqual(w.revision, 0)
        self.assertEqual(w.num_entries, 5)
        self.assertEqual(w.as_dict(), {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0})

        del v
        del w

        for length in range(len(data)):
            with tempfile.NamedTemporaryFile() as temp:
                temp.write(data[:length])
                temp.flush()
                with self.assertRaises(IOError):
                    Vector.load_binary(temp.name)

        v = Vector()
        for i in range(10000):
            v[i] = i

        with tempfile.NamedTemporaryFile() as temp:
            v.save_binary(temp.name)
            w = Vector.load_binary(temp.name)

        self.assertEqual(w.flags, 0)
        self.assertEqual(w.revision, 0)
        self.assertEqual(w.num_entries, 10000)

        for i in range(10000):
            self.assertTrue(w.has_entry(i))
            self.assertEqual(w[i], i)

        del v
        del w

class GraphTests(unittest.TestCase):
    def test_add_edge(self):
        g = Graph(directed=True)
        self.assertTrue(g.empty())
        self.assertTrue(g.empty(drop_cache=True))
        self.assertEqual(g.sum_weights(), 0.0)
        revisions = [g.revision]
        mem = g.memory_usage

        for i in range(100):
            s, t = i//10, i%10
            g[s, t] = i

        self.assertFalse(g.empty())
        self.assertNotIn(g.revision, revisions)
        self.assertGreater(g.memory_usage, mem)
        self.assertEqual(g.sum_weights(), 4950.0)
        revisions.append(g.revision)

        for i in range(100):
            s, t = i//10, i%10
            self.assertTrue(i == 0 or g.has_edge((s, t)))
            self.assertEqual(g[s, t], i)
            g.add_edge((s, t), 1.0)
            self.assertEqual(g[s, t], i + 1)
            g.sub_edge((s, t), 1.0)
            self.assertEqual(g[s, t], i)

        self.assertNotIn(g.revision, revisions)
        self.assertEqual(g.sum_weights(), 4950.0)
        revisions.append(g.revision)

        g.mul_const(2.0)

        self.assertNotIn(g.revision, revisions)
        self.assertEqual(g.sum_weights(), 9900.0)
        revisions.append(g.revision)

        for i in range(100):
            s, t = i//10, i%10
            self.assertTrue(i == 0 or g.has_edge((s, t)))
            self.assertEqual(g[s, t], 2.0 * i)
            del g[s, t]
            self.assertFalse(g.has_edge((s, t)))
            self.assertEqual(g[s, t], 0.0)

        self.assertTrue(g.empty())
        self.assertEqual(g.sum_weights(), 0.0)
        self.assertNotIn(g.revision, revisions)
        del g

    def test_add_graph(self):
        g1 = Graph()
        g1[0, 0] = 1.0
        g1[0, 1] = 2.0
        g1[0, 2] = 3.0

        g2 = Graph()
        g2[0, 1] = 30.0
        g2[0, 2] = 20.0
        g2[0, 3] = 10.0

        g = g1 + g2
        self.assertEqual(g.as_dict(), {(0, 0): 1.0, (0, 1): 32.0,
                                       (0, 2): 23.0, (0, 3): 10.0})
        del g

        g = g2 - g1
        self.assertEqual(g.as_dict(), {(0, 0): -1.0, (0, 1): 28.0,
                                       (0, 2): 17.0, (0, 3): 10.0})
        del g

        del g1
        del g2

    def test_clear(self):
        g = Graph(directed=True)
        self.assertTrue(g.empty())
        for i in range(100):
            s, t = i//10, i%10
            g[s, t] = i
        self.assertFalse(g.empty())
        g.clear()
        self.assertTrue(g.empty())
        for i in range(100):
            s, t = i//10, i%10
            g[s, t] = i
        self.assertFalse(g.empty())
        del g

    def test_power_iteration(self):
        g = Graph(directed=True)
        g[0, 0] = 0.5
        g[0, 1] = 0.5
        g[1, 0] = 0.2
        g[1, 1] = 0.8
        v, e = g.power_iteration()
        self.assertTrue(abs(e - 1.0) < 1e-7)
        self.assertTrue(abs(v[0] - 1.0 / math.sqrt(2)) < 1e-7)
        self.assertTrue(abs(v[1] - 1.0 / math.sqrt(2)) < 1e-7)
        del v
        v, _ = g.power_iteration(ret_eigenvalue=False)
        self.assertTrue(abs(v[0] - 1.0 / math.sqrt(2)) < 1e-7)
        self.assertTrue(abs(v[1] - 1.0 / math.sqrt(2)) < 1e-7)
        del v
        v, _ = g.power_iteration(tolerance=1e-3)
        self.assertTrue(abs(v[0] - 1.0 / math.sqrt(2)) < 1e-3)
        self.assertTrue(abs(v[1] - 1.0 / math.sqrt(2)) < 1e-3)
        del v
        w = Vector()
        w[0] = 1.0 / math.sqrt(2)
        w[1] = 1.0 / math.sqrt(2)
        v, _ = g.power_iteration(initial_guess=w, num_iterations=1)
        self.assertTrue(abs(v[0] - 1.0 / math.sqrt(2)) < 1e-7)
        self.assertTrue(abs(v[1] - 1.0 / math.sqrt(2)) < 1e-7)
        del v
        del w

        g.mul_const(-1)

        v, e = g.power_iteration()
        self.assertTrue(abs(e + 1.0) < 1e-7)
        self.assertTrue(abs(v[0] - 1.0 / math.sqrt(2)) < 1e-7)
        self.assertTrue(abs(v[1] - 1.0 / math.sqrt(2)) < 1e-7)
        del v
        v, _ = g.power_iteration(ret_eigenvalue=False)
        self.assertTrue(abs(v[0] - 1.0 / math.sqrt(2)) < 1e-7)
        self.assertTrue(abs(v[1] - 1.0 / math.sqrt(2)) < 1e-7)
        del v
        v, _ = g.power_iteration(tolerance=1e-3)
        self.assertTrue(abs(v[0] - 1.0 / math.sqrt(2)) < 1e-3)
        self.assertTrue(abs(v[1] - 1.0 / math.sqrt(2)) < 1e-3)
        del v

        del g

    def test_power_iteration_bug(self):
        g = Graph()
        g[0, 0] = 0.0
        g[1, 1] = 1.0

        v = g.power_iteration()[0].as_dict()
        self.assertEqual(v, {0: 0.0, 1: 1.0})

        del g

    def test_batch(self):
        test_indices = np.array([[0, 1], [1, 2], [2, 0]])
        test_weights = np.array([1.0, 2.0, 3.0])

        g = Graph(directed=True)
        self.assertEqual(g.flags, TVG_FLAGS_DIRECTED)
        self.assertTrue(g.directed)

        g.add_edges([], [])
        g.add_edges(test_indices, test_weights)
        indices, weights = g.edges()
        self.assertEqual(indices.tolist(), [[2, 0], [0, 1], [1, 2]])
        self.assertEqual(weights.tolist(), [3.0, 1.0, 2.0])
        self.assertEqual(g.num_edges, 3)
        self.assertEqual(len(g), 3)
        self.assertEqual(g.nodes().tolist(), [0, 1, 2])
        self.assertEqual(g.nodes().as_dict(), {0: 2.0, 1: 2.0, 2: 2.0})
        self.assertEqual(g.num_nodes, 3)

        self.assertEqual(list(g.keys()), [(2, 0), (0, 1), (1, 2)])
        self.assertEqual(list(g.values()), [3.0, 1.0, 2.0])
        self.assertEqual(list(g.items()), [((2, 0), 3.0), ((0, 1), 1.0), ((1, 2), 2.0)])

        self.assertEqual(list(g), [(2, 0), (0, 1), (1, 2)])
        for edge1, edge2 in zip(g, [(2, 0), (0, 1), (1, 2)]):
            self.assertEqual(edge1, edge2)

        indices, _ = g.edges(ret_weights=False)
        self.assertEqual(indices.tolist(), [[2, 0], [0, 1], [1, 2]])

        _, weights = g.edges(ret_indices=False)
        self.assertEqual(weights.tolist(), [3.0, 1.0, 2.0])

        for i in range(3):
            indices, weights = g.adjacent_edges(i)
            self.assertEqual(indices.tolist(), [(i + 1) % 3])
            self.assertEqual(weights.tolist(), [i + 1.0])

            indices, _ = g.adjacent_edges(i, ret_weights=False)
            self.assertEqual(indices.tolist(), [(i + 1) % 3])

            _, weights = g.adjacent_edges(i, ret_indices=False)
            self.assertEqual(weights.tolist(), [i + 1.0])

            with self.assertRaises(ValueError):
                g.adjacent_edges(i, ret_indices=False, as_dict=True)

            result = g.adjacent_edges(i, ret_weights=False, as_dict=True)
            self.assertEqual(result, {(i + 1) % 3: None})

            result = g.adjacent_edges(i, as_dict=True)
            self.assertEqual(result, {(i + 1) % 3: i + 1})

        g.add_edges([], [])
        g.add_edges(test_indices, test_weights)
        indices, weights = g.edges()
        self.assertEqual(indices.tolist(), [[2, 0], [0, 1], [1, 2]])
        self.assertEqual(weights.tolist(), [6.0, 2.0, 4.0])

        g.add_edges([])
        g.add_edges(test_indices)
        indices, weights = g.edges()
        self.assertEqual(indices.tolist(), [[2, 0], [0, 1], [1, 2]])
        self.assertEqual(weights.tolist(), [7.0, 3.0, 5.0])

        g.sub_edges([], [])
        g.sub_edges(test_indices, -test_weights)
        indices, weights = g.edges()
        self.assertEqual(indices.tolist(), [[2, 0], [0, 1], [1, 2]])
        self.assertEqual(weights.tolist(), [10.0, 4.0, 7.0])

        g.sub_edges([])
        g.sub_edges(test_indices)
        indices, weights = g.edges()
        self.assertEqual(indices.tolist(), [[2, 0], [0, 1], [1, 2]])
        self.assertEqual(weights.tolist(), [9.0, 3.0, 6.0])

        g.set_edges([], [])
        g.set_edges(test_indices, test_weights)
        indices, weights = g.edges()
        self.assertEqual(indices.tolist(), [[2, 0], [0, 1], [1, 2]])
        self.assertEqual(weights.tolist(), [3.0, 1.0, 2.0])

        g.set_edges([])
        g.set_edges(test_indices)
        indices, weights = g.edges()
        self.assertEqual(indices.tolist(), [[2, 0], [0, 1], [1, 2]])
        self.assertEqual(weights.tolist(), [1.0, 1.0, 1.0])

        g.del_edges([])
        g.del_edges(test_indices)
        self.assertEqual(g.edges()[0].tolist(), [])
        self.assertEqual(g.num_edges, 0)
        self.assertEqual(len(g), 0)
        self.assertEqual(g.nodes().tolist(), [])
        self.assertEqual(g.nodes().as_dict(), {})
        self.assertEqual(g.num_nodes, 0)

        g.set_edges(set())
        g.set_edges(set([tuple(i) for i in test_indices]))
        indices, weights = g.edges()
        self.assertEqual(indices.tolist(), [[2, 0], [0, 1], [1, 2]])
        self.assertEqual(weights.tolist(), [1.0, 1.0, 1.0])

        g.del_edges(set())
        g.del_edges(set([tuple(i) for i in test_indices]))
        self.assertEqual(g.edges()[0].tolist(), [])
        self.assertEqual(g.num_edges, 0)
        self.assertEqual(len(g), 0)
        self.assertEqual(g.nodes().tolist(), [])
        self.assertEqual(g.nodes().as_dict(), {})
        self.assertEqual(g.num_nodes, 0)

        for i in range(1000):
            g.add_edge((i, i + 1), 1.0)
        indices, _ = g.edges(ret_weights=False)
        indices = [tuple(x) for x in indices]
        for i in range(1000):
            self.assertIn((i, i + 1), indices)
            indices.remove((i, i + 1))
        self.assertEqual(len(indices), 0)
        self.assertEqual(g.num_edges, 1000)
        self.assertEqual(len(g), 1000)

        del g

    def test_directed(self):
        g = Graph(directed=True)
        g[1, 1] = 2.0
        g[1, 2] = 1.0
        self.assertEqual(g[1, 1], 2.0)
        self.assertEqual(g[1, 2], 1.0)
        self.assertEqual(g[2, 1], 0.0)
        indices, weights = g.edges()
        self.assertEqual(indices.tolist(), [[1, 1], [1, 2]])
        self.assertEqual(weights.tolist(), [2.0, 1.0])
        self.assertEqual(g.num_edges, 2)
        self.assertEqual(g.nodes().tolist(), [1, 2])
        self.assertEqual(g.nodes().as_dict(), {1: 3.0, 2: 1.0})
        self.assertEqual(g.num_nodes, 2)
        del g

    def test_undirected(self):
        g = Graph(directed=False)
        g[1, 1] = 2.0
        g[1, 2] = 1.0
        self.assertEqual(g[1, 1], 2.0)
        self.assertEqual(g[1, 2], 1.0)
        self.assertEqual(g[2, 1], 1.0)
        indices, weights = g.edges()
        self.assertEqual(indices.tolist(), [[1, 1], [1, 2]])
        self.assertEqual(weights.tolist(), [2.0, 1.0])
        self.assertEqual(g.num_edges, 2)
        self.assertEqual(g.nodes().tolist(), [1, 2])
        self.assertEqual(g.nodes().as_dict(), {1: 3.0, 2: 1.0})
        self.assertEqual(g.num_nodes, 2)
        del g

    def test_flags(self):
        g = Graph()
        self.assertEqual(g.flags, 0)
        self.assertFalse(g.directed)
        g[0, 0] = 0.0
        self.assertTrue(g.has_edge((0, 0)))
        g.add_edge((0, 0), 1.0)
        self.assertEqual(g[0, 0], 1.0)
        g.add_edge((0, 0), -1.0)
        self.assertTrue(g.has_edge((0, 0)))
        self.assertEqual(g[0, 0], 0.0)
        g.sub_edge((0, 0), 1.0)
        self.assertEqual(g[0, 0], -1.0)
        g.sub_edge((0, 0), -1.0)
        self.assertTrue(g.has_edge((0, 0)))
        self.assertEqual(g[0, 0], 0.0)
        del g

        g = Graph()
        self.assertEqual(g.flags, 0)
        self.assertFalse(g.directed)
        g[0, 0] = 0.0
        g.del_small()
        self.assertFalse(g.has_edge((0, 0)))
        g.add_edge((0, 0), 1.0)
        self.assertEqual(g[0, 0], 1.0)
        g.add_edge((0, 0), -0.75)
        self.assertEqual(g[0, 0], 0.25)
        g.add_edge((0, 0), -0.25)
        g.del_small()
        self.assertFalse(g.has_edge((0, 0)))
        g.sub_edge((0, 0), 1.0)
        self.assertEqual(g[0, 0], -1.0)
        g.sub_edge((0, 0), -0.75)
        self.assertEqual(g[0, 0], -0.25)
        g.sub_edge((0, 0), -0.25)
        g.del_small()
        self.assertFalse(g.has_edge((0, 0)))
        del g

        g = Graph()
        self.assertEqual(g.flags, 0)
        self.assertFalse(g.directed)
        g[0, 0] = 0.0
        g.del_small(eps=0.5)
        self.assertFalse(g.has_edge((0, 0)))
        g.add_edge((0, 0), 1.0)
        self.assertEqual(g[0, 0], 1.0)
        g.add_edge((0, 0), -0.25)
        self.assertEqual(g[0, 0], 0.75)
        g.add_edge((0, 0), -0.25)
        g.del_small(eps=0.5)
        self.assertFalse(g.has_edge((0, 0)))
        g.sub_edge((0, 0), 1.0)
        self.assertEqual(g[0, 0], -1.0)
        g.sub_edge((0, 0), -0.25)
        self.assertEqual(g[0, 0], -0.75)
        g.sub_edge((0, 0), -0.25)
        g.del_small(eps=0.5)
        self.assertFalse(g.has_edge((0, 0)))
        del g

        g = Graph(positive=True)
        self.assertEqual(g.flags, TVG_FLAGS_POSITIVE)
        self.assertFalse(g.directed)
        g[0, 0] = 0.0
        g.del_small()
        self.assertFalse(g.has_edge((0, 0)))
        g.add_edge((0, 0), 1.0)
        self.assertEqual(g[0, 0], 1.0)
        g.add_edge((0, 0), -0.75)
        self.assertEqual(g[0, 0], 0.25)
        g.add_edge((0, 0), -0.25)
        g.del_small()
        self.assertFalse(g.has_edge((0, 0)))
        g.sub_edge((0, 0), 1.0)
        g.del_small()
        self.assertFalse(g.has_edge((0, 0)))
        g.sub_edge((0, 0), -0.25)
        self.assertEqual(g[0, 0], 0.25)
        del g

        g = Graph(positive=True)
        self.assertEqual(g.flags, TVG_FLAGS_POSITIVE)
        self.assertFalse(g.directed)
        g[0, 0] = 0.0
        g.del_small(eps=0.5)
        self.assertFalse(g.has_edge((0, 0)))
        g.add_edge((0, 0), 1.0)
        self.assertEqual(g[0, 0], 1.0)
        g.add_edge((0, 0), -0.25)
        self.assertEqual(g[0, 0], 0.75)
        g.add_edge((0, 0), -0.25)
        g.del_small(eps=0.5)
        self.assertFalse(g.has_edge((0, 0)))
        g.sub_edge((0, 0), 1.0)
        g.del_small(eps=0.5)
        self.assertFalse(g.has_edge((0, 0)))
        g.sub_edge((0, 0), -0.25)
        g.del_small(eps=0.5)
        self.assertFalse(g.has_edge((0, 0)))
        g.sub_edge((0, 0), -0.5)
        g.del_small(eps=0.5)
        self.assertFalse(g.has_edge((0, 0)))
        g.sub_edge((0, 0), -0.75)
        self.assertEqual(g[0, 0], 0.75)
        del g

    def test_bfs(self):
        g = Graph(directed=True)
        g[0, 1] = 1.0
        g[1, 2] = 1.0
        g[2, 3] = 1.0
        g[3, 4] = 1.5
        g[2, 4] = 1.5

        indices, weights = g.edges()
        self.assertEqual(indices.tolist(), [[0, 1], [1, 2], [2, 3], [2, 4], [3, 4]])
        self.assertEqual(weights.tolist(), [1.0, 1.0, 1.0, 1.5, 1.5])

        value = g.distance_count(100, 0)
        self.assertEqual(value, np.inf)
        value = g.distance_weight(100, 0)
        self.assertEqual(value, np.inf)

        value = g.distance_count(0, 0)
        self.assertEqual(value, 0)
        value = g.distance_weight(0, 0)
        self.assertEqual(value, 0.0)

        value = g.distance_count(0, 4)
        self.assertEqual(value, 3)
        value = g.distance_weight(0, 4)
        self.assertEqual(value, 3.5)

        counts = g.all_distances_count(0)
        self.assertEqual(counts.as_dict(), {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0, 4: 3.0})
        counts = g.all_distances_count(0, max_count=2)
        self.assertEqual(counts.as_dict(), {0: 0.0, 1: 1.0, 2: 2.0})

        counts = g.all_distances_count(100)
        self.assertEqual(counts.as_dict(), {100: 0.0})

        weights = g.all_distances_weight(0)
        self.assertEqual(weights.as_dict(), {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0, 4: 3.5})
        weights = g.all_distances_weight(0, max_weight=2.0)
        self.assertEqual(weights.as_dict(), {0: 0.0, 1: 1.0, 2: 2.0})

        weights = g.all_distances_weight(100)
        self.assertEqual(weights.as_dict(), {100: 0.0})

        results = g.bfs_count(0, max_count=2)
        self.assertEqual(results, [(0.0, 0, None, 0), (1.0, 1, 0, 1), (2.0, 2, 1, 2)])
        results = g.bfs_count(0)
        self.assertEqual(results, [(0.0, 0, None, 0), (1.0, 1, 0, 1), (2.0, 2, 1, 2), (3.0, 3, 2, 3), (3.5, 3, 2, 4)])

        results = g.bfs_weight(0, max_weight=2.0)
        self.assertEqual(results, [(0.0, 0, None, 0), (1.0, 1, 0, 1), (2.0, 2, 1, 2)])
        results = g.bfs_weight(0)
        self.assertEqual(results, [(0.0, 0, None, 0), (1.0, 1, 0, 1), (2.0, 2, 1, 2), (3.0, 3, 2, 3), (3.5, 3, 2, 4)])

        distances = g.all_distances_graph(use_weights=False)
        self.assertEqual(distances.as_dict(), {(0, 1): 1.0, (0, 2): 2.0, (0, 3): 3.0, (0, 4): 3.0,
                                               (1, 2): 1.0, (1, 3): 2.0, (1, 4): 2.0, (2, 3): 1.0,
                                               (2, 4): 1.0, (3, 4): 1.0})
        del distances

        distances = g.all_distances_graph(use_weights=True)
        self.assertEqual(distances.as_dict(), {(0, 1): 1.0, (0, 2): 2.0, (0, 3): 3.0, (0, 4): 3.5,
                                               (1, 2): 1.0, (1, 3): 2.0, (1, 4): 2.5, (2, 3): 1.0,
                                               (2, 4): 1.5, (3, 4): 1.5})
        del distances

        del g

    def test_connected_components(self):
        g = Graph(directed=False)
        g[1, 0] = 1.0
        g[1, 2] = 2.0
        g[4, 3] = 3.0
        g[4, 5] = 4.0

        components = g.connected_components()
        self.assertEqual(components.as_dict(), {0: 0.0, 1: 0.0, 2: 0.0, 3: 1.0, 4: 1.0, 5: 1.0})
        del components

        del g

    def test_mul_const(self):
        g = Graph()
        g[0, 0] = 1.0
        g.mul_const(-1.0)
        self.assertEqual(g[0, 0], -1.0)
        g.mul_const(0.0)
        self.assertTrue(g.has_edge((0, 0)))
        self.assertEqual(g[0, 0], 0.0)
        del g

        g = Graph()
        g[0, 0] = 1.0
        g.mul_const(-1.0)
        g.del_small()
        self.assertEqual(g[0, 0], -1.0)
        g.mul_const(0.0)
        g.del_small()
        self.assertFalse(g.has_edge((0, 0)))
        del g

        g = Graph(positive=True)
        g[0, 0] = 1.0
        g.mul_const(-1.0)
        g.del_small()
        self.assertFalse(g.has_edge((0, 0)))
        del g

        g = Graph(positive=True)
        g[0, 0] = 1.0
        for i in range(200):
            g.mul_const(0.5)
            g.del_small()
            if not g.has_edge((0, 0)):
                break
        else:
            self.assertTrue(False)

    def test_mul_vector(self):
        g = Graph(directed=True)
        for i in range(9):
            s, t = i//3, i%3
            g[s, t] = i + 1

        v = Vector()
        for i in range(3):
            v[i] = i + 1

        g2 = g.mul_vector(v)
        self.assertEqual(g2.as_dict(), {0: 14.0, 1: 32.0, 2: 50.0})

        del g2
        del g
        del v

    def test_weights(self):
        g = Graph(directed=True)
        g[0, 0] = 1.0
        g[0, 1] = 2.0
        g[1, 0] = 3.0
        g[1, 2] = 0.0
        g[2, 3] = 0.0

        d = g.in_degrees()
        indices, weights = d.entries()
        self.assertEqual(indices.tolist(), [0, 1, 2, 3])
        self.assertEqual(weights.tolist(), [2.0, 1.0, 1.0, 1.0])
        d = g.in_weights()
        indices, weights = d.entries()
        self.assertEqual(indices.tolist(), [0, 1, 2, 3])
        self.assertEqual(weights.tolist(), [4.0, 2.0, 0.0, 0.0])

        d = g.out_degrees()
        indices, weights = d.entries()
        self.assertEqual(indices.tolist(), [0, 1, 2])
        self.assertEqual(weights.tolist(), [2.0, 2.0, 1.0])
        d = g.out_weights()
        indices, weights = d.entries()
        self.assertEqual(indices.tolist(), [0, 1, 2])
        self.assertEqual(weights.tolist(), [3.0, 3.0, 0.0])
        del g

    def test_anomalies(self):
        g = Graph(directed=False)

        a = g.degree_anomalies()
        indices, weights = a.entries()
        self.assertEqual(indices.tolist(), [])
        self.assertEqual(weights.tolist(), [])
        a = g.weight_anomalies()
        indices, weights = a.entries()
        self.assertEqual(indices.tolist(), [])
        self.assertEqual(weights.tolist(), [])

        c = 0
        for i in range(5):
            for j in range(i + 1):
                if c % 3 != 0: g[i, j] = c
                c += 1

        a = g.degree_anomalies()
        indices, weights = a.entries()
        self.assertEqual(indices.tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(weights.tolist(), [-2.5, 1.5999999046325684, -0.6666667461395264, -1.0, 0.5])
        a = g.weight_anomalies()
        indices, weights = a.entries()
        self.assertEqual(indices.tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(weights.tolist(), [-34.90909194946289,-9.119998931884766, -7.0588226318359375,
                                            -5.392856597900391, 18.39583396911621])
        del g

    def test_repr(self):
        g = Graph()
        self.assertEqual(repr(g), "Graph({})")
        for i in range(10):
            g[1, i] = 1.0
        expected = "Graph({(0, 1): X, (1, 1): X, (1, 2): X, (1, 3): X, (1, 4): X, (1, 5): X, (1, 6): X, (1, 7): X, (1, 8): X, (1, 9): X})"
        self.assertEqual(repr(g).replace("1.000000", "X"), expected)
        g[1, 10] = 2.0
        expected = "Graph({(0, 1): X, (1, 1): X, (1, 2): X, (1, 3): X, (1, 4): X, (1, 5): X, (1, 6): X, (1, 7): X, (1, 8): X, (1, 9): X, ...})"
        self.assertEqual(repr(g).replace("1.000000", "X"), expected)
        del g

        l = Node()
        self.assertEqual(repr(l), "Node({})")
        l["attr1"] = "sample attr1"
        l["attr2"] = "sample attr2"
        l["attr3"] = "sample attr3"
        self.assertEqual(repr(l), "Node({'attr1': 'sample attr1', 'attr2': 'sample attr2', 'attr3': 'sample attr3'})")
        del l

    def test_filter_nodes(self):
        g = Graph(directed=True)
        g[0, 1] = 1.0
        g[1, 2] = 2.0
        g[2, 3] = 3.0
        g[3, 4] = 4.0
        g[2, 4] = 5.0

        h = g.filter_nodes([1, 2, 3])
        indices, weights = h.edges()
        self.assertEqual(indices.tolist(), [[1, 2], [2, 3]])
        self.assertEqual(weights.tolist(), [2.0, 3.0])

        del g
        del h

    def test_normalize(self):
        g = Graph(directed=True)
        g[0, 1] = 1.0
        g[0, 2] = 0.5
        g[0, 3] = 0.5
        g2 = g.normalize()
        self.assertEqual(g2.as_dict(), {(0, 1): 0.5, (0, 2): 0.5, (0, 3): 0.5})
        del g2

        g[2, 1] = 3.0
        g2 = g.normalize()
        self.assertEqual(g2.as_dict(), {(0, 1): 0.125, (0, 2): 0.5, (0, 3): 0.5, (2, 1): 0.25})
        del g2
        del g

        g = Graph(directed=False)
        g[0, 1] = 1.0
        g[0, 2] = 0.5
        g[0, 3] = 0.5
        g2 = g.normalize()
        self.assertEqual(g2.as_dict(), {(0, 1): 0.5, (0, 2): 0.5, (0, 3): 0.5})
        del g2

        g[2, 1] = 3.0
        g2 = g.normalize()
        self.assertEqual(g2.num_edges, 4)
        self.assertEqual(g2[0, 1], 0.125)
        self.assertTrue(abs(g2[0, 2] - 0.07142858) < 1e-7)
        self.assertEqual(g2[0, 3], 0.5)
        self.assertTrue(abs(g2[1, 2] - 0.21428572) < 1e-7)
        del g2
        del g

    def test_save_binary(self):
        g = Graph(directed=True)
        g[0, 1] = 1.0
        g[1, 2] = 2.0
        g[2, 3] = 3.0
        g[3, 4] = 4.0
        g[2, 4] = 5.0

        with tempfile.NamedTemporaryFile() as temp:
            g.save_binary(temp.name)
            data = temp.read()
            with self.assertRaises(IOError):
                Vector.load_binary(temp.name)
            h = Graph.load_binary(temp.name)

        self.assertEqual(data.hex(), "5456474701000000040000000000000000000000" +
                                     "0500000000000000" +
                                     "000000000000000001000000000000000000803f00000000" +
                                     "010000000000000002000000000000000000004000000000" +
                                     "020000000000000003000000000000000000404000000000" +
                                     "020000000000000004000000000000000000a04000000000" +
                                     "030000000000000004000000000000000000804000000000")

        self.assertEqual(h.flags, TVG_FLAGS_DIRECTED)
        self.assertEqual(h.directed, True)
        self.assertEqual(h.revision, 0)
        self.assertEqual(h.num_edges, 5)
        self.assertEqual(h.as_dict(), {(0, 1): 1.0, (1, 2): 2.0, (2, 3): 3.0, (2, 4): 5.0, (3, 4): 4.0})

        del g
        del h

        for length in range(len(data)):
            with tempfile.NamedTemporaryFile() as temp:
                temp.write(data[:length])
                temp.flush()
                with self.assertRaises(IOError):
                    Graph.load_binary(temp.name)

        g = Graph(directed=True)
        for i in range(10000):
            s, t = i//100, i%100
            g[s, t] = i

        with tempfile.NamedTemporaryFile() as temp:
            g.save_binary(temp.name)
            h = Graph.load_binary(temp.name)

        self.assertEqual(h.flags, TVG_FLAGS_DIRECTED)
        self.assertEqual(h.directed, True)
        self.assertEqual(h.revision, 0)
        self.assertEqual(h.num_edges, 10000)

        for i in range(10000):
            s, t = i//100, i%100
            self.assertTrue(h.has_edge((s, t)))
            self.assertEqual(h[s, t], i)

        del g
        del h

    def test_as_dict(self):
        g = Graph(directed=True)
        for i in range(100):
            s, t = i//10, i%10
            g[s, t] = i

        with self.assertRaises(ValueError):
            g.edges(ret_indices=False, as_dict=True)

        result = g.edges(ret_weights=False, as_dict=True)
        self.assertEqual(len(result), 100)
        for i in range(100):
            s, t = i//10, i%10
            self.assertEqual(result[s, t], None)

        result = g.as_dict()
        self.assertEqual(len(result), 100)
        for i in range(100):
            s, t = i//10, i%10
            self.assertEqual(result[s, t], i)

        del g
        g = Graph.from_dict(result, directed=True)

        result = g.as_dict()
        self.assertEqual(len(result), 100)
        for i in range(100):
            s, t = i//10, i%10
            self.assertEqual(result[s, t], i)

        del g

    def test_top_edges(self):
        g = Graph(directed=True)

        for i in range(100):
            s, t = i//10, i%10
            g[s, t] = (i * 13) % 100

        indices, weights = g.top_edges(5, truncate=True)
        self.assertEqual(indices.tolist(), [[2, 3], [4, 6], [6, 9], [9, 2], [1, 5]])
        self.assertEqual(weights.tolist(), [99.0, 98.0, 97.0, 96.0, 95.0])

        indices, _ = g.top_edges(5, ret_weights=False, truncate=True)
        self.assertEqual(indices.tolist(), [[2, 3], [4, 6], [6, 9], [9, 2], [1, 5]])

        _, weights = g.top_edges(5, ret_indices=False, truncate=True)
        self.assertEqual(weights.tolist(), [99.0, 98.0, 97.0, 96.0, 95.0])

        with self.assertRaises(ValueError):
            g.top_edges(5, ret_indices=False, as_dict=True, truncate=True)

        result = g.top_edges(5, ret_weights=False, as_dict=True, truncate=True)
        self.assertEqual(result, {(2, 3): None, (4, 6): None, (6, 9): None, (9, 2): None, (1, 5): None})

        result = g.top_edges(5, as_dict=True, truncate=True)
        self.assertEqual(result, {(2, 3): 99.0, (4, 6): 98.0, (6, 9): 97.0, (9, 2): 96.0, (1, 5): 95.0})

        del g
        g = Graph(directed=True)

        for i in range(100):
            s, t = i//10, i%10
            g[s, t] = (s + t) % 10

        result = g.top_edges(0, as_dict=True, truncate=True)
        self.assertEqual(result, {})

        result = g.top_edges(0, as_dict=True)
        self.assertEqual(result, {})

        result = g.top_edges(1, as_dict=True, truncate=True)
        self.assertEqual(result, {(9, 0): 9.0})

        result = g.top_edges(1, as_dict=True)
        self.assertEqual(result, {(9, 0): 9.0, (8, 1): 9.0, (6, 3): 9.0, (3, 6): 9.0, (2, 7): 9.0,
                                  (5, 4): 9.0, (1, 8): 9.0, (0, 9): 9.0, (7, 2): 9.0, (4, 5): 9.0})

        result = g.top_edges(1, as_dict=True, ret_weights=False)
        self.assertEqual(result, {(9, 0): None, (8, 1): None, (6, 3): None, (3, 6): None, (2, 7): None,
                                  (5, 4): None, (1, 8): None, (0, 9): None, (7, 2): None, (4, 5): None})

        result = g.top_edges(5, as_dict=True, truncate=True)
        self.assertEqual(result, {(9, 0): 9.0, (8, 1): 9.0, (6, 3): 9.0, (3, 6): 9.0, (2, 7): 9.0})

        result = g.top_edges(5, as_dict=True)
        self.assertEqual(result, {(9, 0): 9.0, (8, 1): 9.0, (6, 3): 9.0, (3, 6): 9.0, (2, 7): 9.0,
                                  (5, 4): 9.0, (1, 8): 9.0, (0, 9): 9.0, (7, 2): 9.0, (4, 5): 9.0})

        result = g.top_edges(11, as_dict=True, truncate=True)
        self.assertEqual(result, {(9, 0): 9.0, (8, 1): 9.0, (6, 3): 9.0, (3, 6): 9.0, (2, 7): 9.0,
                                  (5, 4): 9.0, (1, 8): 9.0, (0, 9): 9.0, (7, 2): 9.0, (4, 5): 9.0,
                                  (8, 0): 8.0})

        result = g.top_edges(11, as_dict=True)
        self.assertEqual(result, {(9, 0): 9.0, (8, 1): 9.0, (6, 3): 9.0, (3, 6): 9.0, (2, 7): 9.0,
                                  (5, 4): 9.0, (1, 8): 9.0, (0, 9): 9.0, (7, 2): 9.0, (4, 5): 9.0,
                                  (8, 0): 8.0, (7, 1): 8.0, (1, 7): 8.0, (5, 3): 8.0, (0, 8): 8.0,
                                  (4, 4): 8.0, (9, 9): 8.0, (3, 5): 8.0, (6, 2): 8.0, (2, 6): 8.0})

        del g

    def test_duplicate(self):
        g = Graph(directed=True)

        for i in range(100):
            s, t = i//10, i%10
            g[s, t] = i

        revision = g.revision
        g2 = g.duplicate()

        for i in range(100):
            s, t = i//10, i%10
            g[s, t] = 1.0

        self.assertNotEqual(g.revision, revision)
        self.assertEqual(g2.revision, revision)

        for i in range(100):
            s, t = i//10, i%10
            self.assertEqual(g2[s, t], i)

        del g
        del g2

class TVGTests(unittest.TestCase):
    def test_lookup(self):
        tvg = TVG(positive=True)
        self.assertEqual(tvg.flags, TVG_FLAGS_POSITIVE)
        mem = tvg.memory_usage

        g1 = Graph(positive=True)
        tvg.link_graph(g1, 100)
        self.assertEqual(g1.flags, TVG_FLAGS_POSITIVE | TVG_FLAGS_READONLY)
        self.assertEqual(g1.ts, 100)
        self.assertEqual(g1.id, None)

        g2 = Graph(positive=True)
        tvg.link_graph(g2, 200)
        self.assertEqual(g2.flags, TVG_FLAGS_POSITIVE | TVG_FLAGS_READONLY)
        self.assertEqual(g2.ts, 200)
        self.assertEqual(g2.id, None)

        g3 = Graph(positive=True)
        tvg.link_graph(g3, 300)
        self.assertEqual(g3.flags, TVG_FLAGS_POSITIVE | TVG_FLAGS_READONLY)
        self.assertEqual(g3.ts, 300)
        self.assertEqual(g3.id, None)
        self.assertGreater(tvg.memory_usage, mem)

        g = tvg.lookup_le(50)
        self.assertEqual(g, None)
        g = tvg.lookup_ge(50)
        self.assertEqual(g, g1)

        g = tvg.lookup_le(150)
        self.assertEqual(g, g1)
        g = tvg.lookup_ge(150)
        self.assertEqual(g, g2)

        g = tvg.lookup_le(250)
        self.assertEqual(g, g2)
        g = tvg.lookup_ge(250)
        self.assertEqual(g, g3)

        g = tvg.lookup_le(350)
        self.assertEqual(g, g3)
        g = tvg.lookup_ge(350)
        self.assertEqual(g, None)

        g = tvg.lookup_near(149)
        self.assertEqual(g, g1)
        g = tvg.lookup_near(151)
        self.assertEqual(g, g2)

        graphs = list(tvg.documents())
        self.assertEqual(graphs, [g1, g2, g3])
        graphs = list(tvg.documents(limit=2))
        self.assertEqual(graphs, [g1, g2])
        graphs = list(tvg.documents(limit=1))
        self.assertEqual(graphs, [g1])

        graphs = list(tvg.documents(50, 150))
        self.assertEqual(graphs, [g1])
        graphs = list(tvg.documents(50, 250))
        self.assertEqual(graphs, [g1, g2])
        graphs = list(tvg.documents(50, 350))
        self.assertEqual(graphs, [g1, g2, g3])
        graphs = list(tvg.documents(150, 350))
        self.assertEqual(graphs, [g2, g3])
        graphs = list(tvg.documents(250, 350))
        self.assertEqual(graphs, [g3])

        graphs = list(tvg.documents(0, 0xffffffffffffffff))
        self.assertEqual(graphs, [g1, g2, g3])
        graphs = list(tvg.documents(-100, 0xffffffffffffffff + 100))
        self.assertEqual(graphs, [g1, g2, g3])
        graphs = list(tvg.documents(0x8000000000000000, 0xffffffffffffffff))
        self.assertEqual(graphs, [])

        # For backwards compatibility, we still allow passing float values.

        g = tvg.lookup_ge(100.0)
        self.assertEqual(g, g1)
        g = tvg.lookup_ge(100.01)
        self.assertEqual(g, g2)

        g = tvg.lookup_le(200.0)
        self.assertEqual(g, g2)
        g = tvg.lookup_le(199.99)
        self.assertEqual(g, g1)

        g = tvg.lookup_near(149.49)
        self.assertEqual(g, g1)
        g = tvg.lookup_near(150.51)
        self.assertEqual(g, g2)

        del tvg

    def test_link(self):
        tvg = TVG()
        g1 = Graph()
        g2 = Graph(directed=True)
        g3 = Graph()

        tvg.link_graph(g1, 10)
        with self.assertRaises(RuntimeError):
            tvg.link_graph(g1, 20)
        with self.assertRaises(RuntimeError):
            tvg.link_graph(g2, 20)
        with self.assertRaises(RuntimeError):
            tvg.link_graph(g3, 0x8000000000000000)
        with self.assertRaises(RuntimeError):
            tvg.link_graph(g3, 0xffffffffffffffff)

        g = tvg.lookup_near(10)
        self.assertEqual(g.ts, 10)
        self.assertEqual(addressof(g._obj.contents), addressof(g1._obj.contents))
        del tvg

    def test_compress(self):
        source = np.random.rand(100)

        tvg = TVG(positive=True)

        for t, s in enumerate(source):
            g = Graph()
            g[0, 0] = s
            tvg.link_graph(g, t)

        tvg.compress(step=5, offset=100)

        t = 0
        for g in tvg:
            self.assertEqual(g.ts, t)
            self.assertTrue(abs(g[0, 0] - np.sum(source[t:t+5])) < 1e-6)
            t += 5

        del tvg

        # Repeat the test with the samples parameter instead of step and offset.
        tvg = TVG(positive=True)

        for t, s in enumerate(source):
            g = Graph()
            g[0, 0] = s
            tvg.link_graph(g, t)

        tvg.compress(samples=UniformSamples(step=5, offset=100))

        t = 0
        for g in tvg:
            self.assertEqual(g.ts, t)
            self.assertTrue(abs(g[0, 0] - np.sum(source[t:t+5])) < 1e-6)
            t += 5

        del tvg

    def test_load(self):
        filename_graphs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../datasets/example/example-tvg.graph")
        filename_nodes = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../datasets/example/example-tvg.nodes")
        tvg = TVG.load(filename_graphs, nodes=filename_nodes)
        tvg.set_primary_key(["a", "b", "c"])
        tvg.set_primary_key(["text"])

        l = tvg.node_by_index(1)
        self.assertEqual(l.text, "polic")
        l = tvg.node_by_index(362462)
        self.assertEqual(l.text, "Jay Wright (basketball)")
        with self.assertRaises(KeyError):
            tvg.node_by_index(5)

        l = tvg.node_by_text("polic")
        self.assertEqual(l.index, 1)
        l = tvg.node_by_text("Jay Wright (basketball)")
        self.assertEqual(l.index, 362462)
        with self.assertRaises(KeyError):
            tvg.node_by_text("should-not-exist")

        timestamps = []
        edges = []
        for g in tvg:
            self.assertEqual(g.revision, 0)
            timestamps.append(g.ts)
            edges.append(g.num_edges)

        self.assertEqual(timestamps, [      0,  130000,  141000,  164000,  176000,  272000,  376000,  465000,  666000,  682000,  696000,
                                       770000,  848000, 1217000, 1236000, 1257000, 1266000, 1431000, 1515000, 1539000, 1579000, 1626000,
                                      1763000, 1803000, 1834000, 1920000, 1967000, 2021000, 2188000, 2405000, 2482000, 2542000, 2551000,
                                      2583000, 2591000, 2604000, 2620000, 2830000, 2852000, 2957000, 3008000])

        self.assertEqual(edges, [155, 45, 1250, 90, 178, 85, 367, 98, 18, 528, 158, 201, 267, 214, 613, 567, 1, 137, 532, 59, 184,
                                 40, 99, 285, 326, 140, 173, 315, 211, 120, 19, 137, 170, 42, 135, 348, 168, 132, 147, 218, 321])

        g = tvg.lookup_near(141000)
        self.assertTrue(abs(g[6842, 249977] - 0.367879) < 1e-7)

        g = tvg.lookup_near(1257000)
        self.assertTrue(abs(g[1291, 3529] - 1.013476) < 1e-7)

        g = tvg.lookup_near(2604000)
        self.assertTrue(abs(g[121, 1154] - 3.000000) < 1e-7)

        tvg.compress(step=600000)

        timestamps = []
        edges = []
        for g in tvg:
            timestamps.append(g.ts)
            edges.append(g.num_edges)

        self.assertEqual(timestamps, [0, 600000, 1200000, 1800000, 2400000, 3000000])
        self.assertEqual(edges, [2226, 1172, 2446, 1448, 1632, 321])

        timestamps = []
        edges = []
        for g in reversed(tvg):
            timestamps.append(g.ts)
            edges.append(g.num_edges)

        self.assertEqual(timestamps, [3000000, 2400000, 1800000, 1200000, 600000, 0])
        self.assertEqual(edges, [321, 1632, 1448, 2446, 1172, 2226])

        tvg.compress(step=np.inf, offset=100000)

        timestamps = []
        edges = []
        for g in tvg:
            timestamps.append(g.ts)
            edges.append(g.num_edges)

        self.assertEqual(timestamps, [0])
        self.assertEqual(edges, [9097])

        del tvg

    def test_load_crlf(self):
        filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../datasets/example/example-tvg.graph")
        with open(filename, "rb") as fp:
            content = fp.read()

        temp_graph = tempfile.NamedTemporaryFile()
        temp_graph.write(content.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n'))
        temp_graph.flush()

        filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../datasets/example/example-tvg.nodes")
        with open(filename, "rb") as fp:
            content = fp.read()

        temp_nodes = tempfile.NamedTemporaryFile()
        temp_nodes.write(content.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n'))
        temp_nodes.flush()

        tvg = TVG.load(temp_graph.name, nodes=temp_nodes.name, primary_key=["text"])

        temp_graph.close()
        temp_nodes.close()

        l = tvg.node_by_index(1)
        self.assertEqual(l.text, "polic")
        l = tvg.node_by_index(362462)
        self.assertEqual(l.text, "Jay Wright (basketball)")
        with self.assertRaises(KeyError):
            tvg.node_by_index(5)

        l = tvg.node_by_text("polic")
        self.assertEqual(l.index, 1)
        l = tvg.node_by_text("Jay Wright (basketball)")
        self.assertEqual(l.index, 362462)
        with self.assertRaises(KeyError):
            tvg.node_by_text("should-not-exist")

        timestamps = []
        edges = []
        for g in tvg:
            self.assertEqual(g.revision, 0)
            timestamps.append(g.ts)
            edges.append(g.num_edges)

        self.assertEqual(timestamps, [      0,  130000,  141000,  164000,  176000,  272000,  376000,  465000,  666000,  682000,  696000,
                                       770000,  848000, 1217000, 1236000, 1257000, 1266000, 1431000, 1515000, 1539000, 1579000, 1626000,
                                      1763000, 1803000, 1834000, 1920000, 1967000, 2021000, 2188000, 2405000, 2482000, 2542000, 2551000,
                                      2583000, 2591000, 2604000, 2620000, 2830000, 2852000, 2957000, 3008000])

        self.assertEqual(edges, [155, 45, 1250, 90, 178, 85, 367, 98, 18, 528, 158, 201, 267, 214, 613, 567, 1, 137, 532, 59, 184,
                                 40, 99, 285, 326, 140, 173, 315, 211, 120, 19, 137, 170, 42, 135, 348, 168, 132, 147, 218, 321])

        del tvg

    def test_nodes(self):
        tvg = TVG()
        tvg.set_primary_key("text")

        l = tvg.Node(text="A", other="sample text")
        self.assertEqual(l.index, 0)
        self.assertEqual(l.text, "A")
        l = tvg.Node(text="B")
        self.assertEqual(l.index, 1)
        self.assertEqual(l.text, "B")
        l = tvg.Node(text="C")
        self.assertEqual(l.index, 2)
        self.assertEqual(l.text, "C")
        l = tvg.Node(text="D")
        self.assertEqual(l.index, 3)
        self.assertEqual(l.text, "D")

        l = tvg.Node(text="A")
        self.assertEqual(l.index, 0)
        self.assertEqual(l.text, "A")
        self.assertEqual(l['other'], "sample text")

        l = tvg.node_by_index(1)
        self.assertEqual(l.index, 1)
        self.assertEqual(l.text, "B")

        with self.assertRaises(KeyError):
            tvg.node_by_index(4)

        l = tvg.node_by_text("C")
        self.assertEqual(l.index, 2)
        self.assertEqual(l.text, "C")

        with self.assertRaises(KeyError):
            tvg.node_by_text("E")

        l.unlink()

        with self.assertRaises(KeyError):
            tvg.node_by_index(2)

        with self.assertRaises(KeyError):
            tvg.node_by_text("C")

        l = Node(text="Z")
        tvg.link_node(l, 4)
        self.assertEqual(l.index, 4)
        self.assertEqual(l.text, "Z")

        l = tvg.node_by_index(4)
        self.assertEqual(l.index, 4)
        self.assertEqual(l.text, "Z")

        l = tvg.node_by_text("Z")
        self.assertEqual(l.index, 4)
        self.assertEqual(l.text, "Z")

        l = Node(text="Z")
        self.assertEqual(l.index, 0xffffffffffffffff)
        with self.assertRaises(RuntimeError):
            tvg.link_node(l)
        self.assertEqual(l.index, 0xffffffffffffffff)

        del tvg

    def test_node_label(self):
        tvg = TVG()

        l = tvg.Node(label="A")
        self.assertEqual(tvg.node_label(l.index), "A")

        l = tvg.Node(norm="B")
        self.assertEqual(tvg.node_label(l.index), "B")

        l = tvg.Node(text="C")
        self.assertEqual(tvg.node_label(l.index), "C")

        l = tvg.Node(entity_name="D")
        self.assertEqual(tvg.node_label(l.index), "D")

        del tvg

    def test_node_attrs(self):
        tvg = TVG()
        tvg.set_primary_key("text")

        l = tvg.Node(text="sample text")
        self.assertEqual(l.as_dict(), {'text': "sample text"})

        with self.assertRaises(KeyError):
            l["text"] = "other text"

        with self.assertRaises(KeyError):
            l[""] = "empty key"

        l["attr1"] = "sample attr1"
        l["attr2"] = "sample attr2"
        l["attr3"] = "sample attr3"
        self.assertEqual(l.as_dict(), {'text': "sample text",
                                       'attr1': "sample attr1",
                                       'attr2': "sample attr2",
                                       'attr3': "sample attr3"})

        l["attr1"] = "other attr1"
        self.assertEqual(l.as_dict(), {'text': "sample text",
                                       'attr1': "other attr1",
                                       'attr2': "sample attr2",
                                       'attr3': "sample attr3"})

        tvg.set_primary_key(["text", "attr1"])
        with self.assertRaises(KeyError):
            l["attr1"] = "sample attr1"

        del tvg

    def test_readonly(self):
        tvg = TVG(positive=True)
        g = Graph(positive=True)
        self.assertEqual(g.readonly, False)
        tvg.link_graph(g, 0)
        self.assertEqual(g.ts, 0)
        self.assertEqual(g.readonly, True)

        with self.assertRaises(RuntimeError):
            g.clear()
        with self.assertRaises(MemoryError):
            g[0, 0] = 1.0
        with self.assertRaises(MemoryError):
            g.add_edge((0, 0), 1.0)
        with self.assertRaises(RuntimeError):
            del g[0, 0]
        with self.assertRaises(RuntimeError):
            g.mul_const(2.0)
        with self.assertRaises(RuntimeError):
            g.del_small()

        g.unlink()
        self.assertEqual(g.readonly, False)

        g.clear()
        g[0, 0] = 1.0
        g.add_edge((0, 0), 1.0)
        del g[0, 0]
        g.mul_const(2.0)
        g.del_small()

        del tvg
        del g

    def test_sum_edges(self):
        tvg = TVG(positive=True)
        tvg.verbosity = True
        self.assertEqual(tvg.verbosity, True)

        g = Graph()
        g[0, 0] = 1.0
        tvg.link_graph(g, 100)

        g = Graph()
        g[0, 1] = 2.0
        tvg.link_graph(g, 200)

        g = Graph()
        g[0, 2] = 3.0
        tvg.link_graph(g, 300)

        with self.assertRaises(MemoryError):
            tvg.sum_edges(1, 0)

        g = tvg.sum_edges(51, 150, eps=0.5)
        self.assertEqual(g.readonly, True)
        self.assertEqual(g[0, 0], 1.0)
        self.assertEqual(g[0, 1], 0.0)
        self.assertEqual(g[0, 2], 0.0)

        g = tvg.sum_edges(151, 250, eps=0.5)
        self.assertEqual(g[0, 0], 0.0)
        self.assertEqual(g[0, 1], 2.0)
        self.assertEqual(g[0, 2], 0.0)

        g = tvg.sum_edges(251, 350, eps=0.5)
        self.assertEqual(g[0, 0], 0.0)
        self.assertEqual(g[0, 1], 0.0)
        self.assertEqual(g[0, 2], 3.0)

        g = tvg.sum_edges(51, 150, eps=0.5)
        self.assertEqual(g[0, 0], 1.0)
        self.assertEqual(g[0, 1], 0.0)
        self.assertEqual(g[0, 2], 0.0)

        g = tvg.sum_edges(eps=0.5)
        self.assertEqual(g[0, 0], 1.0)
        self.assertEqual(g[0, 1], 2.0)
        self.assertEqual(g[0, 2], 3.0)

        g = tvg.sum_edges(-100, 0xffffffffffffffff + 100, eps=0.5)
        self.assertEqual(g[0, 0], 1.0)
        self.assertEqual(g[0, 1], 2.0)
        self.assertEqual(g[0, 2], 3.0)

        g = tvg.sum_edges(0x8000000000000000, 0xffffffffffffffff, eps=0.5)
        self.assertEqual(g.as_dict(), {})

        del tvg

    def test_sum_nodes(self):
        tvg = TVG(positive=True)
        tvg.verbosity = True
        self.assertEqual(tvg.verbosity, True)

        g = Graph()
        g[0, 0] = 1.0
        tvg.link_graph(g, 100)

        g = Graph()
        g[0, 1] = 2.0
        tvg.link_graph(g, 200)

        g = Graph()
        g[0, 2] = 3.0
        tvg.link_graph(g, 300)

        with self.assertRaises(MemoryError):
            tvg.sum_nodes(1, 0)

        v = tvg.sum_nodes(51, 150)
        self.assertEqual(v.readonly, True)
        self.assertEqual(v[0], 2.0)
        self.assertEqual(v[1], 0.0)
        self.assertEqual(v[2], 0.0)

        v = tvg.sum_nodes(151, 250)
        self.assertEqual(v[0], 1.0)
        self.assertEqual(v[1], 1.0)
        self.assertEqual(v[2], 0.0)

        v = tvg.sum_nodes(251, 350)
        self.assertEqual(v[0], 1.0)
        self.assertEqual(v[1], 0.0)
        self.assertEqual(v[2], 1.0)

        v = tvg.sum_nodes(51, 350)
        self.assertEqual(v[0], 4.0)
        self.assertEqual(v[1], 1.0)
        self.assertEqual(v[2], 1.0)

        v = tvg.sum_nodes()
        self.assertEqual(v[0], 4.0)
        self.assertEqual(v[1], 1.0)
        self.assertEqual(v[2], 1.0)

        v = tvg.sum_nodes(-100, 0xffffffffffffffff + 100)
        self.assertEqual(v[0], 4.0)
        self.assertEqual(v[1], 1.0)
        self.assertEqual(v[2], 1.0)

        v = tvg.sum_nodes(0x8000000000000000, 0xffffffffffffffff)
        self.assertEqual(v.as_dict(), {})

        del tvg

    def test_sum_edges_exp_precision(self):
        tvg = TVG(positive=True)
        beta = 0.3

        g = Graph()
        g[0, 0] = 1.0
        tvg.link_graph(g, 0)

        with self.assertRaises(MemoryError):
            tvg.sum_edges_exp(1, 0, beta=beta)

        g = tvg.sum_edges_exp(0, 100, beta=beta)
        self.assertEqual(g.readonly, True)
        self.assertTrue(abs(g[0, 0] - math.pow(beta, 100.0)) < 1e-7)

        g = tvg.sum_edges_exp(0, 0, beta=beta)
        self.assertTrue(abs(g[0, 0] - 1.0) < 1e-7)

        del g

    def test_sum_edges_exp_norm(self):
        source = np.random.rand(100)
        beta = 0.3

        tvg = TVG(positive=True)

        for t, s in enumerate(source):
            g = Graph()
            g[0, 0] = s
            tvg.link_graph(g, t)

        with self.assertRaises(MemoryError):
            tvg.sum_edges_exp_norm(1, 0, beta=beta)

        expected = 0.0
        for t, s in enumerate(source):
            g = tvg.sum_edges_exp_norm(0, t, beta=beta)
            expected = beta * expected + (1.0 - beta) * s
            self.assertTrue(abs(g[0, 0] - expected) < 1e-6)

        expected = 0.0
        for t, s in enumerate(source):
            g = tvg.sum_edges_exp_norm(0, t, log_beta=math.log(beta))
            expected = beta * expected + (1.0 - beta) * s
            self.assertTrue(abs(g[0, 0] - expected) < 1e-6)

        del tvg

    def test_count_edges(self):
        tvg = TVG(positive=True)

        g = Graph()
        g[0, 0] = 1.0
        tvg.link_graph(g, 100)

        g = Graph()
        g[0, 1] = 2.0
        tvg.link_graph(g, 200)

        g = Graph()
        g[0, 2] = 3.0
        tvg.link_graph(g, 300)

        with self.assertRaises(MemoryError):
            tvg.count_edges(1, 0)

        g = tvg.count_edges(51, 150)
        self.assertEqual(g.readonly, True)
        self.assertEqual(g[0, 0], 1.0)
        self.assertEqual(g[0, 1], 0.0)
        self.assertEqual(g[0, 2], 0.0)

        g = tvg.count_edges(151, 250)
        self.assertEqual(g[0, 0], 0.0)
        self.assertEqual(g[0, 1], 1.0)
        self.assertEqual(g[0, 2], 0.0)

        g = tvg.count_edges(251, 350)
        self.assertEqual(g[0, 0], 0.0)
        self.assertEqual(g[0, 1], 0.0)
        self.assertEqual(g[0, 2], 1.0)

        g = tvg.count_edges(51, 150)
        self.assertEqual(g[0, 0], 1.0)
        self.assertEqual(g[0, 1], 0.0)
        self.assertEqual(g[0, 2], 0.0)

        g = tvg.count_edges()
        self.assertEqual(g[0, 0], 1.0)
        self.assertEqual(g[0, 1], 1.0)
        self.assertEqual(g[0, 2], 1.0)

        g = tvg.count_edges(-100, 0xffffffffffffffff + 100)
        self.assertEqual(g[0, 0], 1.0)
        self.assertEqual(g[0, 1], 1.0)
        self.assertEqual(g[0, 2], 1.0)

        g = tvg.count_edges(0x8000000000000000, 0xffffffffffffffff)
        self.assertEqual(g.as_dict(), {})

        del tvg

    def test_count_nodes(self):
        tvg = TVG(positive=True)

        g = Graph()
        g[0, 0] = 1.0
        tvg.link_graph(g, 100)

        g = Graph()
        g[0, 1] = 2.0
        g[1, 2] = 2.0
        tvg.link_graph(g, 200)

        g = Graph()
        g[0, 2] = 3.0
        tvg.link_graph(g, 300)

        with self.assertRaises(MemoryError):
            tvg.count_nodes(1, 0)

        v = tvg.count_nodes(51, 150)
        self.assertEqual(v.readonly, True)
        self.assertEqual(v[0], 1.0)
        self.assertEqual(v[1], 0.0)
        self.assertEqual(v[2], 0.0)

        v = tvg.count_nodes(151, 250)
        self.assertEqual(v[0], 1.0)
        self.assertEqual(v[1], 1.0)
        self.assertEqual(v[2], 1.0)

        v = tvg.count_nodes(251, 350)
        self.assertEqual(v[0], 1.0)
        self.assertEqual(v[1], 0.0)
        self.assertEqual(v[2], 1.0)

        v = tvg.count_nodes(51, 150)
        self.assertEqual(v[0], 1.0)
        self.assertEqual(v[1], 0.0)
        self.assertEqual(v[2], 0.0)

        v = tvg.count_nodes()
        self.assertEqual(v[0], 3.0)
        self.assertEqual(v[1], 1.0)
        self.assertEqual(v[2], 2.0)

        v = tvg.count_nodes(-100, 0xffffffffffffffff + 100)
        self.assertEqual(v[0], 3.0)
        self.assertEqual(v[1], 1.0)
        self.assertEqual(v[2], 2.0)

        v = tvg.count_nodes(0x8000000000000000, 0xffffffffffffffff)
        self.assertEqual(v.as_dict(), {})

        del tvg

    def test_count_graphs(self):
        tvg = TVG(positive=True)
        tvg.enable_query_cache(cache_size=0x8000) # 32 kB cache

        g = Graph()
        tvg.link_graph(g, 100)

        g = Graph()
        tvg.link_graph(g, 200)
        g = Graph()
        tvg.link_graph(g, 200)

        g = Graph()
        tvg.link_graph(g, 300)
        g = Graph()
        tvg.link_graph(g, 300)
        g = Graph()
        tvg.link_graph(g, 300)

        with self.assertRaises(MemoryError):
            tvg.count_graphs(1, 0)

        c = tvg.count_graphs(51, 150)
        self.assertEqual(c, 1)

        c = tvg.count_graphs(151, 250)
        self.assertEqual(c, 2)

        c = tvg.count_graphs(251, 350)
        self.assertEqual(c, 3)

        c = tvg.count_graphs(51, 150)
        self.assertEqual(c, 1)

        c = tvg.count_graphs(51, 350)
        self.assertEqual(c, 6)

        c = tvg.count_graphs()
        self.assertEqual(c, 6)

        c = tvg.count_graphs(-100, 0xffffffffffffffff + 100)
        self.assertEqual(c, 6)

        v = tvg.count_graphs(0x8000000000000000, 0xffffffffffffffff)
        self.assertEqual(v, 0)

        del tvg

    def test_topics(self):
        tvg = TVG(positive=True)
        tvg.verbosity = True

        g = Graph()
        g[0, 1] = 1.0
        tvg.link_graph(g, 100)

        g = Graph()
        g[0, 1] = 1.0
        tvg.link_graph(g, 200)

        g = Graph()
        g[1, 2] = 0.5
        tvg.link_graph(g, 200)

        g = Graph()
        g[0, 1] = 0.5
        tvg.link_graph(g, 300)

        g = Graph()
        g[0, 1] = 1.0
        tvg.link_graph(g, 300)

        g = Graph()
        g[1, 2] = 1.0
        tvg.link_graph(g, 300)

        with self.assertRaises(MemoryError):
            tvg.topics(1, 0)

        # |D(0) \cup D(1)| = 1.0
        # |D((0, 1))| = 1.0
        # |L((0, 1))| = 1.0
        # \sum exp(-\delta) = 1.0

        g = tvg.topics(51, 150)
        self.assertTrue(abs(g[0, 1] - 1.0) < 1e-7)

        # |D(0) \cup D(1)| = 2.0
        # |D((0, 1))| = 1.0
        # |L((0, 1))| = 1.0
        # \sum exp(-\delta) = 1.0

        g = tvg.topics(151, 250)
        self.assertTrue(abs(g[0, 1] - 2.0 / 3.0) < 1e-7)

        # |D(0) \cup D(1)| = 3.0
        # |D((0, 1))| = 2.0
        # |L((0, 1))| = 2.0
        # \sum exp(-\delta) = 1.5

        g = tvg.topics(251, 350)
        self.assertTrue(abs(g[0, 1] - 12.0 / 17.0) < 1e-7)

        # |D(0) \cup D(1)| = 6.0
        # |D((0, 1))| = 4.0
        # |L((0, 1))| = 4.0
        # \sum exp(-\delta) = 3.5

        g = tvg.topics(51, 350)
        self.assertTrue(abs(g[0, 1] - 28.0 / 37.0) < 1e-7)

        g = tvg.topics(0, 350)
        self.assertTrue(abs(g[0, 1] - 28.0 / 37.0) < 1e-7)

        g = tvg.topics(0, 400)
        self.assertTrue(abs(g[0, 1] - 28.0 / 37.0) < 1e-7)

        g = tvg.topics()
        self.assertTrue(abs(g[0, 1] - 28.0 / 37.0) < 1e-7)

        g = tvg.topics(-100, 0xffffffffffffffff + 100)
        self.assertTrue(abs(g[0, 1] - 28.0 / 37.0) < 1e-7)

        g = tvg.topics(0x8000000000000000, 0xffffffffffffffff)
        self.assertEqual(g.as_dict(), {})

        # |D(0) \cup D(1)| = 1.0
        # |D((0, 1))| = 1.0
        # |L((0, 1))| = 1.0
        # \sum exp(-\delta) = 1.0
        # \delta T = 1.0
        # |T(e)| = 1.0

        with CaptureStderr() as stderr:
            g = tvg.topics(51, 150, step=100, offset=51)
        self.assertTrue(abs(g[0, 1] - 1.0) < 1e-7)

        snapshots = re.findall("\\[([0-9]+, [0-9]+)\\]", stderr.output)
        self.assertEqual(snapshots, ["51, 150"])

        # |D(0) \cup D(1)| = 2.0
        # |D((0, 1))| = 1.0
        # |L((0, 1))| = 1.0
        # \sum exp(-\delta) = 1.0
        # \delta T = 1.0
        # |T(e)| = 1.0

        with CaptureStderr() as stderr:
            g = tvg.topics(151, 250, step=100, offset=51)
        self.assertTrue(abs(g[0, 1] - 3.0 / 4.0) < 1e-7)

        snapshots = re.findall("\\[([0-9]+, [0-9]+)\\]", stderr.output)
        self.assertEqual(snapshots, ["151, 250"])

        # |D(0) \cup D(1)| = 3.0
        # |D((0, 1))| = 2.0
        # |L((0, 1))| = 2.0
        # \sum exp(-\delta) = 1.5
        # \delta T = 1.0
        # |T(e)| = 1

        with CaptureStderr() as stderr:
            g = tvg.topics(251, 350, step=100, offset=51)
        self.assertTrue(abs(g[0, 1] - 18.0 / 23.0) < 1e-7)

        snapshots = re.findall("\\[([0-9]+, [0-9]+)\\]", stderr.output)
        self.assertEqual(snapshots, ["251, 350"])

        # |D(0) \cup D(1)| = 6.0
        # |D((0, 1))| = 4.0
        # |L((0, 1))| = 4.0
        # \sum exp(-\delta) = 3.5
        # \delta T = 3.0
        # |T(e)| = 3.0

        with CaptureStderr() as stderr:
            g = tvg.topics(51, 350, step=100, offset=51)
        self.assertTrue(abs(g[0, 1] - 14.0 / 17.0) < 1e-7)

        snapshots = re.findall("\\[([0-9]+, [0-9]+)\\]", stderr.output)
        self.assertEqual(snapshots, ["51, 150", "151, 250", "251, 350"])

        # |D(0) \cup D(1)| = 6.0
        # |D((0, 1))| = 4.0
        # |L((0, 1))| = 4.0
        # \sum exp(-\delta) = 3.5
        # \delta T = 4.0
        # |T(e)| = 3.0

        with CaptureStderr() as stderr:
            g = tvg.topics(0, 350, step=100, offset=51)
        self.assertTrue(abs(g[0, 1] - 126.0 / 167.0) < 1e-7)

        snapshots = re.findall("\\[([0-9]+, [0-9]+)\\]", stderr.output)
        self.assertEqual(snapshots, ["0, 50", "51, 150", "151, 250", "251, 350"])

        # |D(0) \cup D(1)| = 6.0
        # |D((0, 1))| = 4.0
        # |L((0, 1))| = 4.0
        # \sum exp(-\delta) = 3.5
        # \delta T = 5.0
        # |T(e)| = 3.0

        with CaptureStderr() as stderr:
            g = tvg.topics(0, 400, step=100, offset=51)
        self.assertTrue(abs(g[0, 1] - 126.0 / 181.0) < 1e-7)

        snapshots = re.findall("\\[([0-9]+, [0-9]+)\\]", stderr.output)
        self.assertEqual(snapshots, ["0, 50", "51, 150", "151, 250", "251, 350", "351, 400"])

        # Repeat the test with the samples parameter instead of step and offset.
        with CaptureStderr() as stderr:
            g = tvg.topics(0, 400, samples=UniformSamples(step=100, offset=51))
        self.assertTrue(abs(g[0, 1] - 126.0 / 181.0) < 1e-7)

        snapshots = re.findall("\\[([0-9]+, [0-9]+)\\]", stderr.output)
        self.assertEqual(snapshots, ["0, 50", "51, 150", "151, 250", "251, 350", "351, 400"])

        del tvg

    def test_topics_sum_weights(self):
        tvg = TVG(positive=True)

        # In the original publication, the metric is only defined
        # for sum_weights=False. In our implementation, however,
        # we also allow sum_weights=True, i.e., weights greater than 1.

        g = Graph()
        g[0, 1] = 50.0
        tvg.link_graph(g, 100)

        g = Graph()
        g[0, 1] = 50.0
        tvg.link_graph(g, 100)

        g = Graph()
        g[1, 2] = 1.0
        tvg.link_graph(g, 100)

        # |D(0) \cup D(1)| = 3.0
        # |D((0, 1))| = 2.0
        # |L((0, 1))| = 2.0
        # \sum exp(-\delta) = 100.0

        g = tvg.topics(51, 150)
        self.assertTrue(abs(g[0, 1] - 25.0 / 19.0) < 1e-7)

        # |D(0) \cup D(1)| = 3.0
        # |D((0, 1))| = 2.0
        # |L((0, 1))| = 2.0
        # \sum exp(-\delta) = 100.0
        # \delta T = 1.0
        # |T(e)| = 1

        g = tvg.topics(51, 150, step=100, offset=51)
        self.assertTrue(abs(g[0, 1] - 25.0 / 21.0) < 1e-7)

        del tvg

    def test_query_cache(self):
        tvg = TVG(positive=True)
        tvg.verbosity = True

        g = Graph()
        g[0, 0] = 1.0
        tvg.link_graph(g, 100)

        g = Graph()
        g[0, 1] = 2.0
        tvg.link_graph(g, 200)

        g = Graph()
        g[0, 2] = 3.0
        tvg.link_graph(g, 300)

        # test 1
        with CaptureStderr() as stderr:
            g = tvg.sum_edges(51, 150, eps=0.5)
        self.assertIn("0 queries and 1 graphs", stderr.output)
        self.assertEqual(g[0, 0], 1.0)
        self.assertEqual(g[0, 1], 0.0)
        self.assertEqual(g[0, 2], 0.0)
        del g
        with CaptureStderr() as stderr:
            g = tvg.sum_edges(51, 150, eps=0.5)
        self.assertIn("0 queries and 1 graphs", stderr.output)
        self.assertEqual(g[0, 0], 1.0)
        self.assertEqual(g[0, 1], 0.0)
        self.assertEqual(g[0, 2], 0.0)
        del g

        # test 2
        with CaptureStderr() as stderr:
            g1 = tvg.sum_edges(51, 150, eps=0.5)
        self.assertIn("0 queries and 1 graphs", stderr.output)
        with CaptureStderr() as stderr:
            g2 = tvg.sum_edges(51, 150, eps=0.5)
        self.assertIn("1 queries and 0 graphs", stderr.output)
        self.assertEqual(g2[0, 0], 1.0)
        self.assertEqual(g2[0, 1], 0.0)
        self.assertEqual(g2[0, 2], 0.0)
        del g2
        tvg.invalidate_queries(100, 100)
        with CaptureStderr() as stderr:
            g2 = tvg.sum_edges(51, 150, eps=0.5)
        self.assertIn("0 queries and 1 graphs", stderr.output)
        self.assertEqual(g2[0, 0], 1.0)
        self.assertEqual(g2[0, 1], 0.0)
        self.assertEqual(g2[0, 2], 0.0)
        del g2
        del g1

        tvg.enable_query_cache(cache_size=0x8000) # 32 kB cache

        # test 3
        with CaptureStderr() as stderr:
            g = tvg.sum_edges(51, 150, eps=0.5)
        self.assertIn("0 queries and 1 graphs", stderr.output)
        self.assertEqual(g[0, 0], 1.0)
        self.assertEqual(g[0, 1], 0.0)
        self.assertEqual(g[0, 2], 0.0)
        del g
        with CaptureStderr() as stderr:
            g = tvg.sum_edges(51, 150, eps=0.5)
        self.assertIn("1 queries and 0 graphs", stderr.output)
        self.assertEqual(g[0, 0], 1.0)
        self.assertEqual(g[0, 1], 0.0)
        self.assertEqual(g[0, 2], 0.0)
        del g
        tvg.invalidate_queries(100, 100)
        with CaptureStderr() as stderr:
            g = tvg.sum_edges(51, 150, eps=0.5)
        self.assertIn("0 queries and 1 graphs", stderr.output)
        self.assertEqual(g[0, 0], 1.0)
        self.assertEqual(g[0, 1], 0.0)
        self.assertEqual(g[0, 2], 0.0)
        del g

        tvg.disable_query_cache()
        tvg.enable_query_cache(cache_size=0x8000) # 32 kB cache

        # test 4
        with CaptureStderr() as stderr:
            g1 = tvg.sum_edges(51, 150, eps=0.5)
        self.assertIn("0 queries and 1 graphs", stderr.output)
        with CaptureStderr() as stderr:
            g2 = tvg.sum_edges(51, 150, eps=0.5)
        self.assertIn("1 queries and 0 graphs", stderr.output)
        self.assertEqual(g2[0, 0], 1.0)
        self.assertEqual(g2[0, 1], 0.0)
        self.assertEqual(g2[0, 2], 0.0)
        del g2
        tvg.invalidate_queries(100, 100)
        with CaptureStderr() as stderr:
            g2 = tvg.sum_edges(51, 150, eps=0.5)
        self.assertIn("0 queries and 1 graphs", stderr.output)
        self.assertEqual(g2[0, 0], 1.0)
        self.assertEqual(g2[0, 1], 0.0)
        self.assertEqual(g2[0, 2], 0.0)
        del g2
        del g1

        del tvg

    def test_sparse_topics(self):
        tvg = TVG(positive=True)

        g = Graph()
        g[0, 1] = 1.0
        g[0, 2] = 0.5
        g[1, 2] = 0.5
        g[0, 3] = 0.25
        g[1, 3] = 0.25
        tvg.link_graph(g, 100)

        g = tvg.topics(51, 150)
        h = g.sparse_subgraph(num_seeds=1, num_neighbors=1)
        self.assertEqual(h.num_edges, 3)
        self.assertTrue(abs(h[0, 1] - 1.0) < 1e-7)
        self.assertTrue(abs(h[0, 2] - 2.0 / 3.0) < 1e-7)
        self.assertTrue(abs(h[1, 2] - 2.0 / 3.0) < 1e-7)

        del tvg

    def test_sample_eigenvectors(self):
        tvg = TVG(positive=True)

        g = Graph()
        g[0, 0] = 1.0
        tvg.link_graph(g, 100)

        g = Graph()
        g[1, 1] = 2.0
        tvg.link_graph(g, 200)

        g = Graph()
        g[2, 2] = 3.0
        tvg.link_graph(g, 300)

        values = tvg.sample_eigenvectors(50, 350, sample_width=101, sample_steps=3)
        values = _convert_values(values)
        self.assertEqual(len(values), 3)
        self.assertEqual(values[0], [1.0, 0.0, 0.0])
        self.assertEqual(values[1], [0.0, 1.0, 0.0])
        self.assertEqual(values[2][0], 0.0)
        self.assertEqual(values[2][1], 0.0)
        self.assertTrue(abs(values[2][2] - 1.0) < 1e-6)

        values = tvg.sample_eigenvectors(50, 350, sample_width=101, sample_steps=3, method='sum_edges')
        values = _convert_values(values)
        self.assertEqual(len(values), 3)
        self.assertEqual(values[0], [1.0, 0.0, 0.0])
        self.assertEqual(values[1], [0.0, 1.0, 0.0])
        self.assertEqual(values[2][0], 0.0)
        self.assertEqual(values[2][1], 0.0)
        self.assertTrue(abs(values[2][2] - 1.0) < 1e-6)

        values = tvg.sample_eigenvectors(50, 350, sample_width=101, sample_steps=3, method='count_edges')
        values = _convert_values(values)
        self.assertEqual(len(values), 3)
        self.assertEqual(values[0], [1.0, 0.0, 0.0])
        self.assertEqual(values[1], [0.0, 1.0, 0.0])
        self.assertEqual(values[2][0], 0.0)
        self.assertEqual(values[2][1], 0.0)
        self.assertTrue(abs(values[2][2] - 1.0) < 1e-6)

        values = tvg.sample_eigenvectors(50, 350, sample_width=101, sample_steps=3, method='topics')
        values = _convert_values(values)
        self.assertEqual(len(values), 3)
        self.assertEqual(values[0], [1.0, 0.0, 0.0])
        self.assertEqual(values[1], [0.0, 1.0, 0.0])
        self.assertEqual(values[2][0], 0.0)
        self.assertEqual(values[2][1], 0.0)
        self.assertTrue(abs(values[2][2] - 1.0) < 1e-6)

        del tvg

    def test_sample_graphs(self):
        tvg = TVG(positive=True)

        g = Graph()
        g[0, 0] = 1.0
        tvg.link_graph(g, 100)

        g = Graph()
        g[1, 1] = 2.0
        tvg.link_graph(g, 200)

        g = Graph()
        g[2, 2] = 3.0
        tvg.link_graph(g, 300)

        values = tvg.sample_graphs(50, 350, sample_width=101, sample_steps=3)
        values = _convert_values(values)
        self.assertEqual(len(values), 3)
        self.assertEqual(values[0, 0], [1.0, 0.0, 0.0])
        self.assertEqual(values[1, 1], [0.0, 2.0, 0.0])
        self.assertEqual(values[2, 2], [0.0, 0.0, 3.0])

        values = tvg.sample_graphs(50, 350, sample_width=101, sample_steps=3, method='sum_edges')
        values = _convert_values(values)
        self.assertEqual(len(values), 3)
        self.assertEqual(values[0, 0], [1.0, 0.0, 0.0])
        self.assertEqual(values[1, 1], [0.0, 2.0, 0.0])
        self.assertEqual(values[2, 2], [0.0, 0.0, 3.0])

        values = tvg.sample_graphs(50, 350, sample_width=101, sample_steps=3, method='count_edges')
        values = _convert_values(values)
        self.assertEqual(len(values), 3)
        self.assertEqual(values[0, 0], [1.0, 0.0, 0.0])
        self.assertEqual(values[1, 1], [0.0, 1.0, 0.0])
        self.assertEqual(values[2, 2], [0.0, 0.0, 1.0])

        values = tvg.sample_graphs(50, 350, sample_width=101, sample_steps=3, method='topics')
        values = _convert_values(values)
        self.assertEqual(len(values), 3)
        self.assertEqual(values[0, 0], [1.0, 0.0, 0.0])
        self.assertEqual(values[1, 1][0], 0.0)
        self.assertTrue(abs(values[1, 1][1] - 4.0 / 3.0) < 1e-6)
        self.assertEqual(values[1, 1][2], 0.0)
        self.assertEqual(values[2, 2], [0.0, 0.0, 1.5])

        del tvg

    def test__convert_values(self):
        values = [
            {(0, 0): 1.0, (0, 1): 0.0, (1, 1): 2.0, (2, 2): 2.0},
            {(0, 0): 1.0, (0, 1): 1.0, (1, 1): 1.0, (2, 2): 2.0},
            {(0, 0): 1.0, (0, 1): 2.0, (1, 1): 0.0, (2, 2): 2.0},
        ]

        expected = {
            (0, 0): [1.0, 1.0, 1.0],
            (0, 1): [0.0, 1.0, 2.0],
            (1, 1): [2.0, 1.0, 0.0],
            (2, 2): [2.0, 2.0, 2.0],
        }

        result = _convert_values(values)
        self.assertEqual(result, expected)

    def test_metric_entropy(self):
        tvg = TVG(positive=True)

        g = Graph()
        g[0, 0] = g[0, 1] = g[0, 2] = 1.0
        g[1, 1] = g[1, 2] = 1.0
        g[2, 2] = 1.0
        tvg.link_graph(g, 100)

        g = Graph()
        g[1, 1] = g[1, 2] = 2.0
        g[2, 2] = 2.0
        tvg.link_graph(g, 200)

        g = Graph()
        g[2, 2] = 3.0
        tvg.link_graph(g, 300)

        values = tvg.sample_eigenvectors(50, 350, sample_width=101, sample_steps=3)
        values = metric_entropy(values, num_bins=6)

        P = np.array([3, 0, 0, 3, 2, 1]) / 9.0
        self.assertEqual(len(values), 3)
        self.assertEqual(values[0], - np.log(P[3]) * P[3] - np.log(P[0]) * P[0] - np.log(P[0]) * P[0])
        self.assertEqual(values[1], - np.log(P[3]) * P[3] - np.log(P[4]) * P[4] - np.log(P[0]) * P[0])
        self.assertEqual(values[2], - np.log(P[3]) * P[3] - np.log(P[4]) * P[4] - np.log(P[5]) * P[5])

        del tvg

    def test_metric_entropy_edges(self):
        values = [
            {(0, 0): 1.0, (0, 1): 0.0, (1, 1): 2.0, (2, 2): 2.0},
            {(0, 0): 1.0, (0, 1): 1.0, (1, 1): 1.0, (2, 2): 2.0},
            {(0, 0): 1.0, (0, 1): 2.0, (1, 1): 0.0, (2, 2): 2.0},
        ]

        result = metric_entropy(values, num_bins=2)
        self.assertEqual(len(result), 4)
        self.assertTrue(abs(result[0, 0] - 0.89587973) < 1e-7)
        self.assertTrue(abs(result[0, 1] - 0.74918778) < 1e-7)
        self.assertTrue(abs(result[1, 1] - 0.74918778) < 1e-7)
        self.assertTrue(abs(result[2, 2] - 0.45580389) < 1e-7)

    def test_metric_entropy_local(self):
        tvg = TVG(positive=True)

        g = Graph()
        g[0, 0] = g[0, 1] = g[0, 2] = 1.0
        g[1, 1] = g[1, 2] = 1.0
        g[2, 2] = 1.0
        tvg.link_graph(g, 100)

        g = Graph()
        g[1, 1] = g[1, 2] = 2.0
        g[2, 2] = 2.0
        tvg.link_graph(g, 200)

        g = Graph()
        g[2, 2] = 3.0
        tvg.link_graph(g, 300)

        values = tvg.sample_eigenvectors(50, 350, sample_width=101, sample_steps=3)
        values = metric_entropy_local(values, num_bins=2)

        P0 = 1.0
        P1 = np.array([1, 2]) / 3.0
        P2 = np.array([2, 1]) / 3.0
        self.assertEqual(len(values), 3)
        self.assertEqual(values[0], - np.log(P0) * P0 - np.log(P1[0]) * P1[0] - np.log(P2[0]) * P2[0])
        self.assertEqual(values[1], - np.log(P0) * P0 - np.log(P1[1]) * P1[1] - np.log(P2[0]) * P2[0])
        self.assertEqual(values[2], - np.log(P0) * P0 - np.log(P1[1]) * P1[1] - np.log(P2[1]) * P2[1])

        del tvg

    def test_metric_entropy_local_edges(self):
        values = [
            {(0, 0): 1.0, (0, 1): 0.0, (1, 1): 2.0, (2, 2): 2.0},
            {(0, 0): 1.0, (0, 1): 1.0, (1, 1): 1.0, (2, 2): 2.0},
            {(0, 0): 1.0, (0, 1): 2.0, (1, 1): 0.0, (2, 2): 2.0},
        ]

        result = metric_entropy_local(values, num_bins=2)
        self.assertEqual(len(result), 4)
        self.assertTrue(abs(result[0, 0] - 0.90890873) < 1e-7)
        self.assertTrue(abs(result[0, 1] - 0.77809669) < 1e-7)
        self.assertTrue(abs(result[1, 1] - 0.77809669) < 1e-7)
        self.assertTrue(abs(result[2, 2] - 0.77809669) < 1e-7)

    def test_metric_entropy_2d(self):
        tvg = TVG(positive=True)

        g = Graph()
        g[0, 0] = g[0, 1] = g[0, 2] = 1.0
        g[1, 1] = g[1, 2] = 1.0
        g[2, 2] = 1.0
        tvg.link_graph(g, 100)

        g = Graph()
        g[1, 1] = g[1, 2] = 2.0
        g[2, 2] = 2.0
        tvg.link_graph(g, 200)

        g = Graph()
        g[2, 2] = 3.0
        tvg.link_graph(g, 300)

        values = tvg.sample_eigenvectors(50, 350, sample_width=101, sample_steps=3)
        values = metric_entropy_2d(values, num_bins=2)

        P = np.array([[1, 0], [2, 3]]) / 6.0
        self.assertEqual(len(values), 3)
        self.assertEqual(values[0], - np.log(P[1, 0]) * P[1, 0] - np.log(P[0, 0]) * P[0, 0])
        self.assertEqual(values[1], - np.log(P[1, 1]) * P[1, 1] - np.log(P[1, 0]) * P[1, 0])
        self.assertEqual(values[2], - np.log(P[1, 1]) * P[1, 1] - np.log(P[1, 1]) * P[1, 1])

        del tvg

    def test_metric_entropy_2d_edges(self):
        values = [
            {(0, 0): 1.0, (0, 1): 0.0, (1, 1): 2.0, (2, 2): 2.0},
            {(0, 0): 1.0, (0, 1): 1.0, (1, 1): 1.0, (2, 2): 2.0},
            {(0, 0): 1.0, (0, 1): 2.0, (1, 1): 0.0, (2, 2): 2.0},
        ]

        result = metric_entropy_2d(values, num_bins=2)
        self.assertEqual(len(result), 4)
        self.assertTrue(abs(result[0, 0] - 0.0) < 1e-7)
        self.assertTrue(abs(result[0, 1] - 0.25993019) < 1e-7)
        self.assertTrue(abs(result[1, 1] - 0.25993019) < 1e-7)
        self.assertTrue(abs(result[2, 2] - 0.43152310) < 1e-7)

    def test_metric_trend(self):
        tvg = TVG(positive=True)

        g = Graph()
        g[0, 0] = g[0, 1] = g[0, 2] = 1.0
        g[1, 1] = g[1, 2] = 1.0
        g[2, 2] = 1.0
        tvg.link_graph(g, 100)

        g = Graph()
        g[1, 1] = g[1, 2] = 2.0
        g[2, 2] = 2.0
        tvg.link_graph(g, 200)

        g = Graph()
        g[2, 2] = 3.0
        tvg.link_graph(g, 300)

        values = tvg.sample_eigenvectors(50, 350, sample_width=101, sample_steps=3)
        values = metric_trend(values)

        self.assertEqual(len(values), 3)
        self.assertTrue(abs(values[0] + 0.288675129) < 1e-7)
        self.assertTrue(abs(values[1] + 0.288675129) < 1e-7)
        self.assertTrue(abs(values[2] - 0.211324870) < 1e-7)

        del tvg

    def test_metric_trend_edges(self):
        values = [
            {(0, 0): 1.0, (0, 1): 0.0, (1, 1): 2.0, (2, 2): 2.0},
            {(0, 0): 1.0, (0, 1): 1.0, (1, 1): 1.0, (2, 2): 2.0},
            {(0, 0): 1.0, (0, 1): 2.0, (1, 1): 0.0, (2, 2): 2.0},
        ]

        result = metric_trend(values)
        self.assertEqual(len(result), 4)
        self.assertTrue(abs(result[0, 0] - 0.0) < 1e-7)
        self.assertTrue(abs(result[0, 1] - 1.0) < 1e-7)
        self.assertTrue(abs(result[1, 1] + 1.0) < 1e-7)
        self.assertTrue(abs(result[2, 2] - 0.0) < 1e-7)

    def test_metric_stability_ratio_edges(self):
        values = [
            {(0, 0): 1.0, (0, 1): 0.0, (1, 1): 2.0, (2, 2): 2.0, (3, 3): 0.0},
            {(0, 0): 1.0, (0, 1): 1.0, (1, 1): 1.0, (2, 2): 2.0, (3, 3): 0.0},
            {(0, 0): 1.0, (0, 1): 2.0, (1, 1): 0.0, (2, 2): 2.0, (3, 3): 0.0},
        ]

        result = metric_stability_ratio(values)
        self.assertEqual(len(result), 5)
        self.assertEqual(result[0, 0], np.inf)
        self.assertTrue(abs(result[0, 1] - 1.22474487) < 1e-7)
        self.assertTrue(abs(result[1, 1] - 1.22474487) < 1e-7)
        self.assertEqual(result[2, 2], np.inf)
        self.assertEqual(result[3, 3], np.inf)

    def test_metric_stability_pareto(self):
        tvg = TVG(positive=True)

        g = Graph()
        g[0, 0] = g[0, 1] = g[0, 2] = 1.0
        g[1, 1] = g[1, 2] = 1.0
        g[2, 2] = 1.0
        tvg.link_graph(g, 100)

        g = Graph()
        g[1, 1] = g[1, 2] = 2.0
        g[2, 2] = 2.0
        tvg.link_graph(g, 200)

        g = Graph()
        g[2, 2] = 3.0
        tvg.link_graph(g, 300)

        values = tvg.sample_eigenvectors(50, 350, sample_width=101, sample_steps=3)
        values = metric_stability_pareto(values)

        self.assertEqual(len(values), 3)
        self.assertEqual(values[2], 1.0)
        self.assertEqual(values[0], 2.0)
        self.assertEqual(values[1], 2.0)

        del tvg

    def test_metric_stability_pareto_edges(self):
        values = [
            {(0, 0): 1.0, (0, 1): 0.0, (1, 1): 2.0, (2, 2): 2.0},
            {(0, 0): 1.0, (0, 1): 1.0, (1, 1): 1.0, (2, 2): 2.0},
            {(0, 0): 1.0, (0, 1): 2.0, (1, 1): 0.0, (2, 2): 2.0},
        ]

        result = metric_stability_pareto(values)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0, 0], 2.0)
        self.assertEqual(result[0, 1], 3.0)
        self.assertEqual(result[1, 1], 3.0)
        self.assertEqual(result[2, 2], 1.0)

        # Fast-path for list of Graphs
        graphs = [Graph.from_dict(v) for v in values]
        result = metric_stability_pareto(graphs)
        self.assertTrue(isinstance(result, Graph))
        self.assertEqual(result.as_dict(), {(0, 0): 2.0, (0, 1): 3.0,
                                            (1, 1): 3.0, (2, 2): 1.0})

        result = metric_stability_pareto(graphs, base=0.5)
        self.assertTrue(isinstance(result, Graph))
        self.assertEqual(result.as_dict(), {(0, 0): 0.5,  (0, 1): 0.25,
                                            (1, 1): 0.25, (2, 2): 1.0})
        del graphs

        values = [
            {0: 1.0, 1: 0.0, 2: 2.0, 3: 2.0},
            {0: 1.0, 1: 1.0, 2: 1.0, 3: 2.0},
            {0: 1.0, 1: 2.0, 2: 0.0, 3: 2.0},
        ]

        # Fast-path for list of Vectors
        vectors = [Vector.from_dict(v) for v in values]
        result = metric_stability_pareto(vectors)
        self.assertTrue(isinstance(result, Vector))
        self.assertEqual(result.as_dict(), {0: 2.0, 1: 3.0, 2: 3.0, 3: 1.0})

        result = metric_stability_pareto(vectors, base=0.5)
        self.assertTrue(isinstance(result, Vector))
        self.assertEqual(result.as_dict(), {0: 0.5, 1: 0.25, 2: 0.25, 3: 1.0})
        del vectors

    def test_metric_stability_pareto_compare(self):
        values = []
        for i in range(20):
            values.append(dict(enumerate(np.random.random(100))))
        result1 = metric_stability_pareto(values)

        graphs = [Graph.from_dict(dict(((0, i), w) for i, w in v.items())) for v in values]
        result2 = metric_stability_pareto(graphs)
        self.assertTrue(isinstance(result2, Graph))
        self.assertEqual(result2.num_edges, 100)
        for i in range(100):
            self.assertEqual(result2[0, i], result1[i])
        del result2
        del graphs

        vectors = [Vector.from_dict(v) for v in values]
        result2 = metric_stability_pareto(vectors)
        self.assertTrue(isinstance(result2, Vector))
        self.assertEqual(result2.num_entries, 100)
        for i in range(100):
            self.assertEqual(result2[i], result1[i])
        del result2
        del vectors

    def test_metric_avg(self):
        values = [
            {(0, 0): 1.0, (0, 1): 0.0, (1, 1): 2.0, (2, 2): 2.0},
            {(0, 0): 1.0, (0, 1): 1.0, (1, 1): 1.0, (2, 2): 2.0},
            {(0, 0): 1.0, (0, 1): 2.0, (1, 1): 0.0, (2, 2): 2.0},
        ]

        result = metric_avg(values)
        self.assertEqual(result, {(0, 0): 1.0, (0, 1): 1.0,
                                  (1, 1): 1.0, (2, 2): 2.0})

        # Fast-path for list of Graphs
        graphs = [Graph.from_dict(v) for v in values]
        result = metric_avg(graphs)
        self.assertTrue(isinstance(result, Graph))
        self.assertEqual(result.as_dict(), {(0, 0): 1.0, (0, 1): 1.0,
                                            (1, 1): 1.0, (2, 2): 2.0})
        del graphs

        values = [
            {0: 1.0, 1: 0.0, 2: 2.0, 3: 2.0},
            {0: 1.0, 1: 1.0, 2: 1.0, 3: 2.0},
            {0: 1.0, 1: 2.0, 2: 0.0, 3: 2.0},
        ]

        # Fast-path for list of Vectors
        vectors = [Vector.from_dict(v) for v in values]
        result = metric_avg(vectors)
        self.assertTrue(isinstance(result, Vector))
        self.assertEqual(result.as_dict(), {0: 1.0, 1: 1.0, 2: 1.0, 3: 2.0})
        del vectors

    def test_metric_std(self):
        values = [
            {(0, 0): 1.0, (0, 1): 0.0, (1, 1): 2.0, (2, 2): 2.0},
            {(0, 0): 1.0, (0, 1): 1.0, (1, 1): 1.0, (2, 2): 2.0},
            {(0, 0): 1.0, (0, 1): 2.0, (1, 1): 0.0, (2, 2): 2.0},
        ]

        result = metric_std(values)
        self.assertEqual(result, {(0, 0): 0.0, (0, 1): 1.0,
                                  (1, 1): 1.0, (2, 2): 0.0})

        # Fast-path for list of Graphs
        graphs = [Graph.from_dict(v) for v in values]
        result = metric_std(graphs)
        self.assertTrue(isinstance(result, Graph))
        self.assertEqual(result.as_dict(), {(0, 0): 0.0, (0, 1): 1.0,
                                            (1, 1): 1.0, (2, 2): 0.0})
        del graphs

        values = [
            {0: 1.0, 1: 0.0, 2: 2.0, 3: 2.0},
            {0: 1.0, 1: 1.0, 2: 1.0, 3: 2.0},
            {0: 1.0, 1: 2.0, 2: 0.0, 3: 2.0},
        ]

        # Fast-path for list of Vectors
        vectors = [Vector.from_dict(v) for v in values]
        result = metric_std(vectors)
        self.assertTrue(isinstance(result, Vector))
        self.assertEqual(result.as_dict(), {0: 0.0, 1: 1.0, 2: 1.0, 3: 0.0})
        del vectors

    def test_metric_pareto(self):
        values = [
            {(0, 0): 1.0, (0, 1): 1.0, (1, 1): 1.0, (2, 2): 2.0},
            {(0, 0): 0.0, (0, 1): 1.0, (1, 1): 1.0, (2, 2): 0.0},
        ]

        result = metric_pareto(values, maximize=[True, False])
        self.assertEqual(result, {(0, 0): 2.0, (0, 1): 3.0,
                                  (1, 1): 3.0, (2, 2): 1.0})

        result = metric_pareto(list(reversed(values)), maximize=[False, True])
        self.assertEqual(result, {(0, 0): 2.0, (0, 1): 3.0,
                                  (1, 1): 3.0, (2, 2): 1.0})

        # Fast-path for list of Graphs
        graphs = [Graph.from_dict(v) for v in values]
        result = metric_pareto(graphs, maximize=[True, False])
        self.assertTrue(isinstance(result, Graph))
        self.assertEqual(result.as_dict(), {(0, 0): 2.0, (0, 1): 3.0,
                                            (1, 1): 3.0, (2, 2): 1.0})

        result = metric_pareto(list(reversed(graphs)), maximize=[False, True])
        self.assertTrue(isinstance(result, Graph))
        self.assertEqual(result.as_dict(), {(0, 0): 2.0, (0, 1): 3.0,
                                            (1, 1): 3.0, (2, 2): 1.0})
        del graphs

        values = [
            {0: 1.0, 1: 1.0, 2: 1.0, 3: 2.0},
            {0: 0.0, 1: 1.0, 2: 1.0, 3: 0.0},
        ]

        # Fast-path for list of Vectors
        vectors = [Vector.from_dict(v) for v in values]
        result = metric_pareto(vectors, maximize=[True, False])
        self.assertTrue(isinstance(result, Vector))
        self.assertEqual(result.as_dict(), {0: 2.0, 1: 3.0, 2: 3.0, 3: 1.0})

        result = metric_pareto(list(reversed(vectors)), maximize=[False, True])
        self.assertTrue(isinstance(result, Vector))
        self.assertEqual(result.as_dict(), {0: 2.0, 1: 3.0, 2: 3.0, 3: 1.0})
        del vectors

class MongoDBTests(unittest.TestCase):
    def MongoDB(self, *args, **kwargs):
        future = mockupdb.go(MongoDB, *args, **kwargs)

        try:
            request = self.s.receives("isMaster", timeout=3)
            request.replies({'ok': 1, 'maxWireVersion': 5})
        except AssertionError:
            raise unittest.SkipTest("MongoDB not supported")

        request = self.s.receives("ping")
        request.replies({'ok': 1})

        return future()

    def setUp(self):
        self.s = mockupdb.MockupDB()
        self.s.run()

        self.db = self.MongoDB(self.s.uri, "database", "col_articles",
                               "_id", "time", "col_entities", "doc", "sen", "ent",
                               use_pool=False, max_distance=5, filter_key="fkey",
                               filter_value="fvalue")

    def tearDown(self):
        self.s.stop()

    def load_from_occurrences(self, occurrences):
        future = mockupdb.go(Graph.load_from_mongodb, self.db, 1337)

        request = self.s.receives()
        self.assertEqual(request["find"], "col_entities")
        self.assertEqual(request["filter"], {'doc': 1337})
        self.assertEqual(request["sort"], {'sen': 1})
        request.replies({'cursor': {'id': 0, 'firstBatch': occurrences}})

        return future()

    def test_invalid(self):
        with self.assertRaises(MemoryError):
            MongoDB("http://localhost", "database", "col_articles",
                    "_id", "time", "col_entities", "doc", "sen", "ent")

    def test_selfloop(self):
        occurrences = []
        for i in range(10):
            occurrences.append({'sen': i, 'ent': 1})
            occurrences.append({'sen': i, 'ent': 1})
        g = self.load_from_occurrences(occurrences)
        self.assertEqual(g.num_edges, 0)
        self.assertEqual(g.nodes().tolist(), [1])
        self.assertEqual(g.nodes().as_dict(), {1: 20})

    def test_weight_sum(self):
        for i in range(10):
            occurrences = [{'sen': 1,     'ent': 1    },
                           {'sen': 1 + i, 'ent': 2    },
                           {              'ent': 1    }, # no sen
                           {'sen': 1                  }] # no ent
            g = self.load_from_occurrences(occurrences)
            if i <= 5:
                self.assertEqual(g.num_edges, 1)
                self.assertTrue(abs(g[1, 2]/math.exp(-i) - 1.0) < 1e-7)
            else:
                self.assertEqual(g.num_edges, 0)
            self.assertEqual(g.nodes().tolist(), [1, 2])
            self.assertEqual(g.nodes().as_dict(), {1: 1.0, 2: 1.0})

            g = g.duplicate()
            if i <= 5:
                self.assertEqual(g.num_edges, 1)
            else:
                self.assertEqual(g.num_edges, 0)
            self.assertEqual(g.nodes().as_dict(), {1: 1.0, 2: 1.0})

            d = g.all_distances_graph()
            c = g.connected_components()
            if i <= 5:
                self.assertEqual(d.as_dict(), {(2, 1): 1.0, (1, 2): 1.0})
                self.assertEqual(c.as_dict(), {1: 0.0, 2: 0.0})
            else:
                self.assertEqual(d.as_dict(), {})
                self.assertEqual(c.as_dict(), {1: 0.0, 2: 1.0})

            g.del_small()
            if i <= 5:
                self.assertEqual(g.num_edges, 1)
                self.assertEqual(g.nodes().as_dict(), {1: 1.0, 2: 1.0})
            else:
                self.assertEqual(g.num_edges, 0)
                self.assertEqual(g.nodes().as_dict(), {})

            d = g.all_distances_graph()
            c = g.connected_components()
            if i <= 5:
                self.assertEqual(d.as_dict(), {(2, 1): 1.0, (1, 2): 1.0})
                self.assertEqual(c.as_dict(), {1: 0.0, 2: 0.0})
            else:
                self.assertEqual(d.as_dict(), {})
                self.assertEqual(c.as_dict(), {})

    def test_load(self):
        future = mockupdb.go(TVG.load, self.db, primary_key="dummy")

        request = self.s.receives()
        self.assertEqual(request["find"], "col_articles")
        self.assertEqual(request["filter"], {'fkey': 'fvalue'})
        self.assertEqual(request["sort"], collections.OrderedDict([('time', 1), ('_id', 1)]))
        documents = [{'_id': 10, 'time': datetime.datetime.utcfromtimestamp(1546300800)},
                     {'_id': 11, 'time': datetime.datetime.utcfromtimestamp(1546387200)},
                     {'_id': 12, 'time': datetime.datetime.utcfromtimestamp(1546473600)},
                     {           'time': datetime.datetime.utcfromtimestamp(1546560000)}, # no id
                     {'_id': 14                                                        }] # no time
        request.replies({'cursor': {'id': 0, 'firstBatch': documents}})

        for i in range(3):
            request = self.s.receives()
            self.assertEqual(request["find"], "col_entities")
            self.assertEqual(request["filter"], {'doc': 10 + i})
            self.assertEqual(request["sort"], {'sen': 1})
            occurrences = [{'sen': 1, 'ent': 1}, {'sen': 1, 'ent': 2 + i}]
            request.replies({'cursor': {'id': 0, 'firstBatch': occurrences}})

        tvg = future()
        for i, g in enumerate(tvg):
            self.assertEqual(g.revision, 0)
            self.assertEqual(g.ts, 1546300800000 + i * 86400000)
            self.assertEqual(g.id, 10 + i)
            self.assertEqual(g[1, 2 + i], 1.0)
            self.assertEqual(g.nodes().tolist(), [1, 2 + i])
            self.assertEqual(g.nodes().as_dict(), {1: 1.0, 2 + i: 1.0})
        del tvg

    def test_sync(self):
        tvg = TVG()
        tvg.enable_mongodb_sync(self.db, batch_size=2, cache_size=0x8000) # 32 kB cache
        tvg.verbosity = True
        self.assertEqual(tvg.verbosity, True)

        future = mockupdb.go(tvg.lookup_ge, 0)

        request = self.s.receives()
        self.assertEqual(request["find"], "col_articles")
        self.assertEqual(request["filter"], {'time': {'$gte': datetime.datetime.utcfromtimestamp(0)},
                                             'fkey': 'fvalue'})
        self.assertEqual(request["sort"], collections.OrderedDict([('time', 1), ('_id', 1)]))
        self.assertEqual(request["limit"], 2)
        documents = [{'_id': 10, 'time': datetime.datetime.utcfromtimestamp(1546300800)},
                     {'_id': 11, 'time': datetime.datetime.utcfromtimestamp(1546387200)}]
        request.replies({'cursor': {'id': 0, 'firstBatch': documents}})

        for i in range(2):
            request = self.s.receives()
            self.assertEqual(request["find"], "col_entities")
            self.assertEqual(request["filter"], {'doc': 10 + i})
            self.assertEqual(request["sort"], {'sen': 1})
            occurrences = [{'sen': 1, 'ent': 1}, {'sen': 1, 'ent': 2 + i}]
            request.replies({'cursor': {'id': 0, 'firstBatch': occurrences}})

        g = future()
        self.assertEqual(g.revision, 0)
        self.assertEqual(g.flags, TVG_FLAGS_READONLY)
        self.assertEqual(g.ts, 1546300800000)
        self.assertEqual(g.id, 10)
        self.assertEqual(g[1, 2], 1.0)
        self.assertEqual(g.nodes().tolist(), [1, 2])
        self.assertEqual(g.nodes().as_dict(), {1: 1.0, 2: 1.0})

        g = g.next
        self.assertEqual(g.revision, 0)
        self.assertEqual(g.flags, TVG_FLAGS_LOAD_NEXT | TVG_FLAGS_READONLY)
        self.assertEqual(g.ts, 1546387200000)
        self.assertEqual(g.id, 11)
        self.assertEqual(g[1, 3], 1.0)
        self.assertEqual(g.nodes().tolist(), [1, 3])
        self.assertEqual(g.nodes().as_dict(), {1: 1.0, 3: 1.0})

        future = mockupdb.go(getattr, g, 'next')

        request = self.s.receives()
        self.assertEqual(request["find"], "col_articles")
        self.assertEqual(request["filter"], {"$or": [{"time": {"$gt": datetime.datetime.utcfromtimestamp(1546387200)}},
                                                     {"time": datetime.datetime.utcfromtimestamp(1546387200), "_id": {"$gt": 11}}],
                                             'fkey': 'fvalue'})
        self.assertEqual(request["sort"], collections.OrderedDict([('time', 1), ('_id', 1)]))
        self.assertEqual(request["limit"], 2)
        documents = [{'_id': 12, 'time': datetime.datetime.utcfromtimestamp(1546473600)},
                     {'_id': 13, 'time': datetime.datetime.utcfromtimestamp(1546560000)}]
        request.replies({'cursor': {'id': 0, 'firstBatch': documents}})

        for i in range(2):
            request = self.s.receives()
            self.assertEqual(request["find"], "col_entities")
            self.assertEqual(request["filter"], {'doc': 12 + i})
            self.assertEqual(request["sort"], {'sen': 1})
            occurrences = [{'sen': 1, 'ent': 1}, {'sen': 1, 'ent': 4 + i}]
            request.replies({'cursor': {'id': 0, 'firstBatch': occurrences}})

        g = future()
        self.assertEqual(g.revision, 0)
        self.assertEqual(g.flags, TVG_FLAGS_READONLY)
        self.assertEqual(g.ts, 1546473600000)
        self.assertEqual(g.id, 12)
        self.assertEqual(g[1, 4], 1.0)
        self.assertEqual(g.nodes().tolist(), [1, 4])
        self.assertEqual(g.nodes().as_dict(), {1: 1.0, 4: 1.0})

        g = g.next
        self.assertEqual(g.revision, 0)
        self.assertEqual(g.flags, TVG_FLAGS_LOAD_NEXT | TVG_FLAGS_READONLY)
        self.assertEqual(g.ts, 1546560000000)
        self.assertEqual(g.id, 13)
        self.assertEqual(g[1, 5], 1.0)
        self.assertEqual(g.nodes().tolist(), [1, 5])
        self.assertEqual(g.nodes().as_dict(), {1: 1.0, 5: 1.0})

        future = mockupdb.go(tvg.lookup_le, 1546732800000)

        request = self.s.receives()
        self.assertEqual(request["find"], "col_articles")
        self.assertEqual(request["filter"], {'time': {'$lte': datetime.datetime.utcfromtimestamp(1546732800)},
                                             'fkey': 'fvalue'})
        self.assertEqual(request["sort"], collections.OrderedDict([('time', -1), ('_id', -1)]))
        self.assertEqual(request["limit"], 2)
        documents = [{'_id': 15, 'time': datetime.datetime.utcfromtimestamp(1546732800)},
                     {'_id': 14, 'time': datetime.datetime.utcfromtimestamp(1546646400)}]
        request.replies({'cursor': {'id': 0, 'firstBatch': documents}})

        for i in range(2):
            request = self.s.receives()
            self.assertEqual(request["find"], "col_entities")
            self.assertEqual(request["filter"], {'doc': 15 - i})
            self.assertEqual(request["sort"], {'sen': 1})
            occurrences = [{'sen': 1, 'ent': 1}, {'sen': 1, 'ent': 7 - i}]
            request.replies({'cursor': {'id': 0, 'firstBatch': occurrences}})

        g = future()
        self.assertEqual(g.revision, 0)
        self.assertEqual(g.flags, TVG_FLAGS_LOAD_NEXT | TVG_FLAGS_READONLY)
        self.assertEqual(g.ts, 1546732800000)
        self.assertEqual(g.id, 15)
        self.assertEqual(g[1, 7], 1.0)
        self.assertEqual(g.nodes().tolist(), [1, 7])
        self.assertEqual(g.nodes().as_dict(), {1: 1.0, 7: 1.0})

        g = g.prev
        self.assertEqual(g.revision, 0)
        self.assertEqual(g.flags, TVG_FLAGS_LOAD_PREV | TVG_FLAGS_READONLY)
        self.assertEqual(g.ts, 1546646400000)
        self.assertEqual(g.id, 14)
        self.assertEqual(g[1, 6], 1.0)
        self.assertEqual(g.nodes().tolist(), [1, 6])
        self.assertEqual(g.nodes().as_dict(), {1: 1.0, 6: 1.0})

        future = mockupdb.go(getattr, g, 'prev')

        request = self.s.receives()
        self.assertEqual(request["find"], "col_articles")
        self.assertEqual(request["filter"], {"$or": [{"time": {"$lt": datetime.datetime.utcfromtimestamp(1546646400)}},
                                                     {"time": datetime.datetime.utcfromtimestamp(1546646400), "_id": {"$lt": 14}}],
                                             'fkey': 'fvalue'})
        self.assertEqual(request["sort"], collections.OrderedDict([('time', -1), ('_id', -1)]))
        self.assertEqual(request["limit"], 2)
        documents = [{'_id': 13, 'time': datetime.datetime.utcfromtimestamp(1546560000)},
                     {'_id': 12, 'time': datetime.datetime.utcfromtimestamp(1546473600)}]
        request.replies({'cursor': {'id': 0, 'firstBatch': documents}})

        g = future()
        self.assertEqual(g.revision, 0)
        self.assertEqual(g.flags, TVG_FLAGS_READONLY)
        self.assertEqual(g.ts, 1546560000000)
        self.assertEqual(g.id, 13)
        self.assertEqual(g[1, 5], 1.0)
        self.assertEqual(g.nodes().tolist(), [1, 5])
        self.assertEqual(g.nodes().as_dict(), {1: 1.0, 5: 1.0})

        g = tvg.lookup_ge(1546732800000)
        self.assertEqual(g.revision, 0)
        self.assertEqual(g.flags, TVG_FLAGS_LOAD_NEXT | TVG_FLAGS_READONLY)
        self.assertEqual(g.ts, 1546732800000)
        self.assertEqual(g.id, 15)
        self.assertEqual(g[1, 7], 1.0)
        self.assertEqual(g.nodes().tolist(), [1, 7])
        self.assertEqual(g.nodes().as_dict(), {1: 1.0, 7: 1.0})

        future = mockupdb.go(getattr, g, 'next')

        request = self.s.receives()
        self.assertEqual(request["find"], "col_articles")
        self.assertEqual(request["filter"], {"$or": [{"time": {"$gt": datetime.datetime.utcfromtimestamp(1546732800)}},
                                                     {"time": datetime.datetime.utcfromtimestamp(1546732800), "_id": {"$gt": 15}}],
                                             'fkey': 'fvalue'})
        self.assertEqual(request["sort"], collections.OrderedDict([('time', 1), ('_id', 1)]))
        self.assertEqual(request["limit"], 2)
        documents = []
        request.replies({'cursor': {'id': 0, 'firstBatch': documents}})

        g = future()
        self.assertEqual(g, None)

        for i, g in enumerate(tvg):
            self.assertEqual(g.revision, 0)
            self.assertEqual(g.flags, TVG_FLAGS_READONLY)
            self.assertEqual(g.ts, 1546300800000 + i * 86400000)
            self.assertEqual(g.id, 10 + i)
            self.assertEqual(g[1, 2 + i], 1.0)
            self.assertEqual(g.nodes().tolist(), [1, 2 + i])
            self.assertEqual(g.nodes().as_dict(), {1: 1.0, 2 + i: 1.0})

        tvg.disable_mongodb_sync()
        del tvg

    def test_streaming(self):
        tvg = TVG(streaming=True)
        tvg.enable_mongodb_sync(self.db, batch_size=2, cache_size=0x8000) # 32 kB cache

        future = mockupdb.go(tvg.lookup_le)

        request = self.s.receives()
        self.assertEqual(request["find"], "col_articles")
        self.assertEqual(request["filter"], {'fkey': 'fvalue'})
        self.assertEqual(request["sort"], collections.OrderedDict([('time', -1), ('_id', -1)]))
        self.assertEqual(request["limit"], 2)
        documents = [{'_id': 11, 'time': datetime.datetime.utcfromtimestamp(1546387200)},
                     {'_id': 10, 'time': datetime.datetime.utcfromtimestamp(1546300800)}]
        request.replies({'cursor': {'id': 0, 'firstBatch': documents}})

        for i in range(2):
            request = self.s.receives()
            self.assertEqual(request["find"], "col_entities")
            self.assertEqual(request["filter"], {'doc': 11 - i})
            self.assertEqual(request["sort"], {'sen': 1})
            occurrences = [{'sen': 1, 'ent': 1}, {'sen': 1, 'ent': 3 - i}]
            request.replies({'cursor': {'id': 0, 'firstBatch': occurrences}})

        g = future()
        self.assertEqual(g.revision, 0)
        self.assertEqual(g.flags, TVG_FLAGS_LOAD_NEXT | TVG_FLAGS_READONLY)
        self.assertEqual(g.ts, 1546387200000)
        self.assertEqual(g[1, 3], 1.0)
        self.assertEqual(g.nodes().tolist(), [1, 3])
        self.assertEqual(g.nodes().as_dict(), {1: 1.0, 3: 1.0})

        g = g.prev
        self.assertEqual(g.revision, 0)
        self.assertEqual(g.flags, TVG_FLAGS_LOAD_PREV | TVG_FLAGS_READONLY)
        self.assertEqual(g.ts, 1546300800000)
        self.assertEqual(g[1, 2], 1.0)
        self.assertEqual(g.nodes().tolist(), [1, 2])
        self.assertEqual(g.nodes().as_dict(), {1: 1.0, 2: 1.0})

        future = mockupdb.go(tvg.lookup_le)

        request = self.s.receives()
        self.assertEqual(request["find"], "col_articles")
        self.assertEqual(request["filter"], {'fkey': 'fvalue'})
        self.assertEqual(request["sort"], collections.OrderedDict([('time', -1), ('_id', -1)]))
        self.assertEqual(request["limit"], 2)
        documents = [{'_id': 12, 'time': datetime.datetime.utcfromtimestamp(1546473600)},
                     {'_id': 11, 'time': datetime.datetime.utcfromtimestamp(1546387200)}]
        request.replies({'cursor': {'id': 0, 'firstBatch': documents}})

        request = self.s.receives()
        self.assertEqual(request["find"], "col_entities")
        self.assertEqual(request["filter"], {'doc': 12})
        self.assertEqual(request["sort"], {'sen': 1})
        occurrences = [{'sen': 1, 'ent': 1}, {'sen': 1, 'ent': 4}]
        request.replies({'cursor': {'id': 0, 'firstBatch': occurrences}})

        g = future()
        self.assertEqual(g.revision, 0)
        self.assertEqual(g.flags, TVG_FLAGS_LOAD_NEXT | TVG_FLAGS_READONLY)
        self.assertEqual(g.ts, 1546473600000)
        self.assertEqual(g[1, 4], 1.0)
        self.assertEqual(g.nodes().tolist(), [1, 4])
        self.assertEqual(g.nodes().as_dict(), {1: 1.0, 4: 1.0})

        g2 = g.prev
        self.assertEqual(g2.revision, 0)
        self.assertEqual(g2.flags, TVG_FLAGS_READONLY)
        self.assertEqual(g2.ts, 1546387200000)
        self.assertEqual(g2[1, 3], 1.0)
        self.assertEqual(g2.nodes().tolist(), [1, 3])
        self.assertEqual(g2.nodes().as_dict(), {1: 1.0, 3: 1.0})

        future = mockupdb.go(getattr, g, 'next')

        request = self.s.receives()
        self.assertEqual(request["find"], "col_articles")
        self.assertEqual(request["filter"], {"$or": [{"time": {"$gt": datetime.datetime.utcfromtimestamp(1546473600)}},
                                                     {"time": datetime.datetime.utcfromtimestamp(1546473600), "_id": {"$gt": 12}}],
                                             'fkey': 'fvalue'})
        self.assertEqual(request["sort"], collections.OrderedDict([('time', 1), ('_id', 1)]))
        self.assertEqual(request["limit"], 2)
        documents = []
        request.replies({'cursor': {'id': 0, 'firstBatch': documents}})

        g2 = future()
        self.assertEqual(g2, None)

        self.assertEqual(g.revision, 0)
        self.assertEqual(g.flags, TVG_FLAGS_LOAD_NEXT | TVG_FLAGS_READONLY)
        self.assertEqual(g.ts, 1546473600000)
        self.assertEqual(g[1, 4], 1.0)
        self.assertEqual(g.nodes().tolist(), [1, 4])
        self.assertEqual(g.nodes().as_dict(), {1: 1.0, 4: 1.0})

        tvg.disable_mongodb_sync()
        del tvg

    def test_objectid(self):
        self.db = self.MongoDB(self.s.uri, "database", "col_articles",
                               "_id", "time", "col_entities", "doc", "sen", "ent",
                               use_pool=False, max_distance=5)

        future = mockupdb.go(TVG.load, self.db, primary_key="dummy")

        request = self.s.receives()
        self.assertEqual(request["find"], "col_articles")
        self.assertEqual(request["filter"], {})
        self.assertEqual(request["sort"], collections.OrderedDict([('time', 1), ('_id', 1)]))
        documents = [{'_id': bson.ObjectId('123456781234567812345678'),
                      'time': datetime.datetime.utcfromtimestamp(1546300800)},
                     {'_id': bson.ObjectId('123456781234567812345679'),
                      'time': datetime.datetime.utcfromtimestamp(1546387200)},
                     {'_id': bson.ObjectId('12345678123456781234567a'),
                      'time': datetime.datetime.utcfromtimestamp(1546473600)},
                     {'time': datetime.datetime.utcfromtimestamp(1546560000)}, # no id
                     {'_id': bson.ObjectId('12345678123456781234567c')}]       # no time
        request.replies({'cursor': {'id': 0, 'firstBatch': documents}})

        for i in range(3):
            request = self.s.receives()
            self.assertEqual(request["find"], "col_entities")
            self.assertEqual(request["filter"], {'doc': [bson.ObjectId('123456781234567812345678'),
                                                         bson.ObjectId('123456781234567812345679'),
                                                         bson.ObjectId('12345678123456781234567a')][i]})
            self.assertEqual(request["sort"], {'sen': 1})
            occurrences = [{'sen': 1, 'ent': 1}, {'sen': 1, 'ent': 2 + i}]
            request.replies({'cursor': {'id': 0, 'firstBatch': occurrences}})

        tvg = future()
        for i, g in enumerate(tvg):
            self.assertEqual(g.revision, 0)
            self.assertEqual(g.ts, 1546300800000 + i * 86400000)
            self.assertEqual(g.id, ['123456781234567812345678',
                                    '123456781234567812345679',
                                    '12345678123456781234567a'][i])
            self.assertEqual(g[1, 2 + i], 1.0)
            self.assertEqual(g.nodes().tolist(), [1, 2 + i])
            self.assertEqual(g.nodes().as_dict(), {1: 1.0, 2 + i: 1.0})
        del tvg

        future = mockupdb.go(Graph.load_from_mongodb, self.db, '112233445566778899aabbcc')

        request = self.s.receives()
        self.assertEqual(request["find"], "col_entities")
        self.assertEqual(request["filter"], {'doc': bson.ObjectId('112233445566778899aabbcc')})
        self.assertEqual(request["sort"], {'sen': 1})
        request.replies({'cursor': {'id': 0, 'firstBatch': []}})

        g = future()
        self.assertTrue(isinstance(g, Graph))
        del g

        future = mockupdb.go(Graph.load_from_mongodb, self.db, b'\x11\x22\x33\x44\x55\x66\x77\x88\x99\xaa\xbb\xcc')

        request = self.s.receives()
        self.assertEqual(request["find"], "col_entities")
        self.assertEqual(request["filter"], {'doc': bson.ObjectId('112233445566778899aabbcc')})
        self.assertEqual(request["sort"], {'sen': 1})
        request.replies({'cursor': {'id': 0, 'firstBatch': []}})

        g = future()
        self.assertTrue(isinstance(g, Graph))
        del g

    def test_max_distance(self):
        occurrences = [{'sen': 0, 'ent': 1 },
                       {'sen': 0x7fffffffffffffff, 'ent': 2 }]

        g = self.load_from_occurrences(occurrences)
        self.assertFalse(g.has_edge((1, 2)))
        self.assertEqual(g.nodes().tolist(), [1, 2])
        self.assertEqual(g.nodes().as_dict(), {1: 1.0, 2: 1.0})
        del g

        self.db = self.MongoDB(self.s.uri, "database", "col_articles",
                               "_id", "time", "col_entities", "doc", "sen", "ent",
                               use_pool=False, max_distance=None)

        g = self.load_from_occurrences(occurrences)
        self.assertTrue(g.has_edge((1, 2)))
        self.assertEqual(g[1, 2], 0.0)
        self.assertEqual(g.nodes().tolist(), [1, 2])
        self.assertEqual(g.nodes().as_dict(), {1: 1.0, 2: 1.0})
        del g

    def test_sum_weights(self):
        occurrences1 = [{'sen': 1, 'ent': 1},
                        {'sen': 2, 'ent': 2},
                        {'sen': 4, 'ent': 1}]

        occurrences2 = [{'sen': 1, 'ent': 1},
                        {'sen': 3, 'ent': 2},
                        {'sen': 4, 'ent': 1}]

        g = self.load_from_occurrences(occurrences1)
        self.assertTrue(abs(g[1, 2]/(math.exp(-1.0) + np.exp(-2.0)) - 1.0) < 1e-7)
        self.assertEqual(g.nodes().tolist(), [1, 2])
        self.assertEqual(g.nodes().as_dict(), {1: 2.0, 2: 1.0})
        del g

        g = self.load_from_occurrences(occurrences2)
        self.assertTrue(abs(g[1, 2]/(math.exp(-1.0) + np.exp(-2.0)) - 1.0) < 1e-7)
        self.assertEqual(g.nodes().tolist(), [1, 2])
        self.assertEqual(g.nodes().as_dict(), {1: 2.0, 2: 1.0})
        del g

        self.db = self.MongoDB(self.s.uri, "database", "col_articles",
                               "_id", "time", "col_entities", "doc", "sen", "ent",
                               use_pool=False, sum_weights=False)

        g = self.load_from_occurrences(occurrences1)
        self.assertTrue(abs(g[1, 2]/math.exp(-1.0) - 1.0) < 1e-7)
        self.assertEqual(g.nodes().tolist(), [1, 2])
        self.assertEqual(g.nodes().as_dict(), {1: 2.0, 2: 1.0})
        del g

        g = self.load_from_occurrences(occurrences2)
        self.assertTrue(abs(g[1, 2]/math.exp(-1.0) - 1.0) < 1e-7)
        self.assertEqual(g.nodes().tolist(), [1, 2])
        self.assertEqual(g.nodes().as_dict(), {1: 2.0, 2: 1.0})
        del g

        self.db = self.MongoDB(self.s.uri, "database", "col_articles",
                               "_id", "time", "col_entities", "doc", "sen", "ent",
                               use_pool=False, norm_weights=True)

        g = self.load_from_occurrences(occurrences1)
        self.assertEqual(g[1, 2], 1.0)
        self.assertEqual(g.nodes().tolist(), [1, 2])
        self.assertEqual(g.nodes().as_dict(), {1: 2.0, 2: 1.0})
        del g

        g = self.load_from_occurrences(occurrences2)
        self.assertEqual(g[1, 2], 1.0)
        self.assertEqual(g.nodes().tolist(), [1, 2])
        self.assertEqual(g.nodes().as_dict(), {1: 2.0, 2: 1.0})
        del g

    def test_readonly(self):
        future = mockupdb.go(TVG.load, self.db, primary_key="dummy")

        request = self.s.receives()
        self.assertEqual(request["find"], "col_articles")
        self.assertEqual(request["filter"], {'fkey': 'fvalue'})
        self.assertEqual(request["sort"], collections.OrderedDict([('time', 1), ('_id', 1)]))
        documents = [{'_id': 10, 'time': datetime.datetime.utcfromtimestamp(1546387200)}]
        request.replies({'cursor': {'id': 0, 'firstBatch': documents}})

        request = self.s.receives()
        self.assertEqual(request["find"], "col_entities")
        self.assertEqual(request["filter"], {'doc': 10})
        self.assertEqual(request["sort"], {'sen': 1})
        occurrences = [{'sen': 1, 'ent': 1}, {'sen': 1, 'ent': 2}]
        request.replies({'cursor': {'id': 0, 'firstBatch': occurrences}})

        tvg = future()
        g = tvg.lookup_ge()
        self.assertEqual(g.ts, 1546387200000)
        self.assertEqual(g.readonly, True)
        self.assertEqual(g.nodes().readonly, True)

        with self.assertRaises(RuntimeError):
            g.clear()
        with self.assertRaises(MemoryError):
            g[0, 0] = 1.0
        with self.assertRaises(MemoryError):
            g.add_edge((0, 0), 1.0)
        with self.assertRaises(RuntimeError):
            del g[0, 0]
        with self.assertRaises(RuntimeError):
            g.mul_const(2.0)
        with self.assertRaises(RuntimeError):
            g.del_small()

        g.unlink()
        self.assertEqual(g.readonly, False)
        self.assertEqual(g.nodes().readonly, True)

        g.clear()
        g[0, 0] = 1.0
        g.add_edge((0, 0), 1.0)
        del g[0, 0]
        g.mul_const(2.0)
        g.del_small()

        del tvg
        del g

        tvg = TVG(streaming=True)
        tvg.enable_mongodb_sync(self.db, batch_size=1, cache_size=0x8000) # 32 kB cache

        future = mockupdb.go(tvg.lookup_le)

        request = self.s.receives()
        self.assertEqual(request["find"], "col_articles")
        self.assertEqual(request["filter"], {'fkey': 'fvalue'})
        self.assertEqual(request["sort"], collections.OrderedDict([('time', -1), ('_id', -1)]))
        self.assertEqual(request["limit"], 1)
        documents = [{'_id': 10, 'time': datetime.datetime.utcfromtimestamp(1546387200)}]
        request.replies({'cursor': {'id': 0, 'firstBatch': documents}})

        request = self.s.receives()
        self.assertEqual(request["find"], "col_entities")
        self.assertEqual(request["filter"], {'doc': 10})
        self.assertEqual(request["sort"], {'sen': 1})
        occurrences = [{'sen': 1, 'ent': 1}, {'sen': 1, 'ent': 2}]
        request.replies({'cursor': {'id': 0, 'firstBatch': occurrences}})

        g = future()
        self.assertEqual(g.ts, 1546387200000)
        self.assertEqual(g.readonly, True)
        self.assertEqual(g.nodes().readonly, True)
        self.assertEqual(g.nodes().as_dict(), {1: 1.0, 2: 1.0})

        with self.assertRaises(RuntimeError):
            g.clear()
        with self.assertRaises(MemoryError):
            g[0, 0] = 1.0
        with self.assertRaises(MemoryError):
            g.add_edge((0, 0), 1.0)
        with self.assertRaises(RuntimeError):
            del g[0, 0]
        with self.assertRaises(RuntimeError):
            g.mul_const(2.0)
        with self.assertRaises(RuntimeError):
            g.del_small()

        g.unlink()
        self.assertEqual(g.readonly, False)
        self.assertEqual(g.nodes().readonly, True)
        self.assertEqual(g.nodes().as_dict(), {1: 1.0, 2: 1.0})

        g.clear()
        g[0, 0] = 1.0
        g.add_edge((0, 0), 1.0)
        del g[0, 0]
        g.mul_const(2.0)
        g.del_small()

        g.add_edge((2, 3), 1.0)
        self.assertEqual(g.readonly, False)
        self.assertEqual(g.nodes().readonly, True)
        self.assertEqual(g.nodes().as_dict(), {2: 1.0, 3: 1.0})

        del tvg
        del g

    def test_sum_nodes(self):
        tvg = TVG(positive=True)

        occurrences = [{'sen': 0, 'ent': 0},
                       {'sen': 0, 'ent': 0},
                       {'sen': 0, 'ent': 1}]
        g = self.load_from_occurrences(occurrences)
        tvg.link_graph(g, 100)

        occurrences = [{'sen': 0, 'ent': 0},
                       {'sen': 0, 'ent': 2},
                       {'sen': 0, 'ent': 2}]
        g = self.load_from_occurrences(occurrences)
        tvg.link_graph(g, 200)

        occurrences = [{'sen': 0, 'ent': 0},
                       {'sen': 0, 'ent': 3},
                       {'sen': 0, 'ent': 3},
                       {'sen': 0, 'ent': 3}]
        g = self.load_from_occurrences(occurrences)
        tvg.link_graph(g, 300)

        with self.assertRaises(MemoryError):
            tvg.sum_nodes(1, 0)

        v = tvg.sum_nodes(51, 150)
        self.assertEqual(v.readonly, True)
        self.assertEqual(v[0], 2.0)
        self.assertEqual(v[1], 1.0)
        self.assertEqual(v[2], 0.0)
        self.assertEqual(v[3], 0.0)

        v = tvg.sum_nodes(151, 250)
        self.assertEqual(v[0], 1.0)
        self.assertEqual(v[1], 0.0)
        self.assertEqual(v[2], 2.0)
        self.assertEqual(v[3], 0.0)

        v = tvg.sum_nodes(251, 350)
        self.assertEqual(v[0], 1.0)
        self.assertEqual(v[1], 0.0)
        self.assertEqual(v[2], 0.0)
        self.assertEqual(v[3], 3.0)

        v = tvg.sum_nodes(51, 350)
        self.assertEqual(v[0], 4.0)
        self.assertEqual(v[1], 1.0)
        self.assertEqual(v[2], 2.0)
        self.assertEqual(v[3], 3.0)

        del tvg

    def test_count_nodes(self):
        tvg = TVG(positive=True)

        occurrences = [{'sen': 0, 'ent': 0},
                       {'sen': 0, 'ent': 0},
                       {'sen': 0, 'ent': 1}]
        g = self.load_from_occurrences(occurrences)
        tvg.link_graph(g, 100)

        occurrences = [{'sen': 0, 'ent': 0},
                       {'sen': 0, 'ent': 2},
                       {'sen': 0, 'ent': 2}]
        g = self.load_from_occurrences(occurrences)
        tvg.link_graph(g, 200)

        occurrences = [{'sen': 0, 'ent': 0},
                       {'sen': 0, 'ent': 3},
                       {'sen': 0, 'ent': 3},
                       {'sen': 0, 'ent': 3}]
        g = self.load_from_occurrences(occurrences)
        tvg.link_graph(g, 300)

        with self.assertRaises(MemoryError):
            tvg.count_nodes(1, 0)

        v = tvg.count_nodes(51, 150)
        self.assertEqual(v.readonly, True)
        self.assertEqual(v[0], 1.0)
        self.assertEqual(v[1], 1.0)
        self.assertEqual(v[2], 0.0)
        self.assertEqual(v[3], 0.0)

        v = tvg.count_nodes(151, 250)
        self.assertEqual(v[0], 1.0)
        self.assertEqual(v[1], 0.0)
        self.assertEqual(v[2], 1.0)
        self.assertEqual(v[3], 0.0)

        v = tvg.count_nodes(251, 350)
        self.assertEqual(v[0], 1.0)
        self.assertEqual(v[1], 0.0)
        self.assertEqual(v[2], 0.0)
        self.assertEqual(v[3], 1.0)

        v = tvg.count_nodes(51, 350)
        self.assertEqual(v[0], 3.0)
        self.assertEqual(v[1], 1.0)
        self.assertEqual(v[2], 1.0)
        self.assertEqual(v[3], 1.0)

        del tvg

if __name__ == '__main__':
    unittest.main()
    gc.collect()
