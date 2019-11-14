/*
 * Time-varying graph library
 * Graph functions.
 *
 * Copyright (c) 2018-2019 Sebastian Lackner
 */

#include <float.h>

#include "internal.h"

static inline struct bucket2 *_graph_get_bucket(struct graph *graph, uint64_t source, uint64_t target)
{
    uint32_t i;
    i  = (uint32_t)(target & ((1ULL << graph->bits_target) - 1));
    i  = (i << graph->bits_source);
    i |= (uint32_t)(source & ((1ULL << graph->bits_source) - 1));
    return &graph->buckets[i];
}

static inline struct entry2 *_graph_get_edge(struct graph *graph, uint64_t source, uint64_t target, int allocate)
{
    struct bucket2 *bucket = _graph_get_bucket(graph, source, target);
    return bucket2_get_entry(bucket, source, target, allocate);
}

static inline int _graph_del_edge(struct graph *graph, uint64_t source, uint64_t target)
{
    struct bucket2 *bucket;
    struct entry2 *edge;

    bucket = _graph_get_bucket(graph, source, target);
    if (!(edge = bucket2_get_entry(bucket, source, target, 0)))
        return 0;

    bucket2_del_entry(bucket, edge);
    return 1;
}

int graph_has_edge(struct graph *graph, uint64_t source, uint64_t target)
{
    return _graph_get_edge(graph, source, target, 0) != NULL;
}

float graph_get_edge(struct graph *graph, uint64_t source, uint64_t target)
{
    struct entry2 *edge;

    if (!(edge = _graph_get_edge(graph, source, target, 0)))
        return 0.0;

    return edge->weight;
}

static int generic_clear(struct graph *graph)
{
    uint64_t i, num_buckets;

    if (UNLIKELY(graph->readonly))
        return 0;

    num_buckets = 1ULL << (graph->bits_source + graph->bits_target);
    for (i = 0; i < num_buckets; i++)
        bucket2_clear(&graph->buckets[i]);

    graph->revision++;
    if (!--graph->optimize)
        graph_optimize(graph);

    return 1;
}

static int generic_set(struct graph *graph, uint64_t source, uint64_t target, float weight)
{
    struct entry2 *edge;

    if (UNLIKELY(graph->readonly))
        return 0;

    if (!(edge = _graph_get_edge(graph, source, target, 1)))
        return 0;

    edge->weight = weight;

    if (!(graph->flags & TVG_FLAGS_DIRECTED) && source != target)
    {
        if (!(edge = _graph_get_edge(graph, target, source, 1)))
        {
            /* Allocation failed, restore the original state. */
            _graph_del_edge(graph, source, target);
            graph->revision++;
            return 0;
        }

        edge->weight = weight;
    }

    graph->revision++;
    if (!--graph->optimize)
        graph_optimize(graph);

    return 1;
}

static int generic_add(struct graph *graph, uint64_t source, uint64_t target, float weight)
{
    struct entry2 *edge;

    if (UNLIKELY(graph->readonly))
        return 0;

    if (!(edge = _graph_get_edge(graph, source, target, 1)))
        return 0;

    weight += edge->weight;
    edge->weight = weight;

    if (!(graph->flags & TVG_FLAGS_DIRECTED) && source != target)
    {
        if (!(edge = _graph_get_edge(graph, target, source, 1)))
        {
            /* Allocation failed, restore the original state. */
            _graph_del_edge(graph, source, target);
            graph->revision++;
            return 0;
        }

        edge->weight = weight;
    }

    graph->revision++;
    if (!--graph->optimize)
        graph_optimize(graph);

    return 1;
}

static int generic_del(struct graph *graph, uint64_t source, uint64_t target)
{
    int changed = 0;

    if (UNLIKELY(graph->readonly))
        return 0;

    if (_graph_del_edge(graph, source, target))
        changed = 1;

    if (!(graph->flags & TVG_FLAGS_DIRECTED) && source != target)
    {
        if (_graph_del_edge(graph, target, source))
            changed = 1;
    }

    if (!changed)
        return 1;

    graph->revision++;
    if (!--graph->optimize)
        graph_optimize(graph);

    return 1;
}

static int generic_mul_const(struct graph *graph, float constant)
{
    struct entry2 *edge;

    if (UNLIKELY(graph->readonly))
        return 0;

    GRAPH_FOR_EACH_DIRECTED_EDGE(graph, edge)
    {
        edge->weight *= constant;
    }

    graph->revision++;
    return 1;
}

static int generic_set_eps(struct graph *graph, float eps)
{
    if (UNLIKELY(graph->readonly))
        return 0;

    graph->eps = (float)fabs(eps);
    return 1;
}

const struct graph_ops graph_generic_ops =
{
    generic_set_eps,
    generic_clear,
    generic_set,
    generic_add,
    generic_del,
    generic_mul_const,
};

static int nonzero_mul_const(struct graph *graph, float constant)
{
    struct entry2 *edge, *out;
    struct bucket2 *bucket;
    uint64_t i, num_buckets;

    if (UNLIKELY(graph->readonly))
        return 0;

    num_buckets = 1ULL << (graph->bits_source + graph->bits_target);
    for (i = 0; i < num_buckets; i++)
    {
        bucket = &graph->buckets[i];
        out = &bucket->entries[0];

        BUCKET2_FOR_EACH_ENTRY(bucket, edge)
        {
            edge->weight *= constant;
            if (fabs(edge->weight) <= graph->eps)
                continue;
            *out++ = *edge;
        }

        bucket->num_entries = (uint64_t)(out - &bucket->entries[0]);
        assert(bucket->num_entries <= bucket->max_entries);
    }

    graph->revision++;
    /* FIXME: Trigger graph_optimize? */
    return 1;
}

static int nonzero_set_eps(struct graph *graph, float eps)
{
    if (UNLIKELY(graph->readonly))
        return 0;

    graph->eps = (float)fabs(eps);
    return nonzero_mul_const(graph, 1.0);
}

const struct graph_ops graph_nonzero_ops =
{
    nonzero_set_eps,
    generic_clear,
    generic_set,
    generic_add,
    generic_del,
    nonzero_mul_const,
};

static int positive_mul_const(struct graph *graph, float constant)
{
    struct entry2 *edge, *out;
    struct bucket2 *bucket;
    uint64_t i, num_buckets;

    if (UNLIKELY(graph->readonly))
        return 0;

    num_buckets = 1ULL << (graph->bits_source + graph->bits_target);
    for (i = 0; i < num_buckets; i++)
    {
        bucket = &graph->buckets[i];
        out = &bucket->entries[0];

        BUCKET2_FOR_EACH_ENTRY(bucket, edge)
        {
            edge->weight *= constant;
            if (edge->weight <= graph->eps)
                continue;
            *out++ = *edge;
        }

        bucket->num_entries = (uint64_t)(out - &bucket->entries[0]);
        assert(bucket->num_entries <= bucket->max_entries);
    }

    graph->revision++;
    /* FIXME: Trigger graph_optimize? */
    return 1;
}

static int positive_set_eps(struct graph *graph, float eps)
{
    if (UNLIKELY(graph->readonly))
        return 0;

    graph->eps = (float)fabs(eps);
    return positive_mul_const(graph, 1.0);
}

const struct graph_ops graph_positive_ops =
{
    positive_set_eps,
    generic_clear,
    generic_set,
    generic_add,
    generic_del,
    positive_mul_const,
};
