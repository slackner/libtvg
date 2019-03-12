/*
 * Time-varying graph library
 * Graph functions.
 *
 * Copyright (c) 2018-2019 Sebastian Lackner
 */

#include <float.h>

#include "tvg.h"
#include "internal.h"

/* graph_debug relies on that */
C_ASSERT(sizeof(long long unsigned int) == sizeof(uint64_t));

struct graph *alloc_graph(uint32_t flags)
{
    static const uint32_t bits_source = 0;
    static const uint32_t bits_target = 0;
    const struct graph_ops *ops;
    struct graph *graph;
    struct bucket2 *buckets;
    uint64_t i, num_buckets;

    if (flags & ~(TVG_FLAGS_NONZERO |
                  TVG_FLAGS_POSITIVE |
                  TVG_FLAGS_DIRECTED))
        return NULL;

    num_buckets = 1ULL << (bits_source + bits_target);
    if (!(buckets = malloc(sizeof(*buckets) * num_buckets)))
        return NULL;

    for (i = 0; i < num_buckets; i++)
        init_bucket2(&buckets[i]);

    if (!(graph = malloc(sizeof(*graph))))
    {
        free(buckets);
        return NULL;
    }

    if (flags & TVG_FLAGS_POSITIVE)
    {
        ops = &graph_positive_ops;
        flags |= TVG_FLAGS_NONZERO;  /* positive implies nonzero */
    }
    else if (flags & TVG_FLAGS_NONZERO)
        ops = &graph_nonzero_ops;
    else
        ops = &graph_generic_ops;

    graph->refcount    = 1;
    graph->flags       = flags;
    graph->revision    = 0;
    graph->eps         = 0.0;
    graph->ts          = 0.0;
    graph->tvg         = NULL;
    list_init(&graph->entry);
    graph->ops         = ops;
    graph->bits_source = bits_source;
    graph->bits_target = bits_target;
    graph->buckets     = buckets;
    graph->optimize    = 0;
    graph->delta       = NULL;

    /* set a proper 'optimize' value */
    graph_optimize(graph);
    return graph;
}

struct graph *grab_graph(struct graph *graph)
{
    if (graph) graph->refcount++;
    return graph;
}

void free_graph(struct graph *graph)
{
    uint64_t i, num_buckets;

    if (!graph) return;
    if (--graph->refcount) return;

    num_buckets = 1ULL << (graph->bits_source + graph->bits_target);
    for (i = 0; i < num_buckets; i++)
        free_bucket2(&graph->buckets[i]);

    /* When the last reference is dropped the graph should no longer
     * be associated with a time-varying-graph object. Triggering this
     * assertion means that 'free_graph' was called too often. */
    assert(!graph->tvg);
    free_graph(graph->delta);
    free(graph->buckets);
    free(graph);
}

void unlink_graph(struct graph *graph)
{
    struct tvg *tvg;

    if (!graph || !(tvg = graph->tvg))
        return;

    list_remove(&graph->entry);
    graph->tvg = NULL;
}

struct graph *prev_graph(struct graph *graph)
{
    struct tvg *tvg;

    if (!graph || !(tvg = graph->tvg))
        return NULL;

    graph = LIST_PREV(graph, &tvg->graphs, struct graph, entry);
    if (graph) assert(graph->tvg == tvg);
    return grab_graph(graph);
}

struct graph *next_graph(struct graph *graph)
{
    struct tvg *tvg;

    if (!graph || !(tvg = graph->tvg))
        return NULL;

    graph = LIST_NEXT(graph, &tvg->graphs, struct graph, entry);
    if (graph) assert(graph->tvg == tvg);
    return grab_graph(graph);
}

int graph_enable_delta(struct graph *graph)
{
    uint32_t graph_flags;

    /* Do not set TVG_FLAGS_NONZERO / TVG_FLAGS_POSITIVE! */
    graph_flags = graph->flags & TVG_FLAGS_DIRECTED;

    free_graph(graph->delta);
    if (!(graph->delta = alloc_graph(graph_flags)))
        return 0;

    graph->delta_mul = 1.0;
    return 1;
}

void graph_disable_delta(struct graph *graph)
{
    free_graph(graph->delta);
    graph->delta = NULL;
}

struct graph *graph_get_delta(struct graph *graph, float *mul)
{
    if (!graph->delta)
        return NULL;

    *mul = graph->delta_mul;
    return grab_graph(graph->delta);
}

void graph_debug(struct graph *graph)
{
    struct entry2 *edge;

    fprintf(stderr, "Graph %p (revision %llu)\n", graph, (long long unsigned int)graph->revision);

    GRAPH_FOR_EACH_EDGE(graph, edge)
    {
        fprintf(stderr, "A[%llu, %llu] = %f\n", (long long unsigned int)edge->source,
                                                (long long unsigned int)edge->target, edge->weight);
    }
}

int graph_inc_bits_target(struct graph *graph)
{
    struct bucket2 *buckets;
    uint64_t i, num_buckets;
    uint64_t mask = 1ULL << graph->bits_target;

    if (graph->bits_target >= 31)
        return 0;

    num_buckets = 1ULL << (graph->bits_source + graph->bits_target);
    if (!(buckets = realloc(graph->buckets, sizeof(*buckets) * 2 * num_buckets)))
        return 0;

    graph->buckets = buckets;

    for (i = 0; i < num_buckets; i++)
    {
        init_bucket2(&buckets[i + num_buckets]);
        if (!bucket2_split(&buckets[i], &buckets[i + num_buckets], 0, mask))
        {
            /* FIXME: Error handling is mostly untested. */

            while (i--)
            {
                bucket2_merge(&buckets[i], &buckets[i + num_buckets]);
                free_bucket2(&buckets[i + num_buckets]);
            }

            if ((buckets = realloc(graph->buckets, sizeof(*buckets) * num_buckets)))
                graph->buckets = buckets;

            return 0;
        }
    }

    for (i = 0; i < 2 * num_buckets; i++)
        bucket2_compress(&buckets[i]);

    graph->bits_target++;
    return 1;
}

int graph_dec_bits_target(struct graph *graph)
{
    struct bucket2 *buckets;
    uint64_t i, num_buckets;
    uint64_t mask = 1ULL << (graph->bits_target - 1);

    if (!graph->bits_target)
        return 0;

    num_buckets = 1ULL << (graph->bits_source + graph->bits_target - 1);
    buckets = graph->buckets;

    for (i = 0; i < num_buckets; i++)
    {
        if (!bucket2_merge(&buckets[i], &buckets[i + num_buckets]))
        {
            /* FIXME: Error handling is mostly untested. */

            while (i--)
                bucket2_split(&buckets[i], &buckets[i + num_buckets], 0, mask);

            return 0;
        }
    }

    for (i = 0; i < num_buckets; i++)
    {
        bucket2_compress(&buckets[i]);
        free_bucket2(&buckets[i + num_buckets]);
    }

    if ((buckets = realloc(graph->buckets, sizeof(*buckets) * num_buckets)))
        graph->buckets = buckets;

    graph->bits_target--;
    return 1;
}

int graph_inc_bits_source(struct graph *graph)
{
    uint64_t num_source, num_target;
    struct bucket2 *buckets;
    uint64_t i, j, num_buckets;
    uint64_t mask = 1ULL << graph->bits_source;

    if (graph->bits_source >= 31)
        return 0;

    num_buckets = 1ULL << (graph->bits_source + graph->bits_target);
    num_source  = 1ULL << graph->bits_source;
    num_target  = 1ULL << graph->bits_target;
    if (!(buckets = realloc(graph->buckets, sizeof(*buckets) * 2 * num_buckets)))
        return 0;

    graph->buckets = buckets;

    for (i = num_target; i--;)
    {
        memmove(&buckets[i * 2 * num_source], &buckets[i * num_source],
                sizeof(*buckets) * num_source);
    }

    for (i = 0; i < 2 * num_buckets; i += 2 * num_source)
    for (j = 0; j < num_source; j++)
    {
        init_bucket2(&buckets[i + j + num_source]);
        if (!bucket2_split(&buckets[i + j], &buckets[i + j + num_source], mask, 0))
        {
            /* FIXME: Error handling is mostly untested. */

            for (;;)
            {
                while (j--)
                {
                    bucket2_merge(&buckets[i + j], &buckets[i + j + num_source]);
                    free_bucket2(&buckets[i + j + num_source]);
                }
                if (!i) break;
                i -= 2 * num_source;
                j = num_source;
            }

            for (i = 0; i < num_target; i++)
            {
                memmove(&buckets[i * num_source], &buckets[i * 2 * num_source],
                        sizeof(*buckets) * num_source);
            }

            if ((buckets = realloc(graph->buckets, sizeof(*buckets) * num_buckets)))
                graph->buckets = buckets;

            return 0;
        }
    }

    for (i = 0; i < 2 * num_buckets; i++)
        bucket2_compress(&buckets[i]);

    graph->bits_source++;
    return 1;
}

int graph_dec_bits_source(struct graph *graph)
{
    uint64_t num_source, num_target;
    uint64_t i, j, num_buckets;
    struct bucket2 *buckets;
    uint64_t mask = 1ULL << (graph->bits_source - 1);

    if (!graph->bits_source)
        return 0;

    num_buckets = 1ULL << (graph->bits_source + graph->bits_target - 1);
    num_source  = 1ULL << (graph->bits_source - 1);
    num_target  = 1ULL << graph->bits_target;
    buckets = graph->buckets;

    for (i = 0; i < 2 * num_buckets; i += 2 * num_source)
    for (j = 0; j < num_source; j++)
    {
        if (!bucket2_merge(&buckets[i + j], &buckets[i + j + num_source]))
        {
            /* FIXME: Error handling is mostly untested. */

            for (;;)
            {
                while (j--)
                    bucket2_split(&buckets[i + j], &buckets[i + j + num_source], mask, 0);
                if (!i) break;
                i -= 2 * num_source;
                j = num_source;
            }

            return 0;
        }
    }

    for (i = 0; i < 2 * num_buckets; i += 2 * num_source)
    for (j = 0; j < num_source; j++)
    {
        bucket2_compress(&buckets[i + j]);
        free_bucket2(&buckets[i + j + num_source]);
    }

    for (i = 0; i < num_target; i++)
    {
        memmove(&buckets[i * num_source], &buckets[i * 2 * num_source],
               sizeof(*buckets) * num_source);
    }

    if ((buckets = realloc(graph->buckets, sizeof(*buckets) * num_buckets)))
        graph->buckets = buckets;

    graph->bits_source--;
    return 1;
}

void graph_optimize(struct graph *graph)
{
    uint64_t i, num_buckets;
    uint64_t num_edges;

    num_buckets = 1ULL << (graph->bits_source + graph->bits_target);

    num_edges = 0;
    for (i = 0; i < num_buckets; i++)
        num_edges += graph->buckets[i].num_entries;

    /* Adjust the number of buckets if the graph is getting too dense.
     * For now, we prefer source bits over target bits. */

    if (num_edges >= num_buckets * 256)
    {
        while (num_edges >= num_buckets * 64)
        {
            if (graph->bits_source <= graph->bits_target)
            {
                if (!graph_inc_bits_source(graph)) goto error;
                num_buckets *= 2;
            }
            else
            {
                if (!graph_inc_bits_target(graph)) goto error;
                num_buckets *= 2;
            }
        }
    }

    if (num_buckets >= 2 && num_edges < num_buckets * 16)
    {
        while (num_buckets >= 2 && num_edges < num_buckets * 64)
        {
            if (graph->bits_source <= graph->bits_target)
            {
                if (!graph_dec_bits_target(graph)) goto error;
                num_buckets /= 2;
            }
            else
            {
                if (!graph_dec_bits_source(graph)) goto error;
                num_buckets /= 2;
            }
        }
    }

    graph->optimize = MIN(num_buckets * 256 - num_edges, num_edges - num_buckets * 16);
    graph->optimize = MAX(graph->optimize, 256);
    if (!(graph->flags & TVG_FLAGS_DIRECTED)) graph->optimize /= 2;
    return;

error:
    fprintf(stderr, "Failed to optimize graph, trying again later.\n");
    graph->optimize = 1024;
}

void graph_set_eps(struct graph *graph, float eps)
{
    graph->eps = fabs(eps);
    graph->ops->mul_const(graph, 1.0);
}

int graph_empty(struct graph *graph)
{
    struct entry2 *edge;

    GRAPH_FOR_EACH_EDGE(graph, edge)
    {
        return 0;
    }

    return 1;
}

int graph_has_edge(struct graph *graph, uint64_t source, uint64_t target)
{
    uint32_t i;
    /* keep in sync with _graph_get_bucket! */
    i  = target & ((1ULL << graph->bits_target) - 1);
    i  = (i << graph->bits_source);
    i |= source & ((1ULL << graph->bits_source) - 1);
    return bucket2_get_entry(&graph->buckets[i], source, target, 0) != NULL;
}

float graph_get_edge(struct graph *graph, uint64_t source, uint64_t target)
{
    return graph->ops->get(graph, source, target);
}

uint64_t graph_get_edges(struct graph *graph, uint64_t *indices, float *weights, uint64_t max_edges)
{
    uint64_t count = 0;
    struct entry2 *edge;

    /* For undirected graph, the GRAPH_FOR_EACH_EDGE macro will
     * automatically skip edges in the reverse direction */

    GRAPH_FOR_EACH_EDGE(graph, edge)
    {
        if (count++ >= max_edges) continue;
        if (indices)
        {
            *indices++ = edge->source;
            *indices++ = edge->target;
        }
        if (weights)
        {
            *weights++ = edge->weight;
        }
    }

    return count;
}

uint64_t graph_get_adjacent_edges(struct graph *graph, uint64_t source, uint64_t *indices, float *weights, uint64_t max_edges)
{
    uint64_t count = 0;
    struct entry2 *edge;

    GRAPH_FOR_EACH_ADJACENT_EDGE(graph, source, edge)
    {
        if (count++ >= max_edges) continue;
        if (indices)
        {
            assert(edge->source == source);
            *indices++ = edge->target;
        }
        if (weights)
        {
            *weights++ = edge->weight;
        }
    }

    return count;
}

int graph_set_edge(struct graph *graph, uint64_t source, uint64_t target, float weight)
{
    return graph->ops->set(graph, source, target, weight);
}

int graph_set_edges(struct graph *graph, uint64_t *indices, float *weights, uint64_t num_edges)
{
    while (num_edges--)
    {
        if (!graph->ops->set(graph, indices[0], indices[1], weights[0]))
            return 0;

        indices += 2;
        weights++;
    }

    return 1;
}

int graph_add_edge(struct graph *graph, uint64_t source, uint64_t target, float weight)
{
    return graph->ops->add(graph, source, target, weight);
}

int graph_add_edges(struct graph *graph, uint64_t *indices, float *weights, uint64_t num_edges)
{
    while (num_edges--)
    {
        if (!graph->ops->add(graph, indices[0], indices[1], weights[0]))
            return 0;

        indices += 2;
        weights++;
    }

    return 1;
}

int graph_add_graph(struct graph *out, struct graph *graph, float weight)
{
    struct entry2 *edge;

    if ((out->flags ^ graph->flags) & TVG_FLAGS_DIRECTED)
        return 0;

    GRAPH_FOR_EACH_EDGE(graph, edge)
    {
        if (!graph_add_edge(out, edge->source, edge->target, edge->weight * weight))
            return 0;
    }

    /* graph_add_edge already updated the revision */
    return 1;
}

int graph_sub_edge(struct graph *graph, uint64_t source, uint64_t target, float weight)
{
    return graph->ops->add(graph, source, target, -weight);
}

int graph_sub_edges(struct graph *graph, uint64_t *indices, float *weights, uint64_t num_edges)
{
    while (num_edges--)
    {
        if (!graph->ops->add(graph, indices[0], indices[1], -weights[0]))
            return 0;

        indices += 2;
        weights++;
    }

    return 1;
}

int graph_sub_graph(struct graph *out, struct graph *graph, float weight)
{
    return graph_add_graph(out, graph, -weight);
}

void graph_del_edge(struct graph *graph, uint64_t source, uint64_t target)
{
    graph->ops->del(graph, source, target);
}

void graph_del_edges(struct graph *graph, uint64_t *indices, uint64_t num_edges)
{
    while (num_edges--)
    {
        graph->ops->del(graph, indices[0], indices[1]);
        indices += 2;
    }
}

void graph_mul_const(struct graph *graph, float constant)
{
    graph->ops->mul_const(graph, constant);
}

struct vector *graph_mul_vector(const struct graph *graph, /* const */ struct vector *vector)
{
    struct vector *out;
    uint64_t i, num_buckets;
    struct entry1 *entry;
    struct entry2 *edge;

    while (vector->bits != graph->bits_source)
    {
        if (vector->bits < graph->bits_source)
        {
            if (!vector_inc_bits(vector)) return NULL;
            vector->optimize = 1024;
        }
        else
        {
            if (!vector_dec_bits(vector)) return NULL;
            vector->optimize = 1024;
        }
    }

    /* FIXME: Appropriate flags? */
    if (!(out = alloc_vector(TVG_FLAGS_NONZERO)))
        return NULL;

    assert(vector->bits == graph->bits_source);

    num_buckets = 1ULL << (graph->bits_source + graph->bits_target);
    for (i = 0; i < num_buckets; i++)
    {
        BUCKET21_FOR_EACH_ENTRY(&graph->buckets[i], edge, &vector->buckets[i >> graph->bits_source], entry)
        {
            if (!entry) continue;
            if (!vector_add_entry(out, edge->source, edge->weight * entry->weight))
            {
                free_vector(out);
                return NULL;
            }
        }
    }

    return out;
}

struct vector *graph_in_degrees(const struct graph *graph)
{
    struct vector *vector;
    struct entry2 *edge;

    /* FIXME: Appropriate flags? */
    if (!(vector = alloc_vector(TVG_FLAGS_NONZERO)))
        return NULL;

    GRAPH_FOR_EACH_DIRECTED_EDGE(graph, edge)
    {
        if (!vector_add_entry(vector, edge->target, 1.0))
        {
            free_vector(vector);
            return NULL;
        }
    }

    return vector;
}

struct vector *graph_in_weights(const struct graph *graph)
{
    struct vector *vector;
    struct entry2 *edge;

    /* FIXME: Appropriate flags? */
    if (!(vector = alloc_vector(TVG_FLAGS_NONZERO)))
        return NULL;

    GRAPH_FOR_EACH_DIRECTED_EDGE(graph, edge)
    {
        if (!vector_add_entry(vector, edge->target, edge->weight))
        {
            free_vector(vector);
            return NULL;
        }
    }

    return vector;
}

struct vector *graph_out_degrees(const struct graph *graph)
{
    struct vector *vector;
    struct entry2 *edge;

    /* FIXME: Appropriate flags? */
    if (!(vector = alloc_vector(TVG_FLAGS_NONZERO)))
        return NULL;

    GRAPH_FOR_EACH_DIRECTED_EDGE(graph, edge)
    {
        if (!vector_add_entry(vector, edge->source, 1.0))
        {
            free_vector(vector);
            return NULL;
        }
    }

    return vector;
}

struct vector *graph_out_weights(const struct graph *graph)
{
    struct vector *vector;
    struct entry2 *edge;

    /* FIXME: Appropriate flags? */
    if (!(vector = alloc_vector(TVG_FLAGS_NONZERO)))
        return NULL;

    GRAPH_FOR_EACH_DIRECTED_EDGE(graph, edge)
    {
        if (!vector_add_entry(vector, edge->source, edge->weight))
        {
            free_vector(vector);
            return NULL;
        }
    }

    return vector;
}

struct vector *graph_power_iteration(const struct graph *graph, uint32_t num_iterations, double *eigenvalue_out)
{
    struct vector *vector;
    struct vector *temp;
    struct entry2 *edge;

    if (!num_iterations)
        num_iterations = 100;

    /* FIXME: Appropriate flags? */
    if (!(vector = alloc_vector(TVG_FLAGS_NONZERO)))
        return NULL;

    GRAPH_FOR_EACH_DIRECTED_EDGE(graph, edge)
    {
        if (vector_has_entry(vector, edge->target)) continue;
        if (!vector_add_entry(vector, edge->target, random_float()))
        {
            free_vector(vector);
            return NULL;
        }
    }

    while (num_iterations--)
    {
        if (!(temp = graph_mul_vector(graph, vector)))
        {
            free_vector(vector);
            return NULL;
        }

        free_vector(vector);
        vector = temp;

        vector_mul_const(vector, 1.0 / vector_norm(vector));
    }

    if (eigenvalue_out)
    {
        if (!(temp = graph_mul_vector(graph, vector)))
        {
            free_vector(vector);
            return NULL;
        }
        *eigenvalue_out = vector_mul_vector(vector, temp);
        free_vector(temp);
    }

    return vector;
}
