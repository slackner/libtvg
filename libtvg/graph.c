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
    objectid_init(&graph->objectid);
    graph->tvg         = NULL;
    list_init(&graph->entry);
    graph->cache       = 0;
    list_init(&graph->cache_entry);
    graph->ops         = ops;
    graph->bits_source = bits_source;
    graph->bits_target = bits_target;
    graph->buckets     = buckets;
    graph->optimize    = 0;

    /* set a proper 'optimize' value */
    graph_optimize(graph);
    return graph;
}

struct graph *grab_graph(struct graph *graph)
{
    if (graph) __sync_fetch_and_add(&graph->refcount, 1);
    return graph;
}

void free_graph(struct graph *graph)
{
    uint64_t i, num_buckets;

    if (!graph) return;
    if (__sync_sub_and_fetch(&graph->refcount, 1)) return;

    num_buckets = 1ULL << (graph->bits_source + graph->bits_target);
    for (i = 0; i < num_buckets; i++)
        free_bucket2(&graph->buckets[i]);

    /* When the last reference is dropped the graph should no longer
     * be associated with a time-varying-graph object. Triggering this
     * assertion means that 'free_graph' was called too often. */
    assert(!graph->tvg);
    assert(!graph->cache);

    free(graph->buckets);
    free(graph);
}

void unlink_graph(struct graph *graph)
{
    struct graph *other_graph;
    struct tvg *tvg;

    if (!graph || !(tvg = graph->tvg))
        return;

    if (!objectid_empty(&graph->objectid))  /* we have to reload later */
        graph->flags |= (TVG_FLAGS_LOAD_NEXT | TVG_FLAGS_LOAD_PREV);

    if (graph->flags & TVG_FLAGS_LOAD_NEXT)
    {
        other_graph = LIST_PREV(graph, &tvg->graphs, struct graph, entry);
        if (other_graph) other_graph->flags |= TVG_FLAGS_LOAD_NEXT;
    }
    if (graph->flags & TVG_FLAGS_LOAD_PREV)
    {
        other_graph = LIST_NEXT(graph, &tvg->graphs, struct graph, entry);
        if (other_graph) other_graph->flags |= TVG_FLAGS_LOAD_PREV;
    }

    if (graph->cache)
    {
        list_remove(&graph->cache_entry);
        tvg->cache_used -= graph->cache;
        graph->cache = 0;
    }

    list_remove(&graph->entry);
    graph->tvg = NULL;
    free_graph(graph);
}

void graph_refresh_cache(struct graph *graph)
{
    if (!graph->cache) return;
    assert(graph->tvg != NULL);
    list_remove(&graph->cache_entry);
    list_add_tail(&graph->tvg->cache, &graph->cache_entry);
}

void graph_debug(struct graph *graph)
{
    struct entry2 *edge;
    char objectid_str[32];

    objectid_to_str(&graph->objectid, objectid_str);
    fprintf(stderr, "Graph %p (ts %llu, objectid %s, revision %llu)\n", graph,
            (long long unsigned int)graph->ts, objectid_str,
            (long long unsigned int)graph->revision);

    GRAPH_FOR_EACH_EDGE(graph, edge)
    {
        fprintf(stderr, "A[%llu, %llu] = %f\n", (long long unsigned int)edge->source,
                                                (long long unsigned int)edge->target, edge->weight);
    }
}

uint64_t graph_memory_usage(struct graph *graph)
{
    uint64_t i, num_buckets;
    struct bucket2 *bucket;
    uint64_t size = sizeof(*graph);

    /* In the following, we underestimate the memory usage a bit, since
     * we do not take into account the heap structure itself. */

    num_buckets = 1ULL << (graph->bits_source + graph->bits_target);
    size += sizeof(*bucket) * num_buckets;

    for (i = 0; i < num_buckets; i++)
    {
        bucket = &graph->buckets[i];
        size += sizeof(bucket->entries) * bucket->max_entries;
    }

    return size;
}

struct graph *prev_graph(struct graph *graph)
{
    struct tvg *tvg;

    if (!graph || !(tvg = graph->tvg))
        return NULL;

    /* Ensure that the user has one additional reference. */
    assert(__sync_fetch_and_add(&graph->refcount, 0) >= 2);

    if (graph->flags & TVG_FLAGS_LOAD_PREV)
        tvg_load_prev_graph(tvg, graph);

    graph = LIST_PREV(graph, &tvg->graphs, struct graph, entry);
    if (!graph)
        return NULL;

    assert(graph->tvg == tvg);
    graph_refresh_cache(graph);
    return grab_graph(graph);
}

struct graph *next_graph(struct graph *graph)
{
    struct tvg *tvg;

    if (!graph || !(tvg = graph->tvg))
        return NULL;

    /* Ensure that the user has one additional reference. */
    assert(__sync_fetch_and_add(&graph->refcount, 0) >= 2);

    if (graph->flags & TVG_FLAGS_LOAD_NEXT)
        tvg_load_next_graph(tvg, graph);

    graph = LIST_NEXT(graph, &tvg->graphs, struct graph, entry);
    if (!graph)
        return NULL;

    assert(graph->tvg == tvg);
    graph_refresh_cache(graph);
    return grab_graph(graph);
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
    graph->optimize = MAX(graph->optimize, 256ULL);
    if (!(graph->flags & TVG_FLAGS_DIRECTED)) graph->optimize /= 2;
    return;

error:
    fprintf(stderr, "%s: Failed to optimize graph, trying again later.\n", __func__);
    graph->optimize = 1024;
}

void graph_set_eps(struct graph *graph, float eps)
{
    graph->eps = (float)fabs(eps);
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
    i  = (uint32_t)(target & ((1ULL << graph->bits_target) - 1));
    i  = (i << graph->bits_source);
    i |= (uint32_t)(source & ((1ULL << graph->bits_source) - 1));
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
    if (weights)
    {
        while (num_edges--)
        {
            if (!graph->ops->set(graph, indices[0], indices[1], weights[0]))
                return 0;

            indices += 2;
            weights++;
        }
    }
    else
    {
        while (num_edges--)
        {
            if (!graph->ops->set(graph, indices[0], indices[1], 1.0f))
                return 0;

            indices += 2;
        }
    }

    return 1;
}

int graph_add_edge(struct graph *graph, uint64_t source, uint64_t target, float weight)
{
    return graph->ops->add(graph, source, target, weight);
}

int graph_add_edges(struct graph *graph, uint64_t *indices, float *weights, uint64_t num_edges)
{
    if (weights)
    {
        while (num_edges--)
        {
            if (!graph->ops->add(graph, indices[0], indices[1], weights[0]))
                return 0;

            indices += 2;
            weights++;
        }
    }
    else
    {
        while (num_edges--)
        {
            if (!graph->ops->add(graph, indices[0], indices[1], 1.0f))
                return 0;

            indices += 2;
        }
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
        /* Note that we intentionally don't use out->ops->add here. It is safer to
         * always use exported functions when working on other objects. */
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
    if (weights)
    {
        while (num_edges--)
        {
            if (!graph->ops->add(graph, indices[0], indices[1], -weights[0]))
                return 0;

            indices += 2;
            weights++;
        }
    }
    else
    {
        while (num_edges--)
        {
            if (!graph->ops->add(graph, indices[0], indices[1], -1.0f))
                return 0;

            indices += 2;
        }
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

struct vector *graph_mul_vector(const struct graph *graph, const struct vector *vector)
{
    struct vector *out;
    struct entry1 *entry;
    struct entry2 *edge;

    /* FIXME: Appropriate flags? */
    if (!(out = alloc_vector(TVG_FLAGS_NONZERO)))
        return NULL;

    GRAPH_VECTOR_FOR_EACH_EDGE(graph, edge, vector, entry)
    {
        if (!entry) continue;
        if (!vector_add_entry(out, edge->source, edge->weight * entry->weight))
        {
            free_vector(out);
            return NULL;
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

struct vector *graph_degree_anomalies(const struct graph *graph)
{
    struct vector *vector, *temp;
    struct entry1 *entry;
    struct entry2 *edge;

    if (!(vector = graph_out_degrees(graph)))
        return NULL;

    if (!(temp = alloc_vector(TVG_FLAGS_NONZERO)))
    {
        free_vector(vector);
        return NULL;
    }

    GRAPH_FOR_EACH_DIRECTED_EDGE(graph, edge)
    {
        if (!vector_add_entry(temp, edge->source, vector_get_entry(vector, edge->target)))
        {
            free_vector(vector);
            free_vector(temp);
            return NULL;
        }
    }

    VECTOR_FOR_EACH_ENTRY(vector, entry)
    {
        entry->weight -= vector_get_entry(temp, entry->index) / entry->weight;
    }

    free_vector(temp);
    return vector;
}

struct vector *graph_weight_anomalies(const struct graph *graph)
{
    struct vector *vector, *temp;
    struct entry1 *entry;
    struct entry2 *edge;

    if (!(vector = graph_out_weights(graph)))
        return NULL;

    if (!(temp = alloc_vector(TVG_FLAGS_NONZERO)))
    {
        free_vector(vector);
        return NULL;
    }

    GRAPH_FOR_EACH_DIRECTED_EDGE(graph, edge)
    {
        if (!vector_add_entry(temp, edge->source, edge->weight * vector_get_entry(vector, edge->target)))
        {
            free_vector(vector);
            free_vector(temp);
            return NULL;
        }
    }

    VECTOR_FOR_EACH_ENTRY(vector, entry)
    {
        entry->weight -= vector_get_entry(temp, entry->index) / entry->weight;
    }

    free_vector(temp);
    return vector;
}

struct vector *graph_power_iteration(const struct graph *graph, struct vector *initial_guess,
                                     uint32_t num_iterations, double tolerance, double *ret_eigenvalue)
{
    struct vector *vector;
    struct vector *temp;
    struct entry2 *edge;
    float value;

    if (!num_iterations)
        num_iterations = 100;

    /* FIXME: Appropriate flags? */
    if (!(vector = alloc_vector(TVG_FLAGS_NONZERO)))
        return NULL;

    GRAPH_FOR_EACH_DIRECTED_EDGE(graph, edge)
    {
        if (vector_has_entry(vector, edge->target)) continue;
        if (!initial_guess || !(value = vector_get_entry(initial_guess, edge->target)))
            value = random_float();
        if (!vector_add_entry(vector, edge->target, value))
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

        vector_mul_const(temp, (float)(1.0 / vector_norm(temp)));

        if (tolerance > 0.0 && vector_sub_vector_norm(vector, temp) <= tolerance)
            num_iterations = 0;

        free_vector(vector);
        vector = temp;
    }

    if (ret_eigenvalue)
    {
        if (!(temp = graph_mul_vector(graph, vector)))
        {
            free_vector(vector);
            return NULL;
        }
        *ret_eigenvalue = vector_mul_vector(vector, temp);
        free_vector(temp);
    }

    return vector;
}

struct graph *graph_filter_nodes(const struct graph *graph, struct vector *nodes)
{
    uint32_t graph_flags;
    struct entry2 *edge;
    struct graph *out;

    graph_flags = graph->flags & (TVG_FLAGS_NONZERO |
                                  TVG_FLAGS_POSITIVE |
                                  TVG_FLAGS_DIRECTED);


    if (!(out = alloc_graph(graph_flags)))
        return NULL;

    GRAPH_FOR_EACH_EDGE(graph, edge)
    {
        if (!vector_has_entry(nodes, edge->source)) continue;
        if (!vector_has_entry(nodes, edge->target)) continue;

        if (!graph_set_edge(out, edge->source, edge->target, edge->weight))
        {
            free_graph(out);
            return NULL;
        }
    }

    return out;
}
