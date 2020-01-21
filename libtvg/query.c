/*
 * Time-varying graph library
 * Query functions.
 *
 * Copyright (c) 2019 Sebastian Lackner
 */

#include "internal.h"

struct operation
{
    struct query *query;
    uint64_t      ts_min;
    uint64_t      ts_max;
    int64_t       weight;
};

static int _sort_operation_by_weight(const void *a, const void *b, void *userdata)
{
    const struct operation *ma = a, *mb = b;
    int res;

    if ((res = -COMPARE(ma->weight, mb->weight))) return res;  /* positive weights first */
    return COMPARE(ma->ts_min, mb->ts_min);
}

static inline double sub_uint64(uint64_t a, uint64_t b)
{
    return (a < b) ? -(double)(b - a) : (double)(a - b);
}

static int graph_add_count_edges(struct graph *out, struct graph *graph, float weight)
{
    struct entry2 *edge;

    if ((out->flags ^ graph->flags) & TVG_FLAGS_DIRECTED)
        return 0;

    GRAPH_FOR_EACH_EDGE(graph, edge)
    {
        if (!graph_add_edge(out, edge->source, edge->target, weight))
            return 0;
    }

    /* graph_add_edge already updated the revision */
    return 1;
}

static int vector_add_count_nodes(struct vector *out, struct graph *graph, float weight)
{
    struct vector *visited;
    struct entry2 *edge;
    int ret = 0;

    if (!(visited = alloc_vector(0)))
        return 0;

    GRAPH_FOR_EACH_EDGE(graph, edge)
    {
        if (!vector_has_entry(visited, edge->source))
        {
            if (!vector_add_entry(out, edge->source, weight))
                goto error;
            if (!vector_set_entry(visited, edge->source, 1))
                goto error;
        }
        if (!vector_has_entry(visited, edge->target))
        {
            if (!vector_add_entry(out, edge->target, weight))
                goto error;
            if (!vector_set_entry(visited, edge->target, 1))
                goto error;
        }
    }

    ret = 1;

error:
    free_vector(visited);
    return ret;
}

static void query_init(struct query *query, const struct query_ops *ops,
                       uint64_t ts_min, uint64_t ts_max)
{
    query->tvg      = NULL;
    list_init(&query->entry);
    query->cache    = 0;
    list_init(&query->cache_entry);
    query->ops      = ops;
    query->ts_min   = ts_min;
    query->ts_max   = ts_max;
}

static void query_refresh_cache(struct query *query)
{
    if (!query->cache) return;
    assert(query->tvg != NULL);
    list_remove(&query->cache_entry);
    list_add_tail(&query->tvg->query_cache, &query->cache_entry);
}

static void *query_compute(struct tvg *tvg, struct query *current)
{
    struct range *range, *next_range;
    struct query *query, *next_query;
    int64_t weight, best_weight;
    struct operation operation;
    struct query *best_query;
    int64_t delta, best_delta;
    struct minheap *queue = NULL;
    struct ranges *ranges = NULL;
    struct graph *graph;
    struct list compatible;
    uint64_t num_queries = 0;
    uint64_t num_graphs = 0;
    uint64_t duration;
    void *result = NULL;

    duration = clock_monotonic();

    if (!(ranges = alloc_ranges()))
        goto done;

    if (!ranges_add_range(ranges, current->ts_min, current->ts_max - current->ts_min + 1, -1))
        goto done;

    if (!(queue = alloc_minheap(sizeof(struct operation), _sort_operation_by_weight, NULL)))
        goto done;

    list_init(&compatible);
    LIST_FOR_EACH(query, &tvg->queries, struct query, entry)
    {
        if (!current->ops->compatible(current, query)) continue;
        list_add_tail(&compatible, &query->todo_entry);
    }

    while (!ranges_empty(ranges))
    {
        best_query = NULL;
        best_delta  = 0;
        best_weight = 0;

        LIST_FOR_EACH(query, &compatible, struct query, todo_entry)
        {
            delta = ranges_get_delta_length(ranges, query->ts_min, query->ts_max - query->ts_min + 1, &weight);
            if (delta >= best_delta) continue;

            /* FIXME: Remove entries if they are definitely not needed. */

            best_query  = query;
            best_delta  = delta;
            best_weight = weight;
        }

        if (!best_query)
            break;

        list_remove(&best_query->todo_entry);
        query_refresh_cache(best_query);

        operation.query     = best_query;
        operation.ts_min    = best_query->ts_min;
        operation.ts_max    = best_query->ts_max;
        operation.weight    = best_weight;

        if (!minheap_push(queue, &operation))
            goto done;

        if (!ranges_add_range(ranges, best_query->ts_min, best_query->ts_max - best_query->ts_min + 1, best_weight))
            goto done;
    }

    /* Fast-path: If the query is identical to a previous query, then
     * don't bother recomputing. Return a reference to the previous result. */

    if (ranges_empty(ranges) && minheap_count(queue) == 1)
    {
        if (minheap_pop(queue, &operation))
        {
            assert(operation.weight == 1);
            result = operation.query->ops->grab(operation.query);
            num_queries = 1;
        }

        goto done;
    }

    AVL_FOR_EACH_SAFE(range, next_range, &ranges->tree, struct range, entry)
    {
        operation.query     = NULL;
        operation.ts_min    = range->pos;
        operation.ts_max    = range->pos + range->len - 1;
        operation.weight    = -range->weight;

        if (!minheap_push(queue, &operation))
            goto done;

        avl_remove(&range->entry);
        free(range);
    }

    while (minheap_pop(queue, &operation))
    {
        if (operation.query)
        {
            if (!current->ops->add_query(current, operation.query, operation.weight))
                goto done;

            num_queries++;
            continue;
        }

        TVG_FOR_EACH_GRAPH_GE(tvg, graph, operation.ts_min)
        {
            if (graph->ts > operation.ts_max) break;
            if (!current->ops->add_graph(current, graph, operation.weight))
            {
                free_graph(graph);
                goto done;
            }
            num_graphs++;
        }
    }

    if (!current->ops->finalize(current))
        goto done;

    result = current->ops->grab(current);

    current->tvg = tvg;
    list_add_head(&tvg->queries, &current->entry);

    if (current->cache)
    {
        list_add_tail(&tvg->query_cache, &current->cache_entry);
        tvg->query_cache_used += current->cache;
        current = NULL; /* don't free it */
    }

done:
    free_minheap(queue);
    free_ranges(ranges);
    if (current) current->ops->free(current);

    LIST_FOR_EACH_SAFE(query, next_query, &tvg->query_cache, struct query, cache_entry)
    {
        assert(query->tvg == tvg);
        assert(query->cache != 0);
        if (tvg->query_cache_used <= tvg->query_cache_size) break;

        unlink_query(query, 0);
    }

    if (result && tvg->verbosity)
    {
        duration = clock_monotonic() - duration;

        fprintf(stderr, "%s: Computed query (using %llu queries and %llu graphs) in %llu ms\n", __func__,
                (long long unsigned int)num_queries, (long long unsigned int)num_graphs,
                (long long unsigned int)duration);
        fprintf(stderr, "%s: Query cache usage %llu / %llu (%.03f%%)\n", __func__,
                (long long unsigned int)tvg->query_cache_used,
                (long long unsigned int)tvg->query_cache_size,
                tvg->query_cache_size ? (float)tvg->query_cache_used * 100.0 / tvg->query_cache_size : 100.0);
    }

    return result;
}

/* query_sum_edges functions */

const struct query_ops query_sum_edges_ops;

static inline struct query_sum_edges *QUERY_SUM_EDGES(struct query *query_base)
{
    assert(query_base->ops == &query_sum_edges_ops);
    return CONTAINING_RECORD(query_base, struct query_sum_edges, base);
}

static void *query_sum_edges_grab(struct query *query_base)
{
    struct query_sum_edges *query = QUERY_SUM_EDGES(query_base);
    return grab_graph(query->result);
}

static void query_sum_edges_free(struct query *query_base)
{
    struct query_sum_edges *query = QUERY_SUM_EDGES(query_base);
    free_graph(query->result);
}

static int query_sum_edges_compatible(struct query *query_base, struct query *other_base)
{
    struct query_sum_edges *query = QUERY_SUM_EDGES(query_base);
    struct query_sum_edges *other;

    if (other_base->ops != &query_sum_edges_ops)
        return 0;

    other = QUERY_SUM_EDGES(other_base);

    if (query->eps != other->eps)
        return 0;

    return 1;
}

static int query_sum_edges_add_graph(struct query *query_base, struct graph *graph, int64_t weight)
{
    struct query_sum_edges *query = QUERY_SUM_EDGES(query_base);
    return graph_add_graph(query->result, graph, weight);
}

static int query_sum_edges_add_query(struct query *query_base, struct query *other_base, int64_t weight)
{
    struct query_sum_edges *query = QUERY_SUM_EDGES(query_base);
    struct query_sum_edges *other = QUERY_SUM_EDGES(other_base);
    return graph_add_graph(query->result, other->result, weight);
}

static int query_sum_edges_finalize(struct query *query_base)
{
    struct query_sum_edges *query = QUERY_SUM_EDGES(query_base);

    if (!graph_del_small(query->result, query->eps))
        return 0;

    query->result->flags |= TVG_FLAGS_READONLY;  /* block changes */
    query->base.cache = sizeof(*query) + graph_memory_usage(query->result);
    return 1;
}

const struct query_ops query_sum_edges_ops =
{
    query_sum_edges_grab,
    query_sum_edges_free,

    query_sum_edges_compatible,
    query_sum_edges_add_graph,
    query_sum_edges_add_query,
    query_sum_edges_finalize,
};

struct graph *tvg_sum_edges(struct tvg *tvg, uint64_t ts_min, uint64_t ts_max, float eps)
{
    struct query_sum_edges *query;
    uint32_t graph_flags;

    if (ts_max < ts_min)
        return NULL;

    if (!(query = malloc(sizeof(*query))))
        return NULL;

    query_init(&query->base, &query_sum_edges_ops, ts_min, ts_max);

    query->eps = eps;

    graph_flags = tvg->flags & (TVG_FLAGS_POSITIVE | TVG_FLAGS_DIRECTED);
    if (!(query->result = alloc_graph(graph_flags)))
    {
        free(query);
        return NULL;
    }

    query->result->query = &query->base;

    return (struct graph *)query_compute(tvg, &query->base);
}

/* query_sum_nodes functions */

const struct query_ops query_sum_nodes_ops;

static inline struct query_sum_nodes *QUERY_SUM_NODES(struct query *query_base)
{
    assert(query_base->ops == &query_sum_nodes_ops);
    return CONTAINING_RECORD(query_base, struct query_sum_nodes, base);
}

static void *query_sum_nodes_grab(struct query *query_base)
{
    struct query_sum_nodes *query = QUERY_SUM_NODES(query_base);
    return grab_vector(query->result);
}

static void query_sum_nodes_free(struct query *query_base)
{
    struct query_sum_nodes *query = QUERY_SUM_NODES(query_base);
    free_vector(query->result);
}

static int query_sum_nodes_compatible(struct query *query_base, struct query *other_base)
{
    return (other_base->ops == &query_sum_nodes_ops);
}

static int query_sum_nodes_add_graph(struct query *query_base, struct graph *graph, int64_t weight)
{
    struct query_sum_nodes *query = QUERY_SUM_NODES(query_base);
    struct vector *nodes;
    int ret;

    if (!(nodes = graph_get_nodes(graph)))
        return 0;

    ret = vector_add_vector(query->result, nodes, weight);
    free_vector(nodes);
    return ret;
}

static int query_sum_nodes_add_query(struct query *query_base, struct query *other_base, int64_t weight)
{
    struct query_sum_nodes *query = QUERY_SUM_NODES(query_base);
    struct query_sum_nodes *other = QUERY_SUM_NODES(other_base);
    return vector_add_vector(query->result, other->result, weight);
}

static int query_sum_nodes_finalize(struct query *query_base)
{
    struct query_sum_nodes *query = QUERY_SUM_NODES(query_base);

    if (!vector_del_small(query->result, 0.5))
        return 0;

    query->result->flags |= TVG_FLAGS_READONLY;  /* block changes */
    query->base.cache = sizeof(*query) + vector_memory_usage(query->result);
    return 1;
}

const struct query_ops query_sum_nodes_ops =
{
    query_sum_nodes_grab,
    query_sum_nodes_free,

    query_sum_nodes_compatible,
    query_sum_nodes_add_graph,
    query_sum_nodes_add_query,
    query_sum_nodes_finalize,
};

struct vector *tvg_sum_nodes(struct tvg *tvg, uint64_t ts_min, uint64_t ts_max)
{
    struct query_sum_nodes *query;
    uint32_t vector_flags = TVG_FLAGS_POSITIVE;

    if (ts_max < ts_min)
        return NULL;

    if (!(query = malloc(sizeof(*query))))
        return NULL;

    query_init(&query->base, &query_sum_nodes_ops, ts_min, ts_max);

    if (!(query->result = alloc_vector(vector_flags)))
    {
        free(query);
        return NULL;
    }

    query->result->query = &query->base;

    return (struct vector *)query_compute(tvg, &query->base);
}

/* query_sum_edges_exp functions */

const struct query_ops query_sum_edges_exp_ops;

static inline struct query_sum_edges_exp *QUERY_SUM_EDGES_EXP(struct query *query_base)
{
    assert(query_base->ops == &query_sum_edges_exp_ops);
    return CONTAINING_RECORD(query_base, struct query_sum_edges_exp, base);
}

static void *query_sum_edges_exp_grab(struct query *query_base)
{
    struct query_sum_edges_exp *query = QUERY_SUM_EDGES_EXP(query_base);
    return grab_graph(query->result);
}

static void query_sum_edges_exp_free(struct query *query_base)
{
    struct query_sum_edges_exp *query = QUERY_SUM_EDGES_EXP(query_base);
    free_graph(query->result);
}

static int query_sum_edges_exp_compatible(struct query *query_base, struct query *other_base)
{
    struct query_sum_edges_exp *query = QUERY_SUM_EDGES_EXP(query_base);
    struct query_sum_edges_exp *other;
    float weight;

    if (other_base->ops != &query_sum_edges_exp_ops)
        return 0;

    other = QUERY_SUM_EDGES_EXP(other_base);

    if (query->weight != other->weight)
        return 0;
    if (query->log_beta != other->log_beta)
        return 0;
    if (query->eps != other->eps)
        return 0;

    weight = (float)exp(query->log_beta * sub_uint64(query->base.ts_max, other->base.ts_max));

    /* If query->base.ts_max << other->base.ts_max the update formula
     * is not numerically stable. Force recomputation if errors could
     * get too large. */

    return (weight <= 1000.0);  /* FIXME: Arbitrary limit. */
}

static int query_sum_edges_exp_add_graph(struct query *query_base, struct graph *graph, int64_t weight)
{
    struct query_sum_edges_exp *query = QUERY_SUM_EDGES_EXP(query_base);
    float total_weight = weight * query->weight * (float)exp(query->log_beta * sub_uint64(query->base.ts_max, graph->ts));
    return graph_add_graph(query->result, graph, total_weight);
}

static int query_sum_edges_exp_add_query(struct query *query_base, struct query *other_base, int64_t weight)
{
    struct query_sum_edges_exp *query = QUERY_SUM_EDGES_EXP(query_base);
    struct query_sum_edges_exp *other = QUERY_SUM_EDGES_EXP(other_base);
    float total_weight = weight * (float)exp(query->log_beta * sub_uint64(query->base.ts_max, other->base.ts_max));
    return graph_add_graph(query->result, other->result, total_weight);
}

static int query_sum_edges_exp_finalize(struct query *query_base)
{
    struct query_sum_edges_exp *query = QUERY_SUM_EDGES_EXP(query_base);

    if (!graph_del_small(query->result, query->eps))
        return 0;

    query->result->flags |= TVG_FLAGS_READONLY;  /* block changes */
    query->base.cache = sizeof(*query) + graph_memory_usage(query->result);
    return 1;
}

const struct query_ops query_sum_edges_exp_ops =
{
    query_sum_edges_exp_grab,
    query_sum_edges_exp_free,

    query_sum_edges_exp_compatible,
    query_sum_edges_exp_add_graph,
    query_sum_edges_exp_add_query,
    query_sum_edges_exp_finalize,
};

struct graph *tvg_sum_edges_exp(struct tvg *tvg, uint64_t ts_min, uint64_t ts_max,
                                float weight, float log_beta, float eps)
{
    struct query_sum_edges_exp *query;
    uint32_t graph_flags;

    if (ts_max < ts_min)
        return NULL;

    if (!(query = malloc(sizeof(*query))))
        return NULL;

    query_init(&query->base, &query_sum_edges_exp_ops, ts_min, ts_max);

    query->weight      = weight;
    query->log_beta    = log_beta;
    query->eps         = eps;

    graph_flags = tvg->flags & (TVG_FLAGS_POSITIVE | TVG_FLAGS_DIRECTED);
    if (!(query->result = alloc_graph(graph_flags)))
    {
        free(query);
        return NULL;
    }

    query->result->query = &query->base;

    return (struct graph *)query_compute(tvg, &query->base);
}

/* query_count_edges functions */

const struct query_ops query_count_edges_ops;

static inline struct query_count_edges *QUERY_COUNT_EDGES(struct query *query_base)
{
    assert(query_base->ops == &query_count_edges_ops);
    return CONTAINING_RECORD(query_base, struct query_count_edges, base);
}

static void *query_count_edges_grab(struct query *query_base)
{
    struct query_count_edges *query = QUERY_COUNT_EDGES(query_base);
    return grab_graph(query->result);
}

static void query_count_edges_free(struct query *query_base)
{
    struct query_count_edges *query = QUERY_COUNT_EDGES(query_base);
    free_graph(query->result);
}

static int query_count_edges_compatible(struct query *query_base, struct query *other_base)
{
    return (other_base->ops == &query_count_edges_ops);
}

static int query_count_edges_add_graph(struct query *query_base, struct graph *graph, int64_t weight)
{
    struct query_count_edges *query = QUERY_COUNT_EDGES(query_base);
    return graph_add_count_edges(query->result, graph, weight);
}

static int query_count_edges_add_query(struct query *query_base, struct query *other_base, int64_t weight)
{
    struct query_count_edges *query = QUERY_COUNT_EDGES(query_base);
    struct query_count_edges *other = QUERY_COUNT_EDGES(other_base);
    return graph_add_graph(query->result, other->result, weight);
}

static int query_count_edges_finalize(struct query *query_base)
{
    struct query_count_edges *query = QUERY_COUNT_EDGES(query_base);

    if (!graph_del_small(query->result, 0.5))
        return 0;

    query->result->flags |= TVG_FLAGS_READONLY;  /* block changes */
    query->base.cache = sizeof(*query) + graph_memory_usage(query->result);
    return 1;
}

const struct query_ops query_count_edges_ops =
{
    query_count_edges_grab,
    query_count_edges_free,

    query_count_edges_compatible,
    query_count_edges_add_graph,
    query_count_edges_add_query,
    query_count_edges_finalize,
};

struct graph *tvg_count_edges(struct tvg *tvg, uint64_t ts_min, uint64_t ts_max)
{
    struct query_count_edges *query;
    uint32_t graph_flags;

    if (ts_max < ts_min)
        return NULL;

    if (!(query = malloc(sizeof(*query))))
        return NULL;

    query_init(&query->base, &query_count_edges_ops, ts_min, ts_max);

    /* Enforce TVG_FLAGS_POSITIVE, our update mechanism relies on it. */
    graph_flags = tvg->flags & TVG_FLAGS_DIRECTED;
    graph_flags |= TVG_FLAGS_POSITIVE;

    if (!(query->result = alloc_graph(graph_flags)))
    {
        free(query);
        return NULL;
    }

    query->result->query = &query->base;

    return (struct graph *)query_compute(tvg, &query->base);
}

/* query_count_nodes functions */

const struct query_ops query_count_nodes_ops;

static inline struct query_count_nodes *QUERY_COUNT_NODES(struct query *query_base)
{
    assert(query_base->ops == &query_count_nodes_ops);
    return CONTAINING_RECORD(query_base, struct query_count_nodes, base);
}

static void *query_count_nodes_grab(struct query *query_base)
{
    struct query_count_nodes *query = QUERY_COUNT_NODES(query_base);
    return grab_vector(query->result);
}

static void query_count_nodes_free(struct query *query_base)
{
    struct query_count_nodes *query = QUERY_COUNT_NODES(query_base);
    free_vector(query->result);
}

static int query_count_nodes_compatible(struct query *query_base, struct query *other_base)
{
    return (other_base->ops == &query_count_nodes_ops);
}

static int query_count_nodes_add_graph(struct query *query_base, struct graph *graph, int64_t weight)
{
    struct query_count_nodes *query = QUERY_COUNT_NODES(query_base);
    return vector_add_count_nodes(query->result, graph, weight);
}

static int query_count_nodes_add_query(struct query *query_base, struct query *other_base, int64_t weight)
{
    struct query_count_nodes *query = QUERY_COUNT_NODES(query_base);
    struct query_count_nodes *other = QUERY_COUNT_NODES(other_base);
    return vector_add_vector(query->result, other->result, weight);
}

static int query_count_nodes_finalize(struct query *query_base)
{
    struct query_count_nodes *query = QUERY_COUNT_NODES(query_base);

    if (!vector_del_small(query->result, 0.5))
        return 0;

    query->result->flags |= TVG_FLAGS_READONLY;  /* block changes */
    query->base.cache = sizeof(*query) + vector_memory_usage(query->result);
    return 1;
}

const struct query_ops query_count_nodes_ops =
{
    query_count_nodes_grab,
    query_count_nodes_free,

    query_count_nodes_compatible,
    query_count_nodes_add_graph,
    query_count_nodes_add_query,
    query_count_nodes_finalize,
};

struct vector *tvg_count_nodes(struct tvg *tvg, uint64_t ts_min, uint64_t ts_max)
{
    struct query_count_nodes *query;
    uint32_t vector_flags = TVG_FLAGS_POSITIVE;

    if (ts_max < ts_min)
        return NULL;

    if (!(query = malloc(sizeof(*query))))
        return NULL;

    query_init(&query->base, &query_count_nodes_ops, ts_min, ts_max);

    if (!(query->result = alloc_vector(vector_flags)))
    {
        free(query);
        return NULL;
    }

    query->result->query = &query->base;

    return (struct vector *)query_compute(tvg, &query->base);
}

/* query_count_graphs functions */

const struct query_ops query_count_graphs_ops;

static inline struct query_count_graphs *QUERY_COUNT_GRAPHS(struct query *query_base)
{
    assert(query_base->ops == &query_count_graphs_ops);
    return CONTAINING_RECORD(query_base, struct query_count_graphs, base);
}

static void *query_count_graphs_grab(struct query *query_base)
{
    struct query_count_graphs *query = QUERY_COUNT_GRAPHS(query_base);
    __sync_fetch_and_add(&query->refcount, 1);
    return query;
}

static void query_count_graphs_free(struct query *query_base)
{
    struct query_count_graphs *query = QUERY_COUNT_GRAPHS(query_base);
    if (__sync_sub_and_fetch(&query->refcount, 1)) return;
    free_query(query_base);
}

static int query_count_graphs_compatible(struct query *query_base, struct query *other_base)
{
    return (other_base->ops == &query_count_graphs_ops);
}

static int query_count_graphs_add_graph(struct query *query_base, struct graph *graph, int64_t weight)
{
    struct query_count_graphs *query = QUERY_COUNT_GRAPHS(query_base);
    query->result += weight;
    return 1;
}

static int query_count_graphs_add_query(struct query *query_base, struct query *other_base, int64_t weight)
{
    struct query_count_graphs *query = QUERY_COUNT_GRAPHS(query_base);
    struct query_count_graphs *other = QUERY_COUNT_GRAPHS(other_base);
    query->result += other->result * weight;
    return 1;
}

static int query_count_graphs_finalize(struct query *query_base)
{
    struct query_count_graphs *query = QUERY_COUNT_GRAPHS(query_base);
    query->base.cache = sizeof(*query);
    return 1;
}

const struct query_ops query_count_graphs_ops =
{
    query_count_graphs_grab,
    query_count_graphs_free,

    query_count_graphs_compatible,
    query_count_graphs_add_graph,
    query_count_graphs_add_query,
    query_count_graphs_finalize,
};

uint64_t tvg_count_graphs(struct tvg *tvg, uint64_t ts_min, uint64_t ts_max)
{
    struct query_count_graphs *query;
    uint64_t result;

    if (ts_max < ts_min)
        return ~0ULL;

    if (!(query = malloc(sizeof(*query))))
        return ~0ULL;

    query_init(&query->base, &query_count_graphs_ops, ts_min, ts_max);

    query->refcount = 1;
    query->result   = 0;

    if (!(query = (struct query_count_graphs *)query_compute(tvg, &query->base)))
        return ~0ULL;

    result = query->result;
    query->base.ops->free(&query->base);
    return result;
}

/* query_topics functions */

struct graph *tvg_topics(struct tvg *tvg, uint64_t ts_min, uint64_t ts_max, uint64_t step, uint64_t offset)
{
    uint64_t ts_snap_min, ts_snap_max;
    struct vector *count_nodes = NULL;
    struct graph **snapshots = NULL;
    struct graph *count_edges = NULL;
    struct graph *sum_edges = NULL;
    struct graph *result = NULL;
    uint64_t num_snapshots = 0;
    uint32_t graph_flags;
    struct entry2 *edge;
    static int warned = 0;
    uint64_t count;
    float weight;
    uint64_t i;

    if (ts_max < ts_min)
        return NULL;

    /* Warn user when trying to use topic metric with sum_weights=True. In the
     * original publication, the metric is only defined for sum_weights=False. */
    if (tvg->mongodb && tvg->mongodb->config->sum_weights && !warned)
    {
        fprintf(stderr, "%s: The topic metric is only defined for sum_weights=False.\n", __func__);
        warned = 1;
    }

    if (step)
    {
        offset -= (offset / step) * step;

        if (ts_max < offset)
            num_snapshots = 1;
        else if (ts_min < offset)
            num_snapshots = (ts_max - offset) / step + 2;
        else
            num_snapshots = (ts_max - offset) / step -
                            (ts_min - offset) / step + 1;

        assert(num_snapshots > 0);
        if (!(snapshots = malloc(sizeof(*snapshots) * num_snapshots)))
            goto error;

        ts_snap_min = ts_min;

        for (i = 0; i < num_snapshots; i++)
        {
            if (i == num_snapshots - 1)
                ts_snap_max = ts_max;
            else if (ts_snap_min >= offset)
                ts_snap_max = ts_snap_min + step - 1 - ((ts_snap_min - offset) % step);
            else
                ts_snap_max = offset - 1;

            if (!(snapshots[i] = tvg_count_edges(tvg, ts_snap_min, ts_snap_max)))
            {
                for (; i < num_snapshots; i++) snapshots[i] = NULL;
                goto error;
            }

            ts_snap_min = ts_snap_max + 1;
        }

        assert(ts_snap_max == ts_max);
    }

    if (!(sum_edges = tvg_sum_edges(tvg, ts_min, ts_max, 0.0)))
        goto error;
    if (!(count_edges = tvg_count_edges(tvg, ts_min, ts_max)))
        goto error;
    if (!(count_nodes = tvg_count_nodes(tvg, ts_min, ts_max)))
        goto error;

    /* Enforce TVG_FLAGS_POSITIVE, our update mechanism relies on it. */
    graph_flags = tvg->flags & TVG_FLAGS_DIRECTED;
    graph_flags |= TVG_FLAGS_POSITIVE;

    if (!(result = alloc_graph(graph_flags)))
        goto error;

    /* The metric uses a combination of the following terms:
     *
     * Article jaccard weight: |D(v1) \cup D(v2)| / |D(e)|
     * Temporal coverage density: \Delta T / |T(e)|
     * Min distance per article weight: |L(e)| / \sum exp(-\delta)
     *
     * Note: In the original publication, the metric is only defined
     * for sum_weights=False. As a result |D(e)| = |L(e)|, and
     * \sum exp(-\delta) is always smaller or equal to |L(e)|. In the
     * following we also allow sum_weights=True, but since we don't
     * know how many weights contributed to each edge, the metric
     * is no longer a value in [0, 1], but instead in [0, 1.5].
     */

    GRAPH_FOR_EACH_EDGE(count_edges, edge)
    {
        weight  = (vector_get_entry(count_nodes, edge->source) +
                   vector_get_entry(count_nodes, edge->target) - edge->weight) / edge->weight;
        weight += edge->weight / graph_get_edge(sum_edges, edge->source, edge->target);

        if (snapshots)
        {
            count = 0;
            for (i = 0; i < num_snapshots; i++)
                count += graph_has_edge(snapshots[i], edge->source, edge->target);
            weight += (float)num_snapshots / count;

            weight = 3.0 / weight;
        }
        else
        {
            weight = 2.0 / weight;
        }

        if (!graph_set_edge(result, edge->source, edge->target, weight))
        {
            free_graph(result);
            result = NULL;
            goto error;
        }
    }

    /* Note: No need to make the object read-only since we do not hold any references. */

error:
    if (snapshots)
    {
        for (i = 0; i < num_snapshots; i++)
            free_graph(snapshots[i]);
        free(snapshots);
    }
    free_graph(sum_edges);
    free_graph(count_edges);
    free_vector(count_nodes);
    return result;
}

/* generic functions */

/* called via free_vector() or free_graph() */
void free_query(struct query *query)
{
    if (!query) return;

    if (query->tvg)
    {
        list_remove(&query->entry);
        query->tvg = NULL;
    }

    assert(!query->cache);
    free(query);
}

void unlink_query(struct query *query, int invalidate)
{
    struct tvg *tvg;
    int cache;

    if (!query || !(tvg = query->tvg))
        return;

    if ((cache = (query->cache != 0)))
    {
        list_remove(&query->cache_entry);
        tvg->query_cache_used -= query->cache;
        query->cache = 0;
    }

    if (invalidate)
    {
        list_remove(&query->entry);
        query->tvg = NULL;
    }

    if (cache)
        query->ops->free(query);
}
