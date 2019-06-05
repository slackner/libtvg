/*
 * Time-varying graph library
 * Metric functions.
 *
 * Copyright (c) 2019 Sebastian Lackner
 */

#include "tvg.h"
#include "internal.h"

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

static int graph_add_count_nodes(struct vector *out, struct graph *graph, float weight)
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

/* metric_sum_edges functions */

const struct metric_ops metric_sum_edges_ops;

static inline struct metric_sum_edges *METRIC_SUM_EDGES(struct metric *metric)
{
    assert(metric->ops == &metric_sum_edges_ops);
    return CONTAINING_RECORD(metric, struct metric_sum_edges, metric);
}

static int metric_sum_edges_init(struct metric *metric)
{
    struct metric_sum_edges *metric_sum_edges = METRIC_SUM_EDGES(metric);
    struct tvg *tvg = metric->window->tvg;
    uint32_t graph_flags;

    if (metric_sum_edges->result)
        return 1;

    /* Enforce TVG_FLAGS_NONZERO, our update mechanism relies on it. */
    graph_flags = tvg->flags & (TVG_FLAGS_POSITIVE | TVG_FLAGS_DIRECTED);
    graph_flags |= TVG_FLAGS_NONZERO;

    if (!(metric_sum_edges->result = alloc_graph(graph_flags)))
        return 0;

    graph_set_eps(metric_sum_edges->result, metric_sum_edges->eps);
    return 1;
}

static void metric_sum_edges_free(struct metric *metric)
{
    struct metric_sum_edges *metric_sum_edges = METRIC_SUM_EDGES(metric);
    free_graph(metric_sum_edges->result);
    metric_sum_edges->result = NULL;
}

static int metric_sum_edges_valid(struct metric *metric)
{
    struct metric_sum_edges *metric_sum_edges = METRIC_SUM_EDGES(metric);
    return (metric_sum_edges->result != NULL);
}

static int metric_sum_edges_clear(struct metric *metric)
{
    struct metric_sum_edges *metric_sum_edges = METRIC_SUM_EDGES(metric);

    if (metric_sum_edges->result && graph_empty(metric_sum_edges->result))
        return 1;

    metric_sum_edges_free(metric);
    return metric_sum_edges_init(metric);
}

static int metric_sum_edges_add(struct metric *metric, struct graph *graph)
{
    return graph_add_graph(METRIC_SUM_EDGES(metric)->result, graph, 1.0);
}

static int metric_sum_edges_sub(struct metric *metric, struct graph *graph)
{
    return graph_sub_graph(METRIC_SUM_EDGES(metric)->result, graph, 1.0);
}

static int metric_sum_edges_move(struct metric *metric, uint64_t ts)
{
    /* nothing to do */
    return 1;
}

const struct metric_ops metric_sum_edges_ops =
{
    metric_sum_edges_init,
    metric_sum_edges_free,
    metric_sum_edges_valid,
    metric_sum_edges_clear,
    metric_sum_edges_add,
    metric_sum_edges_sub,
    metric_sum_edges_move,
};

struct metric *window_alloc_metric_sum_edges(struct window *window, float eps)
{
    struct metric_sum_edges *metric_sum_edges;
    struct metric *metric;

    LIST_FOR_EACH(metric, &window->metrics, struct metric, entry)
    {
        if (metric->ops != &metric_sum_edges_ops) continue;
        metric_sum_edges = METRIC_SUM_EDGES(metric);
        if (metric_sum_edges->eps == eps)
            return grab_metric(metric);
    }

    if (!(metric_sum_edges = malloc(sizeof(*metric_sum_edges))))
        return NULL;

    metric = &metric_sum_edges->metric;
    metric->refcount = 1;
    metric->window   = grab_window(window);
    metric->ops      = &metric_sum_edges_ops;

    metric_sum_edges->result = NULL;
    metric_sum_edges->eps    = eps;

    list_add_tail(&window->metrics, &metric->entry);
    return metric;
}

float metric_sum_edges_get_eps(struct metric *metric)
{
    return METRIC_SUM_EDGES(metric)->eps;
}

struct graph *metric_sum_edges_get_result(struct metric *metric)
{
    return grab_graph(METRIC_SUM_EDGES(metric)->result);
}

/* metric_decay functions */

const struct metric_ops metric_decay_ops;

static inline struct metric_decay *METRIC_DECAY(struct metric *metric)
{
    assert(metric->ops == &metric_decay_ops);
    return CONTAINING_RECORD(metric, struct metric_decay, metric);
}

static int metric_decay_init(struct metric *metric)
{
    struct metric_decay *metric_decay = METRIC_DECAY(metric);
    struct tvg *tvg = metric->window->tvg;
    uint32_t graph_flags;

    if (metric_decay->result)
        return 1;

    /* Enforce TVG_FLAGS_NONZERO, our update mechanism relies on it. */
    graph_flags = tvg->flags & (TVG_FLAGS_POSITIVE | TVG_FLAGS_DIRECTED);
    graph_flags |= TVG_FLAGS_NONZERO;

    if (!(metric_decay->result = alloc_graph(graph_flags)))
        return 0;

    graph_set_eps(metric_decay->result, metric_decay->eps);
    return 1;
}

static void metric_decay_free(struct metric *metric)
{
    struct metric_decay *metric_decay = METRIC_DECAY(metric);
    free_graph(metric_decay->result);
    metric_decay->result = NULL;
}

static int metric_decay_valid(struct metric *metric)
{
    struct metric_decay *metric_decay = METRIC_DECAY(metric);
    return (metric_decay->result != NULL);
}

static int metric_decay_clear(struct metric *metric)
{
    struct metric_decay *metric_decay = METRIC_DECAY(metric);

    if (metric_decay->result && graph_empty(metric_decay->result))
        return 1;

    metric_decay_free(metric);
    return metric_decay_init(metric);
}

static int metric_decay_add(struct metric *metric, struct graph *graph)
{
    struct metric_decay *metric_decay = METRIC_DECAY(metric);
    float weight = (float)exp(metric_decay->log_beta * sub_uint64(metric->window->ts, graph->ts));
    return graph_add_graph(metric_decay->result, graph, weight);
}

static int metric_decay_sub(struct metric *metric, struct graph *graph)
{
    struct metric_decay *metric_decay = METRIC_DECAY(metric);
    float weight = (float)exp(metric_decay->log_beta * sub_uint64(metric->window->ts, graph->ts));
    return graph_sub_graph(metric_decay->result, graph, weight);
}

static int metric_decay_move(struct metric *metric, uint64_t ts)
{
    struct metric_decay *metric_decay = METRIC_DECAY(metric);
    float weight = (float)exp(metric_decay->log_beta * sub_uint64(ts, metric->window->ts));

    /* If ts << window->ts the update formula is not numerically stable:
     * \delta w_new ~ \delta w_old * pow(window->beta, ts - window->ts).
     * Force recomputation if errors could get too large. */

    if (weight >= 1000.0)  /* FIXME: Arbitrary limit. */
        return 0;

    graph_mul_const(metric_decay->result, weight);
    return 1;
}

const struct metric_ops metric_decay_ops =
{
    metric_decay_init,
    metric_decay_free,
    metric_decay_valid,
    metric_decay_clear,
    metric_decay_add,
    metric_decay_sub,
    metric_decay_move,
};

struct metric *window_alloc_metric_decay(struct window *window, float log_beta, float eps)
{
    struct metric_decay *metric_decay;
    struct metric *metric;

    LIST_FOR_EACH(metric, &window->metrics, struct metric, entry)
    {
        if (metric->ops != &metric_decay_ops) continue;
        metric_decay = METRIC_DECAY(metric);
        if (metric_decay->log_beta == log_beta && metric_decay->eps == eps)
            return grab_metric(metric);
    }

    if (!(metric_decay = malloc(sizeof(*metric_decay))))
        return NULL;

    metric = &metric_decay->metric;
    metric->refcount = 1;
    metric->window   = grab_window(window);
    metric->ops      = &metric_decay_ops;

    metric_decay->result    = NULL;
    metric_decay->log_beta  = log_beta;
    metric_decay->eps       = eps;

    list_add_tail(&window->metrics, &metric->entry);
    return metric;
}

float metric_decay_get_eps(struct metric *metric)
{
    return METRIC_DECAY(metric)->eps;
}

struct graph *metric_decay_get_result(struct metric *metric)
{
    return grab_graph(METRIC_DECAY(metric)->result);
}

/* metric_smooth functions */

const struct metric_ops metric_smooth_ops;

static inline struct metric_smooth *METRIC_SMOOTH(struct metric *metric)
{
    assert(metric->ops == &metric_smooth_ops);
    return CONTAINING_RECORD(metric, struct metric_smooth, metric);
}

static int metric_smooth_init(struct metric *metric)
{
    struct metric_smooth *metric_smooth = METRIC_SMOOTH(metric);
    struct tvg *tvg = metric->window->tvg;
    uint32_t graph_flags;

    if (metric_smooth->result)
        return 1;

    /* Enforce TVG_FLAGS_NONZERO, our update mechanism relies on it. */
    graph_flags = tvg->flags & (TVG_FLAGS_POSITIVE | TVG_FLAGS_DIRECTED);
    graph_flags |= TVG_FLAGS_NONZERO;

    if (!(metric_smooth->result = alloc_graph(graph_flags)))
        return 0;

    graph_set_eps(metric_smooth->result, metric_smooth->eps);
    return 1;
}

static void metric_smooth_free(struct metric *metric)
{
    struct metric_smooth *metric_smooth = METRIC_SMOOTH(metric);
    free_graph(metric_smooth->result);
    metric_smooth->result = NULL;
}

static int metric_smooth_valid(struct metric *metric)
{
    struct metric_smooth *metric_smooth = METRIC_SMOOTH(metric);
    return (metric_smooth->result != NULL);
}

static int metric_smooth_clear(struct metric *metric)
{
    struct metric_smooth *metric_smooth = METRIC_SMOOTH(metric);

    if (metric_smooth->result && graph_empty(metric_smooth->result))
        return 1;

    metric_smooth_free(metric);
    return metric_smooth_init(metric);
}

static int metric_smooth_add(struct metric *metric, struct graph *graph)
{
    struct metric_smooth *metric_smooth = METRIC_SMOOTH(metric);
    float weight = metric_smooth->weight * (float)exp(metric_smooth->log_beta *
        sub_uint64(metric->window->ts, graph->ts));
    return graph_add_graph(metric_smooth->result, graph, weight);
}

static int metric_smooth_sub(struct metric *metric, struct graph *graph)
{
    struct metric_smooth *metric_smooth = METRIC_SMOOTH(metric);
    float weight = metric_smooth->weight * (float)exp(metric_smooth->log_beta *
        sub_uint64(metric->window->ts, graph->ts));
    return graph_sub_graph(metric_smooth->result, graph, weight);
}

static int metric_smooth_move(struct metric *metric, uint64_t ts)
{
    struct metric_smooth *metric_smooth = METRIC_SMOOTH(metric);
    float weight = (float)exp(metric_smooth->log_beta * sub_uint64(ts, metric->window->ts));

    if (weight >= 1000.0)  /* FIXME: Arbitrary limit. */
        return 0;

    graph_mul_const(metric_smooth->result, weight);
    return 1;
}

const struct metric_ops metric_smooth_ops =
{
    metric_smooth_init,
    metric_smooth_free,
    metric_smooth_valid,
    metric_smooth_clear,
    metric_smooth_add,
    metric_smooth_sub,
    metric_smooth_move,
};

struct metric *window_alloc_metric_smooth(struct window *window, float log_beta, float eps)
{
    struct metric_smooth *metric_smooth;
    struct metric *metric;

    LIST_FOR_EACH(metric, &window->metrics, struct metric, entry)
    {
        if (metric->ops != &metric_smooth_ops) continue;
        metric_smooth = METRIC_SMOOTH(metric);
        if (metric_smooth->log_beta == log_beta && metric_smooth->eps == eps)
            return grab_metric(metric);
    }

    if (!(metric_smooth = malloc(sizeof(*metric_smooth))))
        return NULL;

    metric = &metric_smooth->metric;
    metric->refcount = 1;
    metric->window   = grab_window(window);
    metric->ops      = &metric_smooth_ops;

    metric_smooth->result   = NULL;
    metric_smooth->weight   = -(float)expm1(log_beta);
    metric_smooth->log_beta = log_beta;
    metric_smooth->eps      = eps;

    list_add_tail(&window->metrics, &metric->entry);
    return metric;
}

float metric_smooth_get_eps(struct metric *metric)
{
    return METRIC_SMOOTH(metric)->eps;
}

struct graph *metric_smooth_get_result(struct metric *metric)
{
    return grab_graph(METRIC_SMOOTH(metric)->result);
}

/* metric_count_edges functions */

const struct metric_ops metric_count_edges_ops;

static inline struct metric_count_edges *METRIC_COUNT_EDGES(struct metric *metric)
{
    assert(metric->ops == &metric_count_edges_ops);
    return CONTAINING_RECORD(metric, struct metric_count_edges, metric);
}

static int metric_count_edges_init(struct metric *metric)
{
    struct metric_count_edges *metric_count_edges = METRIC_COUNT_EDGES(metric);
    struct tvg *tvg = metric->window->tvg;
    uint32_t graph_flags;

    if (metric_count_edges->result)
        return 1;

    /* Enforce TVG_FLAGS_POSITIVE and TVG_FLAGS_NONZERO, our update mechanism relies on it. */
    graph_flags = tvg->flags & TVG_FLAGS_DIRECTED;
    graph_flags |= TVG_FLAGS_POSITIVE | TVG_FLAGS_NONZERO;

    if (!(metric_count_edges->result = alloc_graph(graph_flags)))
        return 0;

    graph_set_eps(metric_count_edges->result, 0.5);
    return 1;
}

static void metric_count_edges_free(struct metric *metric)
{
    struct metric_count_edges *metric_count_edges = METRIC_COUNT_EDGES(metric);
    free_graph(metric_count_edges->result);
    metric_count_edges->result = NULL;
}

static int metric_count_edges_valid(struct metric *metric)
{
    struct metric_count_edges *metric_count_edges = METRIC_COUNT_EDGES(metric);
    return (metric_count_edges->result != NULL);
}

static int metric_count_edges_clear(struct metric *metric)
{
    struct metric_count_edges *metric_count_edges = METRIC_COUNT_EDGES(metric);

    if (metric_count_edges->result && graph_empty(metric_count_edges->result))
        return 1;

    metric_count_edges_free(metric);
    return metric_count_edges_init(metric);
}

static int metric_count_edges_add(struct metric *metric, struct graph *graph)
{
    return graph_add_count_edges(METRIC_COUNT_EDGES(metric)->result, graph, 1.0);
}

static int metric_count_edges_sub(struct metric *metric, struct graph *graph)
{
    return graph_add_count_edges(METRIC_COUNT_EDGES(metric)->result, graph, -1.0);
}

static int metric_count_edges_move(struct metric *metric, uint64_t ts)
{
    /* nothing to do */
    return 1;
}

const struct metric_ops metric_count_edges_ops =
{
    metric_count_edges_init,
    metric_count_edges_free,
    metric_count_edges_valid,
    metric_count_edges_clear,
    metric_count_edges_add,
    metric_count_edges_sub,
    metric_count_edges_move,
};

struct metric *window_alloc_metric_count_edges(struct window *window)
{
    struct metric_count_edges *metric_count_edges;
    struct metric *metric;

    LIST_FOR_EACH(metric, &window->metrics, struct metric, entry)
    {
        if (metric->ops == &metric_count_edges_ops)
            return grab_metric(metric);
    }

    if (!(metric_count_edges = malloc(sizeof(*metric_count_edges))))
        return NULL;

    metric = &metric_count_edges->metric;
    metric->refcount = 1;
    metric->window   = grab_window(window);
    metric->ops      = &metric_count_edges_ops;

    metric_count_edges->result = NULL;

    list_add_tail(&window->metrics, &metric->entry);
    return metric;
}

struct graph *metric_count_edges_get_result(struct metric *metric)
{
    return grab_graph(METRIC_COUNT_EDGES(metric)->result);
}

/* metric_count_nodes functions */

const struct metric_ops metric_count_nodes_ops;

static inline struct metric_count_nodes *METRIC_COUNT_NODES(struct metric *metric)
{
    assert(metric->ops == &metric_count_nodes_ops);
    return CONTAINING_RECORD(metric, struct metric_count_nodes, metric);
}

static int metric_count_nodes_init(struct metric *metric)
{
    struct metric_count_nodes *metric_count_nodes = METRIC_COUNT_NODES(metric);
    uint32_t vector_flags = TVG_FLAGS_POSITIVE | TVG_FLAGS_NONZERO;

    if (metric_count_nodes->result)
        return 1;

    if (!(metric_count_nodes->result = alloc_vector(vector_flags)))
        return 0;

    vector_set_eps(metric_count_nodes->result, 0.5);
    return 1;
}

static void metric_count_nodes_free(struct metric *metric)
{
    struct metric_count_nodes *metric_count_nodes = METRIC_COUNT_NODES(metric);
    free_vector(metric_count_nodes->result);
    metric_count_nodes->result = NULL;
}

static int metric_count_nodes_valid(struct metric *metric)
{
    struct metric_count_nodes *metric_count_nodes = METRIC_COUNT_NODES(metric);
    return (metric_count_nodes->result != NULL);
}

static int metric_count_nodes_clear(struct metric *metric)
{
    struct metric_count_nodes *metric_count_nodes = METRIC_COUNT_NODES(metric);

    if (metric_count_nodes->result && vector_empty(metric_count_nodes->result))
        return 1;

    metric_count_nodes_free(metric);
    return metric_count_nodes_init(metric);
}

static int metric_count_nodes_add(struct metric *metric, struct graph *graph)
{
    return graph_add_count_nodes(METRIC_COUNT_NODES(metric)->result, graph, 1.0);
}

static int metric_count_nodes_sub(struct metric *metric, struct graph *graph)
{
    return graph_add_count_nodes(METRIC_COUNT_NODES(metric)->result, graph, -1.0);
}

static int metric_count_nodes_move(struct metric *metric, uint64_t ts)
{
    /* nothing to do */
    return 1;
}

const struct metric_ops metric_count_nodes_ops =
{
    metric_count_nodes_init,
    metric_count_nodes_free,
    metric_count_nodes_valid,
    metric_count_nodes_clear,
    metric_count_nodes_add,
    metric_count_nodes_sub,
    metric_count_nodes_move,
};

struct metric *window_alloc_metric_count_nodes(struct window *window)
{
    struct metric_count_nodes *metric_count_nodes;
    struct metric *metric;

    LIST_FOR_EACH(metric, &window->metrics, struct metric, entry)
    {
        if (metric->ops == &metric_count_nodes_ops)
            return grab_metric(metric);
    }

    if (!(metric_count_nodes = malloc(sizeof(*metric_count_nodes))))
        return NULL;

    metric = &metric_count_nodes->metric;
    metric->refcount = 1;
    metric->window   = grab_window(window);
    metric->ops      = &metric_count_nodes_ops;

    metric_count_nodes->result = NULL;

    list_add_tail(&window->metrics, &metric->entry);
    return metric;
}

struct vector *metric_count_nodes_get_result(struct metric *metric)
{
    return grab_vector(METRIC_COUNT_NODES(metric)->result);
}

/* generic functions */

struct metric *grab_metric(struct metric *metric)
{
    if (metric) __sync_fetch_and_add(&metric->refcount, 1);
    return metric;
}

void free_metric(struct metric *metric)
{
    if (!metric) return;
    if (__sync_sub_and_fetch(&metric->refcount, 1)) return;

    metric->ops->free(metric);
    list_remove(&metric->entry);
    free_window(metric->window);
    free(metric);
}

void metric_reset(struct metric *metric)
{
    metric->ops->free(metric);
}

struct window *metric_get_window(struct metric *metric)
{
    return grab_window(metric->window);
}
