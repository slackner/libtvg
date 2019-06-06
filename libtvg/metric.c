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

static void metric_sum_edges_free(struct metric *metric)
{
    struct metric_sum_edges *metric_sum_edges = METRIC_SUM_EDGES(metric);
    free_graph(metric_sum_edges->result);
}

static void metric_sum_edges_reset(struct metric *metric)
{
    graph_clear(METRIC_SUM_EDGES(metric)->result);
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
    return 1;  /* nothing to do */
}

const struct metric_ops metric_sum_edges_ops =
{
    metric_sum_edges_free,
    metric_sum_edges_reset,
    metric_sum_edges_add,
    metric_sum_edges_sub,
    metric_sum_edges_move,
};

struct metric *window_alloc_metric_sum_edges(struct window *window, float eps)
{
    struct metric_sum_edges *metric_sum_edges;
    struct metric *metric;
    uint32_t graph_flags;

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
    metric->valid    = 0;

    /* Enforce TVG_FLAGS_NONZERO, our update mechanism relies on it. */
    graph_flags = window->tvg->flags & (TVG_FLAGS_POSITIVE | TVG_FLAGS_DIRECTED);
    graph_flags |= TVG_FLAGS_NONZERO;

    if (!(metric_sum_edges->result = alloc_graph(graph_flags)))
    {
        free_window(metric->window);
        free(metric_sum_edges);
        return NULL;
    }

    graph_set_eps(metric_sum_edges->result, eps);
    metric_sum_edges->eps = eps;

    list_add_tail(&window->metrics, &metric->entry);
    return metric;
}

float metric_sum_edges_get_eps(struct metric *metric)
{
    return METRIC_SUM_EDGES(metric)->eps;
}

struct graph *metric_sum_edges_get_result(struct metric *metric)
{
    if (!metric->valid) return NULL;
    return grab_graph(METRIC_SUM_EDGES(metric)->result);
}

/* metric_sum_edges_exp functions */

const struct metric_ops metric_sum_edges_exp_ops;

static inline struct metric_sum_edges_exp *METRIC_SUM_EDGES_EXP(struct metric *metric)
{
    assert(metric->ops == &metric_sum_edges_exp_ops);
    return CONTAINING_RECORD(metric, struct metric_sum_edges_exp, metric);
}

static void metric_sum_edges_exp_free(struct metric *metric)
{
    struct metric_sum_edges_exp *metric_sum_edges_exp = METRIC_SUM_EDGES_EXP(metric);
    free_graph(metric_sum_edges_exp->result);
}

static void metric_sum_edges_exp_reset(struct metric *metric)
{
    graph_clear(METRIC_SUM_EDGES_EXP(metric)->result);
}

static int metric_sum_edges_exp_add(struct metric *metric, struct graph *graph)
{
    struct metric_sum_edges_exp *metric_sum_edges_exp = METRIC_SUM_EDGES_EXP(metric);
    float weight = metric_sum_edges_exp->weight * (float)exp(metric_sum_edges_exp->log_beta *
        sub_uint64(metric->window->ts, graph->ts));
    return graph_add_graph(metric_sum_edges_exp->result, graph, weight);
}

static int metric_sum_edges_exp_sub(struct metric *metric, struct graph *graph)
{
    struct metric_sum_edges_exp *metric_sum_edges_exp = METRIC_SUM_EDGES_EXP(metric);
    float weight = metric_sum_edges_exp->weight * (float)exp(metric_sum_edges_exp->log_beta *
        sub_uint64(metric->window->ts, graph->ts));
    return graph_sub_graph(metric_sum_edges_exp->result, graph, weight);
}

static int metric_sum_edges_exp_move(struct metric *metric, uint64_t ts)
{
    struct metric_sum_edges_exp *metric_sum_edges_exp = METRIC_SUM_EDGES_EXP(metric);
    float weight = (float)exp(metric_sum_edges_exp->log_beta * sub_uint64(ts, metric->window->ts));

    /* If ts << window->ts the update formula is not numerically stable:
     * \delta w_new ~ \delta w_old * pow(window->beta, ts - window->ts).
     * Force recomputation if errors could get too large. */

    if (weight >= 1000.0)  /* FIXME: Arbitrary limit. */
        return 0;

    graph_mul_const(metric_sum_edges_exp->result, weight);
    return 1;
}

const struct metric_ops metric_sum_edges_exp_ops =
{
    metric_sum_edges_exp_free,
    metric_sum_edges_exp_reset,
    metric_sum_edges_exp_add,
    metric_sum_edges_exp_sub,
    metric_sum_edges_exp_move,
};

struct metric *window_alloc_metric_sum_edges_exp(struct window *window, float weight, float log_beta, float eps)
{
    struct metric_sum_edges_exp *metric_sum_edges_exp;
    struct metric *metric;
    uint32_t graph_flags;

    LIST_FOR_EACH(metric, &window->metrics, struct metric, entry)
    {
        if (metric->ops != &metric_sum_edges_exp_ops) continue;
        metric_sum_edges_exp = METRIC_SUM_EDGES_EXP(metric);
        if (metric_sum_edges_exp->weight == weight &&
            metric_sum_edges_exp->log_beta == log_beta &&
            metric_sum_edges_exp->eps == eps)
        {
            return grab_metric(metric);
        }
    }

    if (!(metric_sum_edges_exp = malloc(sizeof(*metric_sum_edges_exp))))
        return NULL;

    metric = &metric_sum_edges_exp->metric;
    metric->refcount = 1;
    metric->window   = grab_window(window);
    metric->ops      = &metric_sum_edges_exp_ops;
    metric->valid    = 0;

    /* Enforce TVG_FLAGS_NONZERO, our update mechanism relies on it. */
    graph_flags = window->tvg->flags & (TVG_FLAGS_POSITIVE | TVG_FLAGS_DIRECTED);
    graph_flags |= TVG_FLAGS_NONZERO;

    if (!(metric_sum_edges_exp->result = alloc_graph(graph_flags)))
    {
        free_window(metric->window);
        free(metric_sum_edges_exp);
        return NULL;
    }

    graph_set_eps(metric_sum_edges_exp->result, eps);
    metric_sum_edges_exp->weight    = weight;
    metric_sum_edges_exp->log_beta  = log_beta;
    metric_sum_edges_exp->eps       = eps;

    list_add_tail(&window->metrics, &metric->entry);
    return metric;
}

float metric_sum_edges_exp_get_weight(struct metric *metric)
{
    return METRIC_SUM_EDGES_EXP(metric)->weight;
}

float metric_sum_edges_exp_get_log_beta(struct metric *metric)
{
    return METRIC_SUM_EDGES_EXP(metric)->log_beta;
}

float metric_sum_edges_exp_get_eps(struct metric *metric)
{
    return METRIC_SUM_EDGES_EXP(metric)->eps;
}

struct graph *metric_sum_edges_exp_get_result(struct metric *metric)
{
    if (!metric->valid) return NULL;
    return grab_graph(METRIC_SUM_EDGES_EXP(metric)->result);
}

/* metric_count_edges functions */

const struct metric_ops metric_count_edges_ops;

static inline struct metric_count_edges *METRIC_COUNT_EDGES(struct metric *metric)
{
    assert(metric->ops == &metric_count_edges_ops);
    return CONTAINING_RECORD(metric, struct metric_count_edges, metric);
}

static void metric_count_edges_free(struct metric *metric)
{
    struct metric_count_edges *metric_count_edges = METRIC_COUNT_EDGES(metric);
    free_graph(metric_count_edges->result);
}

static void metric_count_edges_reset(struct metric *metric)
{
    graph_clear(METRIC_COUNT_EDGES(metric)->result);
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
    return 1;  /* nothing to do */
}

const struct metric_ops metric_count_edges_ops =
{
    metric_count_edges_free,
    metric_count_edges_reset,
    metric_count_edges_add,
    metric_count_edges_sub,
    metric_count_edges_move,
};

struct metric *window_alloc_metric_count_edges(struct window *window)
{
    struct metric_count_edges *metric_count_edges;
    struct metric *metric;
    uint32_t graph_flags;

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
    metric->valid    = 0;

    /* Enforce TVG_FLAGS_POSITIVE and TVG_FLAGS_NONZERO, our update mechanism relies on it. */
    graph_flags = window->tvg->flags & TVG_FLAGS_DIRECTED;
    graph_flags |= TVG_FLAGS_POSITIVE | TVG_FLAGS_NONZERO;

    if (!(metric_count_edges->result = alloc_graph(graph_flags)))
    {
        free_window(metric->window);
        free(metric_count_edges);
        return NULL;
    }

    graph_set_eps(metric_count_edges->result, 0.5);
    list_add_tail(&window->metrics, &metric->entry);
    return metric;
}

struct graph *metric_count_edges_get_result(struct metric *metric)
{
    if (!metric->valid) return NULL;
    return grab_graph(METRIC_COUNT_EDGES(metric)->result);
}

/* metric_count_nodes functions */

const struct metric_ops metric_count_nodes_ops;

static inline struct metric_count_nodes *METRIC_COUNT_NODES(struct metric *metric)
{
    assert(metric->ops == &metric_count_nodes_ops);
    return CONTAINING_RECORD(metric, struct metric_count_nodes, metric);
}

static void metric_count_nodes_free(struct metric *metric)
{
    struct metric_count_nodes *metric_count_nodes = METRIC_COUNT_NODES(metric);
    free_vector(metric_count_nodes->result);
}

static void metric_count_nodes_reset(struct metric *metric)
{
    vector_clear(METRIC_COUNT_NODES(metric)->result);
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
    return 1;  /* nothing to do */
}

const struct metric_ops metric_count_nodes_ops =
{
    metric_count_nodes_free,
    metric_count_nodes_reset,
    metric_count_nodes_add,
    metric_count_nodes_sub,
    metric_count_nodes_move,
};

struct metric *window_alloc_metric_count_nodes(struct window *window)
{
    struct metric_count_nodes *metric_count_nodes;
    struct metric *metric;
    uint32_t vector_flags = TVG_FLAGS_POSITIVE | TVG_FLAGS_NONZERO;

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
    metric->valid    = 0;

    if (!(metric_count_nodes->result = alloc_vector(vector_flags)))
    {
        free_window(metric->window);
        free(metric_count_nodes);
        return NULL;
    }

    vector_set_eps(metric_count_nodes->result, 0.5);
    list_add_tail(&window->metrics, &metric->entry);
    return metric;
}

struct vector *metric_count_nodes_get_result(struct metric *metric)
{
    if (!metric->valid) return NULL;
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
    metric->ops->reset(metric);
    metric->valid = 0;
}

struct window *metric_get_window(struct metric *metric)
{
    return grab_window(metric->window);
}
