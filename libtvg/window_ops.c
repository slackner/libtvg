/*
 * Time-varying graph library
 * Window functions.
 *
 * Copyright (c) 2019 Sebastian Lackner
 */

#include "tvg.h"
#include "internal.h"

static inline double sub_uint64(uint64_t a, uint64_t b)
{
    return (a < b) ? -(double)(b - a) : (double)(a - b);
}

static float rect_weight(struct window *window, struct graph *graph)
{
    return 1.0f;
}

static int rect_add(struct window *window, struct graph *graph)
{
    return graph_add_graph(window->result, graph, 1.0);
}

static int rect_sub(struct window *window, struct graph *graph)
{
    return graph_sub_graph(window->result, graph, 1.0);
}

static int rect_mov(struct window *window, uint64_t ts)
{
    /* nothing to do */
    return 1;
}

const struct window_ops window_rect_ops =
{
    rect_weight,
    rect_add,
    rect_sub,
    rect_mov,
};

static float decay_weight(struct window *window, struct graph *graph)
{
    return (float)exp(window->log_beta * sub_uint64(window->ts, graph->ts));
}

static int decay_add(struct window *window, struct graph *graph)
{
    float weight = decay_weight(window, graph);
    return graph_add_graph(window->result, graph, weight);
}

static int decay_sub(struct window *window, struct graph *graph)
{
    float weight = decay_weight(window, graph);
    return graph_sub_graph(window->result, graph, weight);
}

static int decay_mov(struct window *window, uint64_t ts)
{
    float weight = (float)exp(window->log_beta * sub_uint64(ts, window->ts));

    /* If ts << window->ts the update formula is not numerically stable:
     * \delta w_new ~ \delta w_old * pow(window->beta, ts - window->ts).
     * Force recomputation if errors could get too large. */

    if (weight >= 1000.0)  /* FIXME: Arbitrary limit. */
        return 0;

    graph_mul_const(window->result, weight);
    return 1;
}

const struct window_ops window_decay_ops =
{
    decay_weight,
    decay_add,
    decay_sub,
    decay_mov,
};

static float smooth_weight(struct window *window, struct graph *graph)
{
    return window->weight * (float)exp(window->log_beta * sub_uint64(window->ts, graph->ts));
}

static int smooth_add(struct window *window, struct graph *graph)
{
    float weight = smooth_weight(window, graph);
    return graph_add_graph(window->result, graph, weight);
}

static int smooth_sub(struct window *window, struct graph *graph)
{
    float weight = smooth_weight(window, graph);
    return graph_sub_graph(window->result, graph, weight);
}

static int smooth_mov(struct window *window, uint64_t ts)
{
    return decay_mov(window, ts);
}

const struct window_ops window_smooth_ops =
{
    smooth_weight,
    smooth_add,
    smooth_sub,
    smooth_mov,
};
