/*
 * Time-varying graph library
 * Window functions.
 *
 * Copyright (c) 2019 Sebastian Lackner
 */

#include "tvg.h"
#include "internal.h"

static int rect_add(struct window *window, struct graph *graph)
{
    return graph_add_graph(window->result, graph, 1.0);
}

static int rect_sub(struct window *window, struct graph *graph)
{
    return graph_sub_graph(window->result, graph, 1.0);
}

static int rect_mov(struct window *window, float ts)
{
    /* nothing to do */
    return 1;
}

const struct window_ops window_rect_ops =
{
    rect_add,
    rect_sub,
    rect_mov,
};

static int decay_add(struct window *window, struct graph *graph)
{
    float weight = (float)exp(window->log_beta * (window->ts - graph->ts));
    return graph_add_graph(window->result, graph, weight);
}

static int decay_sub(struct window *window, struct graph *graph)
{
    float weight = (float)exp(window->log_beta * (window->ts - graph->ts));
    return graph_sub_graph(window->result, graph, weight);
}

static int decay_mov(struct window *window, float ts)
{
    float weight = (float)exp(window->log_beta * (ts - window->ts));

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
    decay_add,
    decay_sub,
    decay_mov,
};

static int smooth_add(struct window *window, struct graph *graph)
{
    float weight = window->weight * (float)exp(window->log_beta * (window->ts - graph->ts));
    return graph_add_graph(window->result, graph, weight);
}

static int smooth_sub(struct window *window, struct graph *graph)
{
    float weight = window->weight * (float)exp(window->log_beta * (window->ts - graph->ts));
    return graph_sub_graph(window->result, graph, weight);
}

static int smooth_mov(struct window *window, float ts)
{
    return decay_mov(window, ts);
}

const struct window_ops window_smooth_ops =
{
    smooth_add,
    smooth_sub,
    smooth_mov,
};
