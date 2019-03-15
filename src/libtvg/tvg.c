/*
 * Time-varying graph library
 * Generic functions.
 *
 * Copyright (c) 2018-2019 Sebastian Lackner
 */

#include "tvg.h"
#include "internal.h"

/* tvg_load_graphs relies on that */
C_ASSERT(sizeof(long long unsigned int) == sizeof(uint64_t));

struct tvg *alloc_tvg(uint32_t flags)
{
    struct tvg *tvg;

    if (flags & ~(TVG_FLAGS_NONZERO |
                  TVG_FLAGS_POSITIVE |
                  TVG_FLAGS_DIRECTED |
                  TVG_FLAGS_STREAMING))
        return NULL;

    if (!(tvg = malloc(sizeof(*tvg))))
        return NULL;

    tvg->refcount = 1;
    tvg->flags    = flags;
    list_init(&tvg->graphs);

    return tvg;
}

struct tvg *grab_tvg(struct tvg *tvg)
{
    if (tvg) tvg->refcount++;
    return tvg;
}

void free_tvg(struct tvg *tvg)
{
    struct graph *next_graph;
    struct graph *graph;

    if (!tvg) return;
    if (--tvg->refcount) return;

    LIST_FOR_EACH_SAFE(graph, next_graph, &tvg->graphs, struct graph, entry)
    {
        assert(graph->tvg == tvg);
        unlink_graph(graph);
        free_graph(graph);
    }

    free(tvg);
}

struct graph *tvg_alloc_graph(struct tvg *tvg, float ts)
{
    struct graph *graph, *other_graph;
    uint32_t graph_flags;

    graph_flags = tvg->flags & (TVG_FLAGS_NONZERO |
                                TVG_FLAGS_POSITIVE |
                                TVG_FLAGS_DIRECTED);

    if (!(graph = alloc_graph(graph_flags)))
        return NULL;

    graph->tvg         = tvg;
    graph->ts          = ts;

    /* The common scenario is that new graphs are appended at the end,
     * so go through the existing graphs in reverse order. */

    LIST_FOR_EACH_REV(other_graph, &tvg->graphs, struct graph, entry)
    {
        assert(other_graph->tvg == tvg);
        if (other_graph->ts <= ts) break;
    }

    list_add_after(&other_graph->entry, &graph->entry);
    return grab_graph(graph);  /* grab extra reference */
}

int tvg_load_graphs(struct tvg *tvg, const char *filename)
{
    long long unsigned int source, target;
    float weight, ts;
    struct graph *graph = NULL;
    uint64_t numlines = 0;
    uint64_t maxlines = 0;
    uint64_t ticks;
    ssize_t read;
    size_t len = 0;
    char *line = NULL;
    int ret = 0;
    FILE *fp;

    if (!(fp = fopen(filename, "r")))
    {
        fprintf(stderr, "tvg_load_graphs: File '%s' not found\n", filename);
        return 0;
    }

    ticks = clock_monotonic();
    while ((read = getline(&line, &len, fp)) > 0)
    {
        if ((++numlines & 0xffff) == 0 && (int64_t)(clock_monotonic() - ticks) >= 250)
        {
            if (!maxlines) maxlines = numlines + count_lines(fp);
            progress("Reading line %llu / %llu", (long long unsigned int)numlines, (long long unsigned int)maxlines);
            ticks = clock_monotonic();
        }

        if (line[read - 1] == '\n') line[read - 1] = 0;
        if (!line[0] || line[0] == '#' || line[0] == ';') continue;

        if (sscanf(line, "%llu %llu %f %f", &source, &target, &weight, &ts) < 4)
        {
            fprintf(stderr, "tvg_load_graphs: Line does not match expected format\n");
            goto error;
        }

        if (graph && ts < graph->ts)
        {
            fprintf(stderr, "tvg_load_graphs: Timestamps are not monotonically increasing\n");
            goto error;
        }

        if (!graph || ts != graph->ts)
        {
            if (graph) graph->revision = 0;
            free_graph(graph);
            if (!(graph = tvg_alloc_graph(tvg, ts)))
            {
                fprintf(stderr, "tvg_load_graphs: Out of memory!\n");
                goto error;
            }
        }

        /* We use graph_set_edge() instead of graph_add_edge() to ensure this method
         * also works for undirected graphs with the reverse direction stored explicitly. */
        if (!graph_set_edge(graph, source, target, weight))
        {
            fprintf(stderr, "tvg_load_graphs: Out of memory!\n");
            goto error;
        }
    }

    if (!graph)
    {
        fprintf(stderr, "tvg_load_graphs: File appears to be empty\n");
        goto error;
    }

    /* Success if we read at least one edge. */
    ret = 1;

error:
    if (graph) graph->revision = 0;
    free_graph(graph);
    fclose(fp);
    free(line);
    return ret;
}

struct window *tvg_alloc_window_rect(struct tvg *tvg, float window_l, float window_r)
{
    return alloc_window(tvg, &window_rect_ops, window_l, window_r, 0.0, 0.0);
}

struct window *tvg_alloc_window_decay(struct tvg *tvg, float window, float log_beta)
{
    return alloc_window(tvg, &window_decay_ops, -window, 0.0, 0.0, log_beta);
}

struct window *tvg_alloc_window_smooth(struct tvg *tvg, float window, float log_beta)
{
    float weight = -expm1(log_beta);
    return alloc_window(tvg, &window_smooth_ops, -window, 0.0, weight, log_beta);
}

struct graph *tvg_lookup_graph_ge(struct tvg *tvg, float ts)
{
    struct graph *graph;

    LIST_FOR_EACH(graph, &tvg->graphs, struct graph, entry)
    {
        assert(graph->tvg == tvg);
        if (graph->ts >= ts) return grab_graph(graph);
    }

    return NULL;
}

struct graph *tvg_lookup_graph_le(struct tvg *tvg, float ts)
{
    struct graph *graph;
    struct graph *ret = NULL;

    LIST_FOR_EACH(graph, &tvg->graphs, struct graph, entry)
    {
        assert(graph->tvg == tvg);
        if (graph->ts <= ts) ret = graph;
        else break;
    }

    return grab_graph(ret);
}

struct graph *tvg_lookup_graph_near(struct tvg *tvg, float ts)
{
    struct graph *graph;
    struct graph *ret = NULL;

    LIST_FOR_EACH(graph, &tvg->graphs, struct graph, entry)
    {
        assert(graph->tvg == tvg);
        if (graph->ts <= ts) ret = graph;
        else
        {
            if (!ret || ((graph->ts - ts) < (ts - ret->ts)))
                ret = graph;
            break;
        }
    }

    return grab_graph(ret);
}

int tvg_compress(struct tvg *tvg, float step, float offset)
{
    struct graph *graph, *next_graph;
    struct graph *prev_graph = NULL;

    LIST_FOR_EACH_SAFE(graph, next_graph, &tvg->graphs, struct graph, entry)
    {
        assert(graph->tvg == tvg);

        /* Round the timestamp to the desired step size. Afterwards,
         * sum up graphs with the same timestamp. If step is INF,
         * reduce the full TVG to a single timestamp. */

        if (!isinf(step))
        {
            graph->ts = offset + floor((graph->ts - offset) / step) * step;
        }
        else
        {
            graph->ts = offset;
        }

        if (prev_graph && graph->ts == prev_graph->ts)
        {
            if (!graph_add_graph(prev_graph, graph, 1.0))
                return 0;

            unlink_graph(graph);
            free_graph(graph);
            continue;
        }

        prev_graph = graph;
    }

    return 1;
}

struct graph *tvg_extract(struct tvg *tvg, float ts, float (*weight_func)(struct graph *,
                          float, void *), void *userdata)
{
    uint32_t graph_flags;
    struct graph *graph;
    struct graph *out;
    float weight;

    graph_flags = tvg->flags & (TVG_FLAGS_NONZERO |
                                TVG_FLAGS_POSITIVE |
                                TVG_FLAGS_DIRECTED);

    if (!(out = alloc_graph(graph_flags)))
        return NULL;

    LIST_FOR_EACH(graph, &tvg->graphs, struct graph, entry)
    {
        assert(graph->tvg == tvg);

        if ((weight = weight_func(graph, ts, userdata)) == 0.0)
            continue;

        if (!graph_add_graph(out, graph, weight))
        {
            free_graph(out);
            return NULL;
        }
    }

    return out;
}
