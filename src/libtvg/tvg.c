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

    tvg->refcount   = 1;
    tvg->flags      = flags;
    list_init(&tvg->graphs);
    tvg->mongodb    = NULL;
    tvg->batch_size = 0;
    list_init(&tvg->cache);
    tvg->cache_used = 0;
    tvg->cache_size = 0;

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

    assert(list_empty(&tvg->cache));
    assert(!tvg->cache_used);
    free(tvg);
}

void tvg_debug(struct tvg *tvg)
{
    struct graph *graph;

    fprintf(stderr, "TVG %p\n", tvg);

    LIST_FOR_EACH(graph, &tvg->graphs, struct graph, entry)
    {
        fprintf(stderr, "-> Graph %p (ts %llu, id %llu, revision %llu) %s%s\n", graph,
                (long long unsigned int)graph->ts,
                (long long unsigned int)graph->id,
                (long long unsigned int)graph->revision,
                (graph->flags & TVG_FLAGS_LOAD_PREV) ? "load_prev " : "",
                (graph->flags & TVG_FLAGS_LOAD_NEXT) ? "load_next " : "");
    }
}

int tvg_link_graph(struct tvg *tvg, struct graph *graph, uint64_t ts)
{
    struct graph *other_graph;
    uint64_t id = graph->id;
    int res;

    if (graph->tvg)
        return 0;
    if ((tvg->flags ^ graph->flags) & TVG_FLAGS_DIRECTED)
        return 0;

    /* The common scenario is that new graphs are appended at the end,
     * so go through the existing graphs in reverse order. */

    LIST_FOR_EACH_REV(other_graph, &tvg->graphs, struct graph, entry)
    {
        assert(other_graph->tvg == tvg);
        if ((res = compare_graph_ts_id(other_graph, ts, id)) < 0) break;
        if (!res && id != ~0ULL) return 0;  /* MongoDB graphs can only be added once */
    }

    /* FIXME: Inherit flags of neighboring graphs. */
    if (tvg->mongodb)
        graph->flags |= TVG_FLAGS_LOAD_NEXT | TVG_FLAGS_LOAD_PREV;

    graph->ts  = ts;
    graph->tvg = tvg;
    list_add_after(&other_graph->entry, &graph->entry);
    grab_graph(graph);  /* grab extra reference */
    return 1;
}

struct graph *tvg_alloc_graph(struct tvg *tvg, uint64_t ts)
{
    struct graph *graph;
    uint32_t graph_flags;

    graph_flags = tvg->flags & (TVG_FLAGS_NONZERO |
                                TVG_FLAGS_POSITIVE |
                                TVG_FLAGS_DIRECTED);

    if (!(graph = alloc_graph(graph_flags)))
        return NULL;

    if (!tvg_link_graph(tvg, graph, ts))
    {
        free_graph(graph);
        return NULL;
    }

    return graph;
}

int tvg_load_graphs_from_file(struct tvg *tvg, const char *filename)
{
    long long unsigned int source, target;
    long long unsigned int ts;
    struct graph *graph = NULL;
    uint64_t numlines = 0;
    uint64_t maxlines = 0;
    uint64_t ticks;
    ssize_t read;
    float weight;
    size_t len = 0;
    char *line = NULL;
    int ret = 0;
    FILE *fp;

    if (!(fp = fopen(filename, "r")))
    {
        fprintf(stderr, "%s: File '%s' not found\n", __func__, filename);
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

        /* FIXME: Check that the full line was parsed. */
        if (sscanf(line, "%llu %llu %f %llu", &source, &target, &weight, &ts) < 4)
        {
            fprintf(stderr, "%s: Line does not match expected format\n", __func__);
            goto error;
        }

        if (graph && ts < graph->ts)
        {
            fprintf(stderr, "%s: Timestamps are not monotonically increasing\n", __func__);
            goto error;
        }

        if (!graph || ts != graph->ts)
        {
            if (graph) graph->revision = 0;
            free_graph(graph);
            if (!(graph = tvg_alloc_graph(tvg, ts)))
            {
                fprintf(stderr, "%s: Out of memory!\n", __func__);
                goto error;
            }
        }

        /* We use graph_set_edge() instead of graph_add_edge() to ensure this method
         * also works for undirected graphs with the reverse direction stored explicitly. */
        if (!graph_set_edge(graph, source, target, weight))
        {
            fprintf(stderr, "%s: Out of memory!\n", __func__);
            goto error;
        }
    }

    if (!graph)
    {
        fprintf(stderr, "%s: File appears to be empty\n", __func__);
        goto error;
    }

    /* Success if we read at least one edge. */
    ret = 1;

error:
    if (graph) graph->revision = 0;
    if (!ret) unlink_graph(graph);
    free_graph(graph);
    fclose(fp);
    free(line);
    return ret;
}

int tvg_enable_mongodb_sync(struct tvg *tvg, struct mongodb *mongodb,
                            uint64_t batch_size, uint64_t cache_size)
{
    struct graph *graph;

    if (!mongodb)
        return 0;
    if (!batch_size || batch_size > 4096)
        batch_size = 4096;

    free_mongodb(tvg->mongodb);
    tvg->mongodb = grab_mongodb(mongodb);
    tvg->batch_size = batch_size;
    tvg->cache_size = cache_size;

    LIST_FOR_EACH(graph, &tvg->graphs, struct graph, entry)
    {
        graph->flags |= TVG_FLAGS_LOAD_NEXT | TVG_FLAGS_LOAD_PREV;
    }

    return 1;
}

void tvg_disable_mongodb_sync(struct tvg *tvg)
{
    struct graph *graph;

    free_mongodb(tvg->mongodb);
    tvg->mongodb = NULL;

    LIST_FOR_EACH(graph, &tvg->graphs, struct graph, entry)
    {
        graph->flags &= ~(TVG_FLAGS_LOAD_NEXT | TVG_FLAGS_LOAD_PREV);
    }
}

struct window *tvg_alloc_window_rect(struct tvg *tvg, int64_t window_l, int64_t window_r)
{
    return alloc_window(tvg, &window_rect_ops, window_l, window_r, 0.0, 0.0);
}

struct window *tvg_alloc_window_decay(struct tvg *tvg, int64_t window, float log_beta)
{
    if (window <= 0) return NULL;
    return alloc_window(tvg, &window_decay_ops, -window, 0, 0.0, log_beta);
}

struct window *tvg_alloc_window_smooth(struct tvg *tvg, int64_t window, float log_beta)
{
    float weight = -(float)expm1(log_beta);
    if (window <= 0) return NULL;
    return alloc_window(tvg, &window_smooth_ops, -window, 0, weight, log_beta);
}

static inline struct graph *grab_prev_graph(struct tvg *tvg, struct graph *graph)
{
    struct graph *prev_graph = LIST_PREV(graph, &tvg->graphs, struct graph, entry);
    return grab_graph(prev_graph);
}

static inline struct graph *grab_next_graph(struct tvg *tvg, struct graph *graph)
{
    struct graph *next_graph = LIST_NEXT(graph, &tvg->graphs, struct graph, entry);
    return grab_graph(next_graph);
}

struct graph *tvg_lookup_graph_ge(struct tvg *tvg, uint64_t ts)
{
    struct graph *prev_graph;
    struct graph *graph;

    LIST_FOR_EACH(graph, &tvg->graphs, struct graph, entry)
    {
        assert(graph->tvg == tvg);
        if (graph->ts >= ts)
        {
            if (graph->flags & TVG_FLAGS_LOAD_PREV)
            {
                prev_graph = grab_prev_graph(tvg, graph);
                tvg_load_graphs_ge(tvg, graph, ts);
                graph = grab_next_graph(tvg, prev_graph);
                free_graph(prev_graph);
                assert(graph != NULL);
            }
            else
            {
                grab_graph(graph);
            }

            graph_refresh_cache(graph);
            return graph;
        }
    }

    if (tvg->mongodb)
    {
        prev_graph = grab_prev_graph(tvg, NULL);
        tvg_load_graphs_ge(tvg, NULL, ts);
        graph = grab_next_graph(tvg, prev_graph);
        free_graph(prev_graph);
        if (graph)
        {
            graph_refresh_cache(graph);
            return graph;
        }
    }

    return NULL;
}

struct graph *tvg_lookup_graph_le(struct tvg *tvg, uint64_t ts)
{
    struct graph *next_graph;
    struct graph *graph;

    LIST_FOR_EACH_REV(graph, &tvg->graphs, struct graph, entry)
    {
        assert(graph->tvg == tvg);
        if (graph->ts <= ts)
        {
            if (graph->flags & TVG_FLAGS_LOAD_NEXT)
            {
                next_graph = grab_next_graph(tvg, graph);
                tvg_load_graphs_le(tvg, graph, ts);
                graph = grab_prev_graph(tvg, next_graph);
                free_graph(next_graph);
                assert(graph != NULL);
            }
            else
            {
                grab_graph(graph);
            }

            graph_refresh_cache(graph);
            return graph;
        }
    }

    if (tvg->mongodb)
    {
        next_graph = grab_next_graph(tvg, NULL);
        tvg_load_graphs_le(tvg, NULL, ts);
        graph = grab_prev_graph(tvg, next_graph);
        free_graph(next_graph);
        if (graph)
        {
            graph_refresh_cache(graph);
            return graph;
        }
    }

    return NULL;
}

struct graph *tvg_lookup_graph_near(struct tvg *tvg, uint64_t ts)
{
    struct graph *other_graph;
    struct graph *graph;

    if (!(graph = tvg_lookup_graph_ge(tvg, ts)))
        return tvg_lookup_graph_le(tvg, ts);

    if ((other_graph = prev_graph(graph)))
    {
        assert(graph->ts >= ts && other_graph->ts < ts);
        if ((ts - other_graph->ts) < (graph->ts - ts))
        {
            free_graph(graph);
            graph = other_graph;
        }
        else
        {
            free_graph(other_graph);
        }
    }

    return graph;
}

int tvg_compress(struct tvg *tvg, uint64_t step, uint64_t offset)
{
    struct graph *graph, *next_graph;
    struct graph *prev_graph = NULL;

    if (tvg->mongodb)
        return 0;

    if (step)
        offset -= (offset / step) * step;

    LIST_FOR_EACH_SAFE(graph, next_graph, &tvg->graphs, struct graph, entry)
    {
        assert(graph->tvg == tvg);

        /* Round the timestamp to the desired step size. Afterwards,
         * sum up graphs with the same timestamp. If step is 0,
         * reduce the full TVG to a single timestamp. */

        if (!step)
            graph->ts = offset;
        else if (graph->ts >= offset)
            graph->ts = offset + ((graph->ts - offset) / step) * step;
        else
            graph->ts = 0;

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

struct graph *tvg_extract(struct tvg *tvg, uint64_t ts, float (*weight_func)(struct graph *,
                          uint64_t, void *), void *userdata)
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

    TVG_FOR_EACH_GRAPH_GE(tvg, graph, 0)
    {
        assert(graph->tvg == tvg);

        if ((weight = weight_func(graph, ts, userdata)) == 0.0)
            continue;

        if (!graph_add_graph(out, graph, weight))
        {
            free_graph(graph);
            free_graph(out);
            return NULL;
        }
    }

    return out;
}
