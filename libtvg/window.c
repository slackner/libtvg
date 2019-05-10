/*
 * Time-varying graph library
 * Window functions.
 *
 * Copyright (c) 2019 Sebastian Lackner
 */

#include "tvg.h"
#include "internal.h"

struct window *alloc_window(struct tvg *tvg, const struct window_ops *ops, int64_t window_l, int64_t window_r, float weight, float log_beta)
{
    struct window *window;

    if (window_l >= window_r)
        return NULL;
    if (!(window = malloc(sizeof(*window))))
        return NULL;

    window->refcount    = 1;
    window->eps         = 0.0;
    window->ts          = 0;
    window->window_l    = window_l;
    window->window_r    = window_r;
    window->tvg         = grab_tvg(tvg);
    window->ops         = ops;
    window->weight      = weight;
    window->log_beta    = log_beta;
    list_init(&window->sources);
    window->result      = NULL;

    return window;
}

struct window *grab_window(struct window *window)
{
    if (window) window->refcount++;
    return window;
}

void free_window(struct window *window)
{
    if (!window) return;
    if (--window->refcount) return;

    window_clear(window);
    free_tvg(window->tvg);
    free(window);
}

void window_set_eps(struct window *window, float eps)
{
    window->eps = (float)fabs(eps);
    window_clear(window);
}

void window_clear(struct window *window)
{
    struct source *source, *next_source;

    LIST_FOR_EACH_SAFE(source, next_source, &window->sources, struct source, entry)
    {
        list_remove(&source->entry);
        free_graph(source->graph);
        free(source);
    }

    free_graph(window->result);
    window->result = NULL;
}

struct graph *window_update(struct window *window, uint64_t ts)
{
    struct source *source_cursor, *source;
    struct tvg *tvg = window->tvg;
    uint32_t graph_flags;
    uint64_t ts_window_l;
    uint64_t ts_window_r;
    struct graph *graph;
    struct list todo;
    int update;

restart:
    if (!window->result)
    {
        /* Enforce TVG_FLAGS_NONZERO, our update mechanism relies on it. */
        graph_flags = tvg->flags & (TVG_FLAGS_POSITIVE | TVG_FLAGS_DIRECTED);
        graph_flags |= TVG_FLAGS_NONZERO;

        if (!(window->result = alloc_graph(graph_flags)))
            return NULL;

        /* Output graph should inherit window->eps. */
        graph_set_eps(window->result, window->eps);

        window->ts     = ts;
        assert(list_empty(&window->sources));
        update = 0;
    }
    else
    {
        update = (ts != window->ts);
    }

    list_init(&todo);
    source_cursor = LIST_ENTRY(window->sources.next, struct source, entry);

    if (ts >= (uint64_t)(-MIN(window->window_l + 1, 0)))
        ts_window_l = ts + (uint64_t)window->window_l + 1;
    else
        ts_window_l = 0;

    if (ts <= (uint64_t)(-MAX(window->window_r, 0) - 1))
        ts_window_r = ts + (uint64_t)window->window_r;
    else
        ts_window_r = ~0ULL;

    assert(ts_window_l <= ts_window_r);

    TVG_FOR_EACH_GRAPH_GE(tvg, graph, ts_window_l)
    {
        assert(graph->tvg == tvg);

        assert(graph->ts >= ts_window_l);
        if (graph->ts > ts_window_r)
            break;

        /* The current graph is included in our sliding window. Check if it
         * is already included in the current output and if it has not been
         * modified since. */

        source = source_cursor;
        while (&source->entry != &window->sources)
        {
            if (source->graph == graph) break;
            source = LIST_ENTRY(source->entry.next, struct source, entry);
        }

        if (&source->entry != &window->sources)
        {
            /* If the graph was modified since the last window_update() call,
             * drop the existing graph and start from scratch. */
            if (source->revision != graph->revision)
            {
                window_clear(window);
                free_graph(graph);
                goto restart;
            }

            if (source == source_cursor)
            {
                source_cursor = LIST_ENTRY(source->entry.next, struct source, entry);
            }
            else
            {
                list_remove(&source->entry);
                list_add_before(&source_cursor->entry, &source->entry);
            }
            continue;
        }

        /* The current graph is not included in the snapshot yet.
         * Add it as a new source and update the graph. */

        if (!(source = malloc(sizeof(*source))))
        {
            window_clear(window);
            free_graph(graph);
            return NULL;
        }

        source->graph    = grab_graph(graph);
        source->revision = graph->revision;
        list_add_before(&source_cursor->entry, &source->entry);

        /* Add to the TODO list - we will later add it to the graph. */
        list_add_tail(&todo, &source->todo_entry);
    }

    while (&source_cursor->entry != &window->sources)
    {
        source = source_cursor;
        source_cursor = LIST_ENTRY(source->entry.next, struct source, entry);

        /* If the graph was modified since the last window_update() call,
         * drop the existing graph and start from scratch. */
        graph = source->graph;
        if (source->revision != graph->revision)
        {
            window_clear(window);
            goto restart;
        }

        if (!window->ops->sub(window, graph))
        {
            window_clear(window);
            goto restart;
        }

        list_remove(&source->entry);
        free_graph(source->graph);
        free(source);
    }

    if (list_empty(&window->sources) && !graph_empty(window->result))
    {
        /* FIXME: This could be made more efficiently. */
        window_clear(window);
        goto restart;
    }

    if (update)
    {
        if (!window->ops->mov(window, ts))
        {
            window_clear(window);
            goto restart;
        }
        window->ts = ts;
    }

    LIST_FOR_EACH(source, &todo, struct source, todo_entry)
    {
        graph = source->graph;
        assert(source->revision == graph->revision);

        if (!window->ops->add(window, graph))
        {
            window_clear(window);
            return NULL;
        }
    }

    return grab_graph(window->result);
}
