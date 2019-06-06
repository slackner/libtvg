/*
 * Time-varying graph library
 * Window functions.
 *
 * Copyright (c) 2019 Sebastian Lackner
 */

#include "tvg.h"
#include "internal.h"

struct window *tvg_alloc_window(struct tvg *tvg, int64_t window_l, int64_t window_r)
{
    struct window *window;

    if (window_l >= window_r)
        return NULL;
    if (!(window = malloc(sizeof(*window))))
        return NULL;

    window->refcount    = 1;
    window->ts          = 0;
    window->window_l    = window_l;
    window->window_r    = window_r;
    window->tvg         = grab_tvg(tvg);
    list_init(&window->sources);
    list_init(&window->metrics);

    return window;
}

struct window *grab_window(struct window *window)
{
    if (window) __sync_fetch_and_add(&window->refcount, 1);
    return window;
}

void free_window(struct window *window)
{
    if (!window) return;
    if (__sync_sub_and_fetch(&window->refcount, 1)) return;

    window_reset(window);

    assert(list_empty(&window->metrics));
    free_tvg(window->tvg);
    free(window);
}

void window_reset(struct window *window)
{
    struct source *source, *next_source;
    struct metric *metric;

    LIST_FOR_EACH_SAFE(source, next_source, &window->sources, struct source, entry)
    {
        list_remove(&source->entry);
        free_graph(source->graph);
        free(source);
    }

    LIST_FOR_EACH(metric, &window->metrics, struct metric, entry)
    {
        metric_reset(metric);
    }
}

int window_update(struct window *window, uint64_t ts)
{
    struct source *source_cursor, *source;
    struct tvg *tvg = window->tvg;
    struct metric *metric;
    uint64_t ts_window_l;
    uint64_t ts_window_r;
    struct graph *graph;
    struct list todo;

restart:
    LIST_FOR_EACH(metric, &window->metrics, struct metric, entry)
    {
        if (!metric->valid)
        {
            window_reset(window);
            window->ts = ts;
            break;
        }
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
                window_reset(window);
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
            window_reset(window);
            free_graph(graph);
            return 0;
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
            window_reset(window);
            goto restart;
        }

        LIST_FOR_EACH(metric, &window->metrics, struct metric, entry)
        {
            if (!metric->ops->sub(metric, graph))
            {
                window_reset(window);
                goto restart;
            }
        }

        list_remove(&source->entry);
        free_graph(source->graph);
        free(source);
    }

    if (list_empty(&window->sources))
    {
        LIST_FOR_EACH(metric, &window->metrics, struct metric, entry)
        {
            metric->ops->reset(metric);
        }
        window->ts = ts;
    }

    if (ts != window->ts)
    {
        LIST_FOR_EACH(metric, &window->metrics, struct metric, entry)
        {
            if (!metric->ops->move(metric, ts))
            {
                window_reset(window);
                window->ts = ts;
                goto restart;
            }
        }
        window->ts = ts;
    }

    LIST_FOR_EACH(source, &todo, struct source, todo_entry)
    {
        graph = source->graph;
        assert(source->revision == graph->revision);

        LIST_FOR_EACH(metric, &window->metrics, struct metric, entry)
        {
            if (!metric->ops->add(metric, graph))
            {
                window_reset(window);
                return 0;
            }
        }
    }

    LIST_FOR_EACH(metric, &window->metrics, struct metric, entry)
    {
        metric->valid = 1;
    }

    return 1;
}

uint64_t window_get_sources(struct window *window, struct graph **graphs, uint64_t max_graphs)
{
    struct source *source;
    uint64_t count = 0;

    LIST_FOR_EACH(source, &window->sources, struct source, entry)
    {
        if (count++ >= max_graphs) continue;
        if (graphs)
        {
            *graphs++ = grab_graph(source->graph);
        }
    }

    return count;
}
