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
    uint32_t i;

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
    for (i = 0; i < ARRAY_SIZE(tvg->nodes_ind); i++)
        list_init(&tvg->nodes_ind[i]);
    for (i = 0; i < ARRAY_SIZE(tvg->nodes_key); i++)
        list_init(&tvg->nodes_key[i]);
    list_init(&tvg->primary_key);
    tvg->next_node  = 0;
    list_init(&tvg->cache);
    tvg->cache_used = 0;
    tvg->cache_size = 0;

    return tvg;
}

struct tvg *grab_tvg(struct tvg *tvg)
{
    if (tvg) __sync_fetch_and_add(&tvg->refcount, 1);
    return tvg;
}

void free_tvg(struct tvg *tvg)
{
    struct attribute *attr, *next_attr;
    struct graph *graph, *next_graph;
    struct node *node, *next_node;
    uint32_t i;

    if (!tvg) return;
    if (__sync_sub_and_fetch(&tvg->refcount, 1)) return;

    LIST_FOR_EACH_SAFE(graph, next_graph, &tvg->graphs, struct graph, entry)
    {
        assert(graph->tvg == tvg);
        unlink_graph(graph);
    }

    for (i = 0; i < ARRAY_SIZE(tvg->nodes_ind); i++)
    {
        LIST_FOR_EACH_SAFE(node, next_node, &tvg->nodes_ind[i], struct node, entry_ind)
        {
            assert(node->tvg == tvg);
            unlink_node(node);
        }
    }

    for (i = 0; i < ARRAY_SIZE(tvg->nodes_key); i++)
        assert(list_empty(&tvg->nodes_key[i]));

    LIST_FOR_EACH_SAFE(attr, next_attr, &tvg->primary_key, struct attribute, entry)
    {
        list_remove(&attr->entry);
        free(attr);
    }

    assert(list_empty(&tvg->cache));
    assert(!tvg->cache_used);
    free(tvg);
}

void tvg_debug(struct tvg *tvg)
{
    struct graph *graph;
    char objectid_str[32];

    fprintf(stderr, "TVG %p\n", tvg);

    LIST_FOR_EACH(graph, &tvg->graphs, struct graph, entry)
    {
        objectid_to_str(&graph->objectid, objectid_str);
        fprintf(stderr, "-> Graph %p (ts %llu, objectid %s, revision %llu) %s%s\n", graph,
                (long long unsigned int)graph->ts, objectid_str,
                (long long unsigned int)graph->revision,
                (graph->flags & TVG_FLAGS_LOAD_PREV) ? "load_prev " : "",
                (graph->flags & TVG_FLAGS_LOAD_NEXT) ? "load_next " : "");
    }
}

uint64_t tvg_memory_usage(struct tvg *tvg)
{
    struct graph *graph;
    uint64_t size = sizeof(*tvg);

    LIST_FOR_EACH(graph, &tvg->graphs, struct graph, entry)
    {
        size += graph_memory_usage(graph);
    }

    /* FIXME: This is not fully correct, there are some more structures
     * associated with a TVG that are not taken into account here. */

    return size;
}

int tvg_link_graph(struct tvg *tvg, struct graph *graph, uint64_t ts)
{
    struct graph *other_graph;
    struct objectid *objectid = &graph->objectid;
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
        if ((res = compare_graph_ts_objectid(other_graph, ts, objectid)) < 0) break;
        if (!res && !objectid_empty(objectid)) return 0;  /* MongoDB graphs can only be added once */
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

int tvg_set_primary_key(struct tvg *tvg, const char *key)
{
    struct attribute *attr, *next_attr, *other_attr;
    struct list primary_key;
    struct node *node;
    const char *next;
    uint32_t hash;
    size_t keylen;
    uint32_t i;
    int res;

    list_init(&primary_key);

    for (;;)
    {
        next = strchr(key, ';');
        keylen = next ? (next - key) : strlen(key);

        if (!(attr = malloc(offsetof(struct attribute, buffer[keylen + 1]))))
            goto error;

        attr->key     = attr->buffer;
        attr->value   = &attr->buffer[keylen];
        memcpy((char *)attr->key, key, keylen);
        ((char *)attr->key)[keylen] = 0;

        LIST_FOR_EACH(other_attr, &primary_key, struct attribute, entry)
        {
            if ((res = strcmp(other_attr->key, attr->key)) > 0) break;
            if (!res)  /* Attribute keys shouldn't collide */
            {
                free(attr);
                goto skip;
            }
        }

        list_add_before(&other_attr->entry, &attr->entry);
    skip:
        if (!next) break;
        key = next + 1;
    }

    LIST_FOR_EACH_SAFE(attr, next_attr, &tvg->primary_key, struct attribute, entry)
    {
        list_remove(&attr->entry);
        free(attr);
    }

    LIST_FOR_EACH_SAFE(attr, next_attr, &primary_key, struct attribute, entry)
    {
        list_remove(&attr->entry);
        list_add_tail(&tvg->primary_key, &attr->entry);
    }

    for (i = 0; i < ARRAY_SIZE(tvg->nodes_ind); i++)
    {
        LIST_FOR_EACH(node, &tvg->nodes_ind[i], struct node, entry_ind)
        {
            list_remove(&node->entry_key);

            /* FIXME: When creating nodes before setting the primary key,
             * there could be conflicts. */

            hash = node_hash_primary_key(tvg, node);
            if (hash == ~0U) list_init(&node->entry_key);
            else list_add_head(&tvg->nodes_key[hash], &node->entry_key);
        }
    }

    return 1;

error:
    LIST_FOR_EACH_SAFE(attr, next_attr, &primary_key, struct attribute, entry)
    {
        list_remove(&attr->entry);
        free(attr);
    }
    return 0;
}

int tvg_link_node(struct tvg *tvg, struct node *node, struct node **ret_node, uint64_t index)
{
    struct node *other_node;
    uint32_t hash_ind, hash_key;

    if (ret_node)
        *ret_node = NULL;

    if (node->tvg)
        return 0;

    if ((hash_key = node_hash_primary_key(tvg, node)) != ~0U)
    {
        LIST_FOR_EACH(other_node, &tvg->nodes_key[hash_key], struct node, entry_key)
        {
            assert(other_node->tvg == tvg);
            if (node_equal_key(tvg, node, other_node))
            {
                if (ret_node) *ret_node = grab_node(other_node);
                return 0;  /* collision */
            }
        }
    }

    if (index != ~0ULL)
    {
        hash_ind = node_hash_index(tvg, index);
        LIST_FOR_EACH(other_node, &tvg->nodes_ind[hash_ind], struct node, entry_ind)
        {
            assert(other_node->tvg == tvg);
            if (other_node->index == index) return 0;
        }
    }
    else
    {
    retry:
        index = tvg->next_node++;

        hash_ind = node_hash_index(tvg, index);
        LIST_FOR_EACH(other_node, &tvg->nodes_ind[hash_ind], struct node, entry_ind)
        {
            assert(other_node->tvg == tvg);
            if (other_node->index == index) goto retry;
        }
    }


    node->index = index;
    node->tvg   = tvg;
    list_add_head(&tvg->nodes_ind[hash_ind], &node->entry_ind);

    if (hash_key == ~0U) list_init(&node->entry_key);
    else list_add_head(&tvg->nodes_key[hash_key], &node->entry_key);

    grab_node(node);  /* grab extra reference */
    return 1;
}

struct node *tvg_get_node_by_index(struct tvg *tvg, uint64_t index)
{
    struct node *node;
    uint32_t hash;

    hash = node_hash_index(tvg, index);
    LIST_FOR_EACH(node, &tvg->nodes_ind[hash], struct node, entry_ind)
    {
        assert(node->tvg == tvg);
        if (node->index == index) return grab_node(node);
    }

    return NULL;
}

struct node *tvg_get_node_by_primary_key(struct tvg *tvg, struct node *primary_key)
{
    struct node *node;
    uint32_t hash;

    if ((hash = node_hash_primary_key(tvg, primary_key)) == ~0U)
        return NULL;

    LIST_FOR_EACH(node, &tvg->nodes_key[hash], struct node, entry_key)
    {
        assert(node->tvg == tvg);
        if (node_equal_key(tvg, node, primary_key)) return grab_node(node);
    }

    return NULL;
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

        if (line[read - 1] == '\n') line[--read] = 0;
        if (read > 0 && line[read - 1] == '\r') line[--read] = 0;

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

int tvg_load_nodes_from_file(struct tvg *tvg, const char *filename)
{
    long long unsigned int index;
    struct node *node;
    const char *text;
    ssize_t read;
    size_t len = 0;
    char *line = NULL;
    int offset;
    int ret = 0;
    FILE *fp;

    if (!(fp = fopen(filename, "r")))
    {
        fprintf(stderr, "%s: File '%s' not found\n", __func__, filename);
        return 0;
    }

    while ((read = getline(&line, &len, fp)) > 0)
    {
        if (line[read - 1] == '\n') line[--read] = 0;
        if (read > 0 && line[read - 1] == '\r') line[--read] = 0;

        if (!line[0] || line[0] == '#' || line[0] == ';') continue;

        if (sscanf(line, "%llu%n", &index, &offset) < 1)
        {
            fprintf(stderr, "%s: Line does not match expected format\n", __func__);
            goto error;
        }

        text = &line[offset];
        if (*text != ' ' && *text != '\t')
        {
            fprintf(stderr, "%s: Line does not match expected format\n", __func__);
            goto error;
        }
        while (*text == ' ' || *text == '\t') text++;

        if (!(node = alloc_node()))
        {
            fprintf(stderr, "%s: Out of memory!\n", __func__);
            goto error;
        }

        if (!node_set_attribute(node, "text", text))
        {
            fprintf(stderr, "%s: Out of memory!\n", __func__);
            free_node(node);
            goto error;
        }

        if (!tvg_link_node(tvg, node, NULL, index))
        {
            fprintf(stderr, "%s: Index %llu or text '%s' already used?\n", __func__, index, text);
            free_node(node);
            continue;
        }

        free_node(node);
    }

    /* Success if we read at least one ĺabel. */
    ret = 1;

error:
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
