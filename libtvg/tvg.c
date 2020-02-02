/*
 * Time-varying graph library
 * Generic functions.
 *
 * Copyright (c) 2018-2019 Sebastian Lackner
 */

#include "internal.h"

/* tvg_load_graphs relies on that */
C_ASSERT(sizeof(long long unsigned int) == sizeof(uint64_t));

static int _graph_compar(const void *a, const void *b, void *userdata)
{
    const struct graph *ga = AVL_ENTRY(a, struct graph, entry);
    const struct graph *gb = AVL_ENTRY(b, struct graph, entry);
    return compare_graph_ts_objectid(ga, gb->ts, &gb->objectid);
}

static int _graph_lookup(const void *a, const void *b, void *userdata)
{
    const struct graph *ga = AVL_ENTRY(a, struct graph, entry);
    const uint64_t *ts = b;
    return COMPARE(ga->ts, *ts);
}

static int _nodes_ind_compar(const void *a, const void *b, void *userdata)
{
    const struct node *na = AVL_ENTRY(a, struct node, entry_ind);
    const struct node *nb = AVL_ENTRY(b, struct node, entry_ind);
    return COMPARE(na->index, nb->index);
}

static int _nodes_ind_lookup(const void *a, const void *b, void *userdata)
{
    const struct node *na = AVL_ENTRY(a, struct node, entry_ind);
    const uint64_t *index = b;
    return COMPARE(na->index, *index);
}

static int _nodes_key_compar(const void *a, const void *b, void *userdata)
{
    /* const */ struct node *node1 = AVL_ENTRY(a, struct node, entry_key);
    /* const */ struct node *node2 = AVL_ENTRY(b, struct node, entry_key);
    struct attribute *attr1, *attr2;
    struct tvg *tvg = userdata;
    int res;

    NODE_FOR_EACH_PRIMARY_ATTRIBUTE2(tvg, node1, attr1, node2, attr2)
    {
        if (attr1 && attr2)
        {
            if ((res = strcmp(attr1->value, attr2->value)))
                return res;
        }
        else if (attr1 && !attr2)
            return 1;
        else if (!attr1 && attr2)
            return -1;
    }

    return 0;
}

static int _nodes_key_lookup(const void *a, const void *b, void *userdata)
{
    return _nodes_key_compar(a, b, userdata);
}

struct tvg *alloc_tvg(uint32_t flags)
{
    struct tvg *tvg;

    if (flags & ~(TVG_FLAGS_POSITIVE |
                  TVG_FLAGS_DIRECTED |
                  TVG_FLAGS_STREAMING))
        return NULL;

    if (!(tvg = malloc(sizeof(*tvg))))
        return NULL;

    tvg->refcount   = 1;
    tvg->flags      = flags;
    tvg->verbosity  = 0;
    avl_init(&tvg->graphs, _graph_compar, _graph_lookup, NULL);
    list_init(&tvg->queries);
    tvg->mongodb    = NULL;
    tvg->batch_size = 0;
    avl_init(&tvg->nodes_ind, _nodes_ind_compar, _nodes_ind_lookup, NULL);
    avl_init(&tvg->nodes_key, _nodes_key_compar, _nodes_key_lookup, tvg);
    list_init(&tvg->primary_key);
    tvg->next_node  = 0;
    list_init(&tvg->graph_cache);
    tvg->graph_cache_used = 0;
    tvg->graph_cache_size = 0;
    list_init(&tvg->query_cache);
    tvg->query_cache_used = 0;
    tvg->query_cache_size = 0;

    return tvg;
}

struct tvg *grab_tvg(struct tvg *tvg)
{
    if (tvg) __sync_fetch_and_add(&tvg->refcount, 1);
    return tvg;
}

void free_tvg(struct tvg *tvg)
{
    struct query *query, *next_query;
    struct attribute *attr, *next_attr;
    struct graph *graph, *next_graph;
    struct node *node, *next_node;

    if (!tvg) return;
    if (__sync_sub_and_fetch(&tvg->refcount, 1)) return;

    AVL_FOR_EACH_SAFE(graph, next_graph, &tvg->graphs, struct graph, entry)
    {
        assert(graph->tvg == tvg);
        unlink_graph(graph);
    }

    LIST_FOR_EACH_SAFE(query, next_query, &tvg->queries, struct query, entry)
    {
        assert(query->tvg == tvg);
        unlink_query(query, 1);
    }

    AVL_FOR_EACH_SAFE(node, next_node, &tvg->nodes_ind, struct node, entry_ind)
    {
        assert(node->tvg == tvg);
        unlink_node(node);
    }

    LIST_FOR_EACH_SAFE(attr, next_attr, &tvg->primary_key, struct attribute, entry)
    {
        list_remove(&attr->entry);
        free(attr);
    }

    assert(avl_empty(&tvg->nodes_key));
    assert(list_empty(&tvg->graph_cache));
    assert(!tvg->graph_cache_used);
    assert(list_empty(&tvg->query_cache));
    assert(!tvg->query_cache_used);

    free_mongodb(tvg->mongodb);
    free(tvg);
}

void tvg_set_verbosity(struct tvg *tvg, int verbosity)
{
    tvg->verbosity = verbosity;
}

void tvg_debug(struct tvg *tvg)
{
    struct graph *graph;
    char objectid_str[32];

    fprintf(stderr, "TVG %p\n", tvg);

    AVL_FOR_EACH(graph, &tvg->graphs, struct graph, entry)
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

    AVL_FOR_EACH(graph, &tvg->graphs, struct graph, entry)
    {
        size += graph_memory_usage(graph);
    }

    /* FIXME: This is not fully correct, there are some more structures
     * associated with a TVG that are not taken into account here. */

    return size;
}

int tvg_link_graph(struct tvg *tvg, struct graph *graph, uint64_t ts)
{
    if (graph->tvg)
        return 0;
    if ((tvg->flags ^ graph->flags) & TVG_FLAGS_DIRECTED)
        return 0;

    graph->ts = ts;
    if (avl_insert(&tvg->graphs, &graph->entry, objectid_empty(&graph->objectid)))
        return 0;  /* MongoDB graphs can only be added once */

    /* FIXME: Inherit flags of neighboring graphs. */
    if (tvg->mongodb)
        graph->flags |= TVG_FLAGS_LOAD_NEXT | TVG_FLAGS_LOAD_PREV;

    graph->flags |= TVG_FLAGS_READONLY;  /* block changes */
    graph->tvg = tvg;
    grab_graph(graph);  /* grab extra reference */

    tvg_invalidate_queries(tvg, ts, ts);
    return 1;
}

int tvg_set_primary_key(struct tvg *tvg, const char *key)
{
    struct attribute *attr, *next_attr, *other_attr;
    struct node *node, *next_node;
    struct list primary_key;
    const char *next;
    size_t keylen;
    int res;

    list_init(&primary_key);

    for (;;)
    {
        next = strchr(key, ';');
        keylen = next ? (next - key) : strlen(key);

        if (!keylen)
            goto skip;

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

    AVL_FOR_EACH_SAFE(node, next_node, &tvg->nodes_key, struct node, entry_key)
    {
        avl_remove(&node->entry_key);
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

    if (!list_empty(&tvg->primary_key))
    {
        AVL_FOR_EACH(node, &tvg->nodes_ind, struct node, entry_ind)
        {
            /* FIXME: When creating nodes before setting the primary key,
             * there could be conflicts. */
            avl_insert(&tvg->nodes_key, &node->entry_key, 0);
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
    struct avl_entry *entry;

    if (ret_node)
        *ret_node = NULL;

    if (node->tvg)
        return 0;

    if (!list_empty(&tvg->primary_key))
    {
        if ((entry = avl_insert(&tvg->nodes_key, &node->entry_key, 0)))
        {
            node = AVL_ENTRY(entry, struct node, entry_key);
            assert(node->tvg == tvg);
            if (ret_node) *ret_node = grab_node(node);
            return 0;  /* collision */
        }
    }
    else
    {
        /* entry_key is unused since no primary key is set */
        node->entry_key.parent = NULL;
    }

    if (index != ~0ULL)
    {
        node->index = index;
        if (avl_insert(&tvg->nodes_ind, &node->entry_ind, 0))
        {
            avl_remove(&node->entry_key);
            return 0;
        }
    }
    else for (;;)
    {
        node->index = tvg->next_node++;
        if (!avl_insert(&tvg->nodes_ind, &node->entry_ind, 0))
            break;
    }

    node->tvg = tvg;
    grab_node(node);  /* grab extra reference */
    return 1;
}

struct node *tvg_get_node_by_index(struct tvg *tvg, uint64_t index)
{
    struct node *node;

    node = AVL_LOOKUP(&tvg->nodes_ind, &index, struct node, entry_ind);
    return grab_node(node);
}

struct node *tvg_get_node_by_primary_key(struct tvg *tvg, struct node *primary_key)
{
    struct node *node;

    if (list_empty(&tvg->primary_key))
        return NULL;  /* no primary key set */

    node = AVL_LOOKUP(&tvg->nodes_key, &primary_key->entry_key, struct node, entry_key);
    return grab_node(node);
}

int tvg_load_graphs_from_file(struct tvg *tvg, const char *filename)
{
    long long unsigned int source, target;
    long long unsigned int ts;
    struct graph *graph = NULL;
    uint32_t graph_flags;
    uint64_t numlines = 0;
    uint64_t maxlines = 0;
    uint64_t ticks;
    ssize_t read;
    float weight;
    size_t len = 0;
    char *line = NULL;
    int ret = 0;
    FILE *fp;

    graph_flags = tvg->flags & (TVG_FLAGS_POSITIVE |
                                TVG_FLAGS_DIRECTED);

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
            if (graph)
            {
                graph->revision = 0;
                if (!tvg_link_graph(tvg, graph, graph->ts))
                {
                    fprintf(stderr, "%s: Failed to link graph!\n", __func__);
                    goto error;
                }
                free_graph(graph);
            }

            if (!(graph = alloc_graph(graph_flags)))
            {
                fprintf(stderr, "%s: Out of memory!\n", __func__);
                goto error;
            }

            graph->ts = ts;
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

    graph->revision = 0;
    if (!tvg_link_graph(tvg, graph, graph->ts))
    {
        fprintf(stderr, "%s: Failed to link graph!\n", __func__);
        goto error;
    }

    /* Success if we read at least one edge. */
    ret = 1;

error:
    free_graph(graph);
    fclose(fp);
    free(line);
    return ret;
}

static int load_node_attributes(struct node *node, const char *key, char *text)
{
    const char *next_key;
    char *next_text;
    size_t keylen;

    for (;;)
    {
        next_key = strchr(key, ';');
        keylen = next_key ? (next_key - key) : strlen(key);

        if (!keylen)
            goto skip;

        next_text = strchr(text, '\t');
        if (next_text) *next_text = 0;

        if (!node_set_attribute_internal(node, key, keylen, text))
        {
            fprintf(stderr, "%s: Out of memory!\n", __func__);
            return 0;
        }

        if (!next_text) break;
        text = next_text + 1;

    skip:
        if (!next_key) break;
        key = next_key + 1;
    }

    return 1;
}

int tvg_load_nodes_from_file(struct tvg *tvg, const char *filename, const char *key)
{
    long long unsigned int index;
    struct node *node;
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

        if (line[offset++] != '\t')
        {
            fprintf(stderr, "%s: Line does not match expected format\n", __func__);
            goto error;
        }

        if (!(node = alloc_node()))
        {
            fprintf(stderr, "%s: Out of memory!\n", __func__);
            goto error;
        }

        if (!load_node_attributes(node, key, &line[offset]))
        {
            free_node(node);
            goto error;
        }

        if (!tvg_link_node(tvg, node, NULL, index))
        {
            fprintf(stderr, "%s: Node %llu appears to be a duplicate\n", __func__, index);
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
    tvg->graph_cache_size = cache_size;

    AVL_FOR_EACH(graph, &tvg->graphs, struct graph, entry)
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

    AVL_FOR_EACH(graph, &tvg->graphs, struct graph, entry)
    {
        graph->flags &= ~(TVG_FLAGS_LOAD_NEXT | TVG_FLAGS_LOAD_PREV);
    }
}

int tvg_enable_query_cache(struct tvg *tvg, uint64_t cache_size)
{
    tvg->query_cache_size = cache_size;
    return 1;
}

void tvg_disable_query_cache(struct tvg *tvg)
{
    tvg->query_cache_size = 0;
    tvg_invalidate_queries(tvg, 0, ~0ULL);
}

static inline struct graph *grab_prev_graph(struct tvg *tvg, struct graph *graph)
{
    struct graph *prev_graph = AVL_PREV(graph, &tvg->graphs, struct graph, entry);
    return grab_graph(prev_graph);
}

static inline struct graph *grab_next_graph(struct tvg *tvg, struct graph *graph)
{
    struct graph *next_graph = AVL_NEXT(graph, &tvg->graphs, struct graph, entry);
    return grab_graph(next_graph);
}

struct graph *tvg_lookup_graph_ge(struct tvg *tvg, uint64_t ts)
{
    struct graph *prev_graph;
    struct graph *graph;

    if ((graph = AVL_LOOKUP_GE(&tvg->graphs, &ts, struct graph, entry)))
    {
        assert(graph->tvg == tvg);
        assert(graph->ts >= ts);

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

    if ((graph = AVL_LOOKUP_LE(&tvg->graphs, &ts, struct graph, entry)))
    {
        assert(graph->tvg == tvg);
        assert(graph->ts <= ts);

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

int tvg_compress(struct tvg *tvg, int (*callback)(uint64_t, struct snapshot_entry *, void *),
                 void *userdata)
{
    struct graph *prev_graph = NULL;
    struct snapshot_entry prev_entry;
    struct graph *graph;
    int ret;

    if (tvg->mongodb)
        return 0;

    tvg_invalidate_queries(tvg, 0, ~0ULL);

    TVG_FOR_EACH_GRAPH_GE(tvg, graph, 0)
    {
        assert(graph->tvg == tvg);

        if (prev_graph && graph->ts <= prev_entry.ts_max)
        {
            assert(graph->ts >= prev_entry.ts_min);
            if (!graph_add_graph(prev_graph, graph, 1.0))
            {
                tvg_link_graph(tvg, prev_graph, prev_graph->ts);
                free_graph(prev_graph);
                return 0;
            }

            unlink_graph(graph);
            continue;
        }

        if (prev_graph)
        {
            ret = tvg_link_graph(tvg, prev_graph, prev_graph->ts);
            free_graph(prev_graph);
            if (!ret) return 0;
        }

        prev_entry.ts_min = ~0ULL;
        prev_entry.ts_max = 0;

        if (!callback(graph->ts, &prev_entry, userdata))
        {
            fprintf(stderr, "%s: Callback failed.\n", __func__);
            return 0;
        }

        if (prev_entry.ts_min > graph->ts || prev_entry.ts_max < graph->ts)
        {
            fprintf(stderr, "%s: Time %llu not within [%llu, %llu].\n", __func__,
                            (long long unsigned int)graph->ts,
                            (long long unsigned int)prev_entry.ts_min,
                            (long long unsigned int)prev_entry.ts_max);
            return 0;
        }

        prev_graph = grab_graph(graph);
        unlink_graph(graph);

        prev_graph->ts = prev_entry.ts_min;
        prev_graph->objectid.type = OBJECTID_NONE;
    }

    if (prev_graph)
    {
        ret = tvg_link_graph(tvg, prev_graph, prev_graph->ts);
        free_graph(prev_graph);
        if (!ret) return 0;
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

    graph_flags = tvg->flags & (TVG_FLAGS_POSITIVE |
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
            free_graph(out);
            return NULL;
        }
    }

    return out;
}

void tvg_invalidate_queries(struct tvg *tvg, uint64_t ts_min, uint64_t ts_max)
{
    struct query *query, *next_query;

    if (ts_max < ts_min)
        return;

    LIST_FOR_EACH_SAFE(query, next_query, &tvg->queries, struct query, entry)
    {
        if (query->ts_max < ts_min) continue;
        if (query->ts_min > ts_max) continue;
        unlink_query(query, 1);
    }
}
