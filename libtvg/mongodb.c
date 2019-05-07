/*
 * Time-varying graph library
 * MongoDB support.
 *
 * Copyright (c) 2019 Sebastian Lackner
 */

#ifdef HAVE_LIBMONGOC

#include <mongoc.h>

#include "tvg.h"
#include "internal.h"

struct occurrence
{
    uint64_t sen;
    uint64_t ent;
};

static void free_mongodb_config(struct mongodb_config *config)
{
    free(config->uri);
    free(config->database);
    free(config->col_articles);
    free(config->article_id);
    free(config->article_time);
    free(config->col_entities);
    free(config->entity_doc);
    free(config->entity_sen);
    free(config->entity_ent);
    free(config);
}

static struct mongodb_config *alloc_mongodb_config(const struct mongodb_config *orig)
{
    struct mongodb_config *config;

    if (!orig->uri)          return NULL;
    if (!orig->database)     return NULL;
    if (!orig->col_articles) return NULL;
    if (!orig->article_id)   return NULL;
    if (!orig->article_time) return NULL;
    if (!orig->col_entities) return NULL;
    if (!orig->entity_doc)   return NULL;
    if (!orig->entity_sen)   return NULL;
    if (!orig->entity_ent)   return NULL;

    if (!(config = malloc(sizeof(*config))))
        return NULL;

    memset(config, 0, sizeof(*config));
    if (!(config->uri          = strdup(orig->uri)))          goto error;
    if (!(config->database     = strdup(orig->database)))     goto error;
    if (!(config->col_articles = strdup(orig->col_articles))) goto error;
    if (!(config->article_id   = strdup(orig->article_id)))   goto error;
    if (!(config->article_time = strdup(orig->article_time))) goto error;
    if (!(config->col_entities = strdup(orig->col_entities))) goto error;
    if (!(config->entity_doc   = strdup(orig->entity_doc)))   goto error;
    if (!(config->entity_sen   = strdup(orig->entity_sen)))   goto error;
    if (!(config->entity_ent   = strdup(orig->entity_ent)))   goto error;
    config->load_nodes   = orig->load_nodes;
    config->max_distance = orig->max_distance;
    return config;

error:
    free_mongodb_config(config);
    return NULL;
}

struct mongodb *alloc_mongodb(const struct mongodb_config *config)
{
    struct mongodb *mongodb;
    bson_t *command, reply;
    bson_error_t error;
    mongoc_uri_t *uri;
    bool success;
    char *str;

    if (!(mongodb = malloc(sizeof(*mongodb))))
        return NULL;

    if (!(mongodb->config = alloc_mongodb_config(config)))
    {
        free(mongodb);
        return NULL;
    }

    mongodb->refcount = 1;
    mongodb->client   = NULL;
    mongodb->articles = NULL;
    mongodb->entities = NULL;

    /* From now on, always use the copied version. */
    config = mongodb->config;

    uri = mongoc_uri_new_with_error(config->uri, &error);
    if (!uri)
    {
        fprintf(stderr, "%s: Parsing URI '%s' failed: %s\n", __func__, config->uri, error.message);
        free_mongodb(mongodb);
        return NULL;
    }

    mongodb->client = mongoc_client_new_from_uri(uri);
    mongoc_uri_destroy(uri);
    if (!mongodb->client)
    {
        free_mongodb(mongodb);
        return NULL;
    }

    mongoc_client_set_appname(mongodb->client, "libtvg");

    command = BCON_NEW("ping", BCON_INT32(1));
    if (!command)
    {
        free_mongodb(mongodb);
        return NULL;
    }

    success = mongoc_client_command_simple(mongodb->client, config->database, command, NULL, &reply, &error);
    bson_destroy(command);
    if (!success)
    {
        fprintf(stderr, "%s: Ping command failed: %s\n", __func__, error.message);
        free_mongodb(mongodb);
        return NULL;
    }

    if ((str = bson_as_canonical_extended_json(&reply, NULL)))
    {
        fprintf(stderr, "%s: Ping command returned: %s\n", __func__, str);
        bson_free(str);
    }

    bson_destroy(&reply);

    mongodb->articles = mongoc_client_get_collection(mongodb->client, config->database, config->col_articles);
    if (!mongodb->articles)
    {
        free_mongodb(mongodb);
        return NULL;
    }

    mongodb->entities = mongoc_client_get_collection(mongodb->client, config->database, config->col_entities);
    if (!mongodb->entities)
    {
        free_mongodb(mongodb);
        return NULL;
    }

    return mongodb;
}

struct mongodb *grab_mongodb(struct mongodb *mongodb)
{
    if (mongodb) mongodb->refcount++;
    return mongodb;
}

void free_mongodb(struct mongodb *mongodb)
{
    if (!mongodb) return;
    if (--mongodb->refcount) return;

    if (mongodb->articles)
        mongoc_collection_destroy(mongodb->articles);
    if (mongodb->entities)
        mongoc_collection_destroy(mongodb->entities);
    if (mongodb->client)
        mongoc_client_destroy(mongodb->client);

    free_mongodb_config(mongodb->config);
    free(mongodb);
}

static int bson_parse_integer(const bson_t *doc, const char *field, uint64_t *value)
{
    bson_iter_t iter;
    const char *str;
    uint32_t len;

    if (!bson_iter_init_find(&iter, doc, field))
        return 0;

    if (BSON_ITER_HOLDS_INT32(&iter))
    {
        int32_t v = bson_iter_int32(&iter);
        if (v < 0) return 0;
        *value = (uint64_t)v;
        return 1;
    }

    if (BSON_ITER_HOLDS_INT64(&iter))
    {
        int64_t v = bson_iter_int64(&iter);
        if (v < 0) return 0;
        *value = (uint64_t)v;
        return 1;
    }

    if (BSON_ITER_HOLDS_DATE_TIME(&iter))
    {
        int64_t v = bson_iter_date_time(&iter);
        if (v < 0) return 0;
        *value = (uint64_t)v;
        return 1;
    }

    if (BSON_ITER_HOLDS_UTF8(&iter))
    {
        str = bson_iter_utf8(&iter, &len);
        if (!bson_utf8_validate(str, len, false))
            return 0;

        for (; *str; str++)
            if (*str >= '0' && *str <= '9') break;

        if (!*str)
            return 0;

        *value = strtoul(str, (char **)&str, 0);

        for (; *str; str++)
            if (*str >= '0' && *str <= '9') return 0;

        return 1;
    }

    return 0;
}

/* Replacement for bson_iter_init_find_w_len, which only exists
 * in new versions and takes an int parameter instead of size_t. */
static int bson_iter_init_find_key(bson_iter_t *iter, const bson_t *bson, const char *key, size_t keylen)
{
    const char *ikey;

    if (!bson_iter_init(iter, bson))
        return 0;

    while (bson_iter_next(iter))
    {
        ikey = bson_iter_key(iter);
        if (!strncmp(key, ikey, keylen) && ikey[keylen] == '\0')
        {
            return 1;
        }
    }

    return 0;
}

static int bson_parse_entity(struct tvg *tvg, const bson_t *doc,
                             struct mongodb_config *config, uint64_t *value)
{
    struct node *other_node;
    struct node *node;
    bson_iter_t iter;
    const char *next;
    const char *key;
    const char *str;
    size_t keylen;
    uint32_t len;
    int ret = 0;

    if (!config->load_nodes)
        return bson_parse_integer(doc, config->entity_ent, value);

    if (!tvg)
        return 0;

    if (!(node = alloc_node()))
        return 0;

    key = config->entity_ent;
    for (;;)
    {
        next = strchr(key, ';');
        keylen = next ? (next - key) : strlen(key);

        if (!bson_iter_init_find_key(&iter, doc, key, keylen))
            goto skip;

        if (!BSON_ITER_HOLDS_UTF8(&iter))
            goto skip;

        str = bson_iter_utf8(&iter, &len);
        if (!bson_utf8_validate(str, len, false))
            goto error;

        if (!node_set_attribute_internal(node, key, keylen, str))
            goto error;

    skip:
        if (!next) break;
        key = next + 1;
    }

    if (!tvg_link_node(tvg, node, &other_node, ~0ULL))
    {
        free_node(node);
        if (!(node = other_node)) goto error;
    }

    *value = node->index;
    ret = 1;

error:
    free_node(node);
    return ret;
}

struct graph *mongodb_load_graph(struct tvg *tvg, struct mongodb *mongodb, uint64_t id, uint32_t flags)
{
    struct mongodb_config *config = mongodb->config;
    const struct occurrence *entry;
    struct occurrence new_entry;
    mongoc_cursor_t *cursor;
    bson_t *filter, *opts;
    struct graph *graph = NULL;
    struct queue *queue = NULL;
    bson_error_t error;
    const bson_t *doc;
    float weight;
    int ret = 0;
    size_t i;

    filter = BCON_NEW(config->entity_doc, BCON_INT64((int64_t)id));
    if (!filter)
        return NULL;

    opts = BCON_NEW("sort", "{", config->entity_sen, BCON_INT32(1), "}");
    if (!opts)
    {
        bson_destroy(filter);
        return NULL;
    }

    cursor = mongoc_collection_find_with_opts(mongodb->entities, filter, opts, NULL);
    bson_destroy(filter);
    bson_destroy(opts);
    if (!cursor)
        return NULL;

    if (!(graph = alloc_graph(flags)))
    {
        fprintf(stderr, "%s: Out of memory!\n", __func__);
        goto error;
    }

    if (!(queue = alloc_queue(sizeof(struct occurrence))))
    {
        fprintf(stderr, "%s: Out of memory!\n", __func__);
        goto error;
    }

    while (mongoc_cursor_next(cursor, &doc))
    {
        if (!bson_parse_integer(doc, config->entity_sen, &new_entry.sen))
        {
            fprintf(stderr, "%s: %s field not found or not an integer\n", __func__, config->entity_sen);
            continue;
        }
        if (!bson_parse_entity(tvg, doc, config, &new_entry.ent))
        {
            fprintf(stderr, "%s: %s field not found or not the correct type\n", __func__, config->entity_ent);
            continue;
        }

        while ((entry = (const struct occurrence *)queue_ptr(queue, 0)))
        {
            if (new_entry.sen - entry->sen <= config->max_distance) break;
            queue_get(queue, NULL);
        }

        for (i = 0; (entry = (const struct occurrence *)queue_ptr(queue, i)); i++)
        {
            if (entry->ent == new_entry.ent)
                continue;  /* never create self-loops */
            if (entry->sen > new_entry.sen)
                continue;  /* unexpected, entries are not sorted! */

            weight = (float)exp(-(double)(new_entry.sen - entry->sen));
            if (!graph_add_edge(graph, entry->ent, new_entry.ent, weight))
            {
                fprintf(stderr, "%s: Out of memory!\n", __func__);
                goto error;
            }
        }

        if (!queue_put(queue, &new_entry))
        {
            fprintf(stderr, "%s: Out of memory!\n", __func__);
            goto error;
        }
    }

    if (mongoc_cursor_error(cursor, &error))
    {
        fprintf(stderr, "%s: Query failed: %s\n", __func__, error.message);
        goto error;
    }

    /* Success if we didn't encounter any error. */
    ret = 1;

error:
    mongoc_cursor_destroy(cursor);
    if (graph) graph->revision = 0;
    if (!ret)
    {
        free_graph(graph);
        graph = NULL;
    }
    free_queue(queue);
    return graph;
}

int tvg_load_graphs_from_mongodb(struct tvg *tvg, struct mongodb *mongodb)
{
    struct mongodb_config *config = mongodb->config;
    bson_t *opts, filter = BSON_INITIALIZER;
    mongoc_cursor_t *cursor;
    uint32_t graph_flags;
    struct graph *graph;
    bson_error_t error;
    const bson_t *doc;
    uint64_t id, ts;
    int ret = 0;

    opts = BCON_NEW("sort", "{", config->article_time, BCON_INT32(1),
                                 config->article_id,   BCON_INT32(1), "}");
    if (!opts)
    {
        bson_destroy(&filter);
        return 0;
    }

    cursor = mongoc_collection_find_with_opts(mongodb->articles, &filter, opts, NULL);
    bson_destroy(&filter);
    bson_destroy(opts);
    if (!cursor)
        return 0;

    graph_flags = tvg->flags & (TVG_FLAGS_NONZERO |
                                TVG_FLAGS_POSITIVE |
                                TVG_FLAGS_DIRECTED);

    while (mongoc_cursor_next(cursor, &doc))
    {
        if (!bson_parse_integer(doc, config->article_id, &id))
        {
            fprintf(stderr, "%s: %s field not found or not an integer\n", __func__, config->article_id);
            continue;
        }
        if (!bson_parse_integer(doc, config->article_time, &ts))
        {
            fprintf(stderr, "%s: %s field not found or not an integer\n", __func__, config->article_time);
            continue;
        }

        if (!(graph = mongodb_load_graph(tvg, mongodb, id, graph_flags)))
        {
            fprintf(stderr, "%s: Failed to load document %llu\n", __func__, (long long unsigned int)id);
            goto error;
        }

        graph->id = id;
        if (!tvg_link_graph(tvg, graph, ts))
        {
            free_graph(graph);
            goto error;
        }

        free_graph(graph);
    }

    if (mongoc_cursor_error(cursor, &error))
    {
        fprintf(stderr, "%s: Query failed: %s\n", __func__, error.message);
        goto error;
    }

    /* Success if we didn't encounter any error. */
    ret = 1;

error:
    mongoc_cursor_destroy(cursor);
    return ret;
}

static void tvg_load_batch_from_mongodb(struct tvg *tvg, struct graph *other_graph,
                                        bson_t *filter, int direction, int jump)
{
    struct mongodb *mongodb = tvg->mongodb;
    struct mongodb_config *config = mongodb->config;
    struct graph *next_graph;
    mongoc_cursor_t *cursor;
    uint64_t cache_reserve = 0;
    uint32_t graph_flags;
    struct graph *graph;
    bson_error_t error;
    const bson_t *doc;
    struct list todo;
    uint64_t id, ts;
    uint64_t count;
    bson_t *opts;
    int first;
    int res;

    assert(direction == 1 || direction == -1);

    if (!other_graph)
        other_graph = LIST_ENTRY(&tvg->graphs, struct graph, entry);

    opts = BCON_NEW("sort", "{", config->article_time, BCON_INT32(direction),
                                 config->article_id,   BCON_INT32(direction), "}",
                    "limit", BCON_INT64(tvg->batch_size));
    if (!opts)
    {
        fprintf(stderr, "%s: Out of memory!\n", __func__);
        return;
    }

    cursor = mongoc_collection_find_with_opts(mongodb->articles, filter, opts, NULL);
    bson_destroy(opts);
    if (!cursor)
    {
        fprintf(stderr, "%s: Out of memory!\n", __func__);
        return;
    }

    graph_flags = tvg->flags & (TVG_FLAGS_NONZERO |
                                TVG_FLAGS_POSITIVE |
                                TVG_FLAGS_DIRECTED);

    list_init(&todo);
    for (count = 0; mongoc_cursor_next(cursor, &doc); count++)
    {
        if (!bson_parse_integer(doc, config->article_id, &id))
        {
            fprintf(stderr, "%s: %s field not found or not an integer\n", __func__, config->article_id);
            continue;
        }
        if (!bson_parse_integer(doc, config->article_time, &ts))
        {
            fprintf(stderr, "%s: %s field not found or not an integer\n", __func__, config->article_time);
            continue;
        }

        res = 1;
        if (direction > 0)
        {
            first = 1;
            while (&other_graph->entry != &tvg->graphs)
            {
                if ((res = compare_graph_ts_id(other_graph, ts, id)) >= 0) break;
                if (!jump)
                {
                    if (!first) other_graph->flags &= ~TVG_FLAGS_LOAD_PREV;
                    other_graph->flags &= ~TVG_FLAGS_LOAD_NEXT;
                }
                other_graph = LIST_ENTRY(other_graph->entry.next, struct graph, entry);
                first = 0;
            }
        }
        else
        {
            first = 1;
            while (&other_graph->entry != &tvg->graphs)
            {
                if ((res = compare_graph_ts_id(other_graph, ts, id)) <= 0) break;
                if (!jump)
                {
                    if (!first) other_graph->flags &= ~TVG_FLAGS_LOAD_NEXT;
                    other_graph->flags &= ~TVG_FLAGS_LOAD_PREV;
                }
                other_graph = LIST_ENTRY(other_graph->entry.prev, struct graph, entry);
                first = 0;
            }
        }

        if (!res)
        {
            assert(&other_graph->entry != &tvg->graphs);
            if (!jump && !first)
            {
                if (direction > 0)
                    other_graph->flags &= ~TVG_FLAGS_LOAD_PREV;
                else
                    other_graph->flags &= ~TVG_FLAGS_LOAD_NEXT;
            }

            assert(other_graph->cache != 0);
            list_remove(&other_graph->cache_entry);
            list_add_head(&todo, &other_graph->cache_entry);
            tvg->cache_used -= other_graph->cache;
            cache_reserve += other_graph->cache;

            jump = 0;
            continue;
        }

        if (!(graph = mongodb_load_graph(tvg, mongodb, id, graph_flags)))
        {
            fprintf(stderr, "%s: Failed to load document %llu\n", __func__, (long long unsigned int)id);
            goto error;
        }

        /* Keep in sync with tvg_link_graph! */
        assert(!graph->tvg);
        graph->ts  = ts;
        graph->id  = id;
        graph->tvg = tvg;

        if (direction > 0)
        {
            if (jump) graph->flags |= TVG_FLAGS_LOAD_PREV;
            graph->flags |= TVG_FLAGS_LOAD_NEXT;  /* will be cleared later */
            if (&other_graph->entry != &tvg->graphs)
                assert(compare_graph_ts_id(other_graph, ts, id) > 0);
            list_add_before(&other_graph->entry, &graph->entry);
        }
        else
        {
            if (jump) graph->flags |= TVG_FLAGS_LOAD_NEXT;
            graph->flags |= TVG_FLAGS_LOAD_PREV;  /* will be cleared later */
            if (&other_graph->entry != &tvg->graphs)
                assert(compare_graph_ts_id(other_graph, ts, id) < 0);
            list_add_after(&other_graph->entry, &graph->entry);
        }

        graph->cache = graph_memory_usage(graph);
        list_add_head(&todo, &graph->cache_entry);
        cache_reserve += graph->cache;

        other_graph = graph;
        jump = 0;
    }

    if (mongoc_cursor_error(cursor, &error))
    {
        fprintf(stderr, "%s: Query failed: %s\n", __func__, error.message);
        goto error;
    }

    if (count < tvg->batch_size)  /* end of data stream */
    {
        if (direction > 0)
        {
            first = 1;
            while (&other_graph->entry != &tvg->graphs)
            {
                if (!first) other_graph->flags &= ~TVG_FLAGS_LOAD_PREV;
                other_graph->flags &= ~TVG_FLAGS_LOAD_NEXT;
                other_graph = LIST_ENTRY(other_graph->entry.next, struct graph, entry);
                first = 0;
            }
        }
        else
        {
            first = 1;
            while (&other_graph->entry != &tvg->graphs)
            {
                if (!first) other_graph->flags &= ~TVG_FLAGS_LOAD_NEXT;
                other_graph->flags &= ~TVG_FLAGS_LOAD_PREV;
                other_graph = LIST_ENTRY(other_graph->entry.prev, struct graph, entry);
                first = 0;
            }
        }
    }

error:
    LIST_FOR_EACH_SAFE(graph, next_graph, &tvg->cache, struct graph, cache_entry)
    {
        assert(graph->tvg == tvg);
        assert(graph->cache != 0);
        if (tvg->cache_used + cache_reserve <= tvg->cache_size) break;
        if (graph->refcount > 1) continue;  /* cannot free space */
        unlink_graph(graph);
        free_graph(graph);
    }

    LIST_FOR_EACH_SAFE(graph, next_graph, &todo, struct graph, cache_entry)
    {
        assert(graph->tvg == tvg);
        assert(graph->cache != 0);
        list_remove(&graph->cache_entry);
        list_add_tail(&tvg->cache, &graph->cache_entry);
        tvg->cache_used += graph->cache;
    }

    mongoc_cursor_destroy(cursor);
}

void tvg_load_next_graph(struct tvg *tvg, struct graph *graph)
{
    struct mongodb_config *config;
    bson_t *filter;

    assert(&graph->entry != &tvg->graphs);
    if (!tvg->mongodb || (int64_t)graph->ts < 0)
    {
        graph->flags &= ~TVG_FLAGS_LOAD_NEXT;
        return;
    }

    config = tvg->mongodb->config;
    if ((int64_t)graph->id < 0)
    {
        filter = BCON_NEW(config->article_time, "{", "$gt", BCON_DATE_TIME((int64_t)graph->ts), "}");
    }
    else
    {
        filter = BCON_NEW("$or", "[", "{", config->article_time, "{", "$gt", BCON_DATE_TIME((int64_t)graph->ts), "}", "}",
                                      "{", config->article_time,             BCON_DATE_TIME((int64_t)graph->ts),
                                           config->article_id,   "{", "$gt", BCON_INT64((int64_t)graph->id), "}", "}", "]");
    }

    if (!filter)
    {
        fprintf(stderr, "%s: Out of memory!\n", __func__);
        return;
    }

    tvg_load_batch_from_mongodb(tvg, graph, filter, 1, 0);
    bson_destroy(filter);
}

void tvg_load_prev_graph(struct tvg *tvg, struct graph *graph)
{
    struct mongodb_config *config;
    bson_t *filter;

    assert(&graph->entry != &tvg->graphs);
    if (!tvg->mongodb || (int64_t)graph->ts < 0)
    {
        graph->flags &= ~TVG_FLAGS_LOAD_PREV;
        return;
    }

    config = tvg->mongodb->config;
    if ((int64_t)graph->id < 0)
    {
        filter = BCON_NEW(config->article_time, "{", "$lte", BCON_DATE_TIME((int64_t)graph->ts), "}");
    }
    else
    {
        filter = BCON_NEW("$or", "[", "{", config->article_time, "{", "$lt", BCON_DATE_TIME((int64_t)graph->ts), "}", "}",
                                      "{", config->article_time,             BCON_DATE_TIME((int64_t)graph->ts),
                                           config->article_id,   "{", "$lt", BCON_INT64((int64_t)graph->id), "}", "}", "]");
    }

    if (!filter)
    {
        fprintf(stderr, "%s: Out of memory!\n", __func__);
        return;
    }

    tvg_load_batch_from_mongodb(tvg, graph, filter, -1, 0);
    bson_destroy(filter);
}

void tvg_load_graphs_ge(struct tvg *tvg, struct graph *graph, uint64_t ts)
{
    struct mongodb_config *config;
    bson_t *filter;

    assert(!graph || graph->ts >= ts);
    if (!tvg->mongodb || (int64_t)ts < 0)
    {
        if (graph) graph->flags &= ~TVG_FLAGS_LOAD_PREV;
        return;
    }

    config = tvg->mongodb->config;
    filter = BCON_NEW(config->article_time, "{", "$gte", BCON_DATE_TIME((int64_t)ts), "}");

    if (!filter)
    {
        fprintf(stderr, "%s: Out of memory!\n", __func__);
        return;
    }

    tvg_load_batch_from_mongodb(tvg, graph, filter, 1, (ts > 0));
    bson_destroy(filter);
}

void tvg_load_graphs_le(struct tvg *tvg, struct graph *graph, uint64_t ts)
{
    struct mongodb_config *config;
    bson_t empty_filter = BSON_INITIALIZER;
    bson_t *filter = &empty_filter;

    assert(!graph || graph->ts <= ts);
    if (!tvg->mongodb)
    {
        if (graph) graph->flags &= ~TVG_FLAGS_LOAD_NEXT;
        return;
    }

    config = tvg->mongodb->config;
    if ((int64_t)(ts + 1) > 0)
    {
        filter = BCON_NEW(config->article_time, "{", "$lte", BCON_DATE_TIME((int64_t)ts), "}");
    }

    if (!filter)
    {
        fprintf(stderr, "%s: Out of memory!\n", __func__);
        return;
    }

    tvg_load_batch_from_mongodb(tvg, graph, filter, -1, ((int64_t)(ts + 1) > 0));
    if (filter != &empty_filter) bson_destroy(filter);
}

#else   /* HAVE_LIBMONGOC */

#include "tvg.h"
#include "internal.h"

struct mongodb *alloc_mongodb(const struct mongodb_config *config)
{
    fprintf(stderr, "%s: MongoDB support not compiled in!\n", __func__);
    return NULL;
}

struct mongodb *grab_mongodb(struct mongodb *mongodb)
{
    assert(!mongodb);
    return mongodb;
}

void free_mongodb(struct mongodb *mongodb)
{
    assert(!mongodb);
}

struct graph *mongodb_load_graph(struct tvg *tvg, struct mongodb *mongodb, uint64_t id, uint32_t flags)
{
    return NULL;
}

int tvg_load_graphs_from_mongodb(struct tvg *tvg, struct mongodb *mongodb)
{
    return 0;
}

void tvg_load_next_graph(struct tvg *tvg, struct graph *graph)
{
}

void tvg_load_prev_graph(struct tvg *tvg, struct graph *graph)
{
}

void tvg_load_graphs_ge(struct tvg *tvg, struct graph *graph, uint64_t ts)
{
}

void tvg_load_graphs_le(struct tvg *tvg, struct graph *graph, uint64_t ts)
{
}

#endif  /* HAVE_LIBMONGOC */
