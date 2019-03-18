/*
 * Time-varying graph library
 * MongoDB support.
 *
 * Copyright (c) 2019 Sebastian Lackner
 */

#ifdef HAVE_LIBMONGOC

#include <mongoc/mongoc.h>

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
    free(config->col_entities);
    free(config->doc_field);
    free(config->sen_field);
    free(config->ent_field);
    free(config);
}

static struct mongodb_config *alloc_mongodb_config(const struct mongodb_config *orig)
{
    struct mongodb_config *config;

    if (!orig->uri)          return NULL;
    if (!orig->database)     return NULL;
    if (!orig->col_articles) return NULL;
    if (!orig->col_entities) return NULL;
    if (!orig->doc_field)    return NULL;
    if (!orig->sen_field)    return NULL;
    if (!orig->ent_field)    return NULL;

    if (!(config = malloc(sizeof(*config))))
        return NULL;

    memset(config, 0, sizeof(*config));
    if (!(config->uri          = strdup(orig->uri)))          goto error;
    if (!(config->database     = strdup(orig->database)))     goto error;
    if (!(config->col_articles = strdup(orig->col_articles))) goto error;
    if (!(config->col_entities = strdup(orig->col_entities))) goto error;
    if (!(config->doc_field    = strdup(orig->doc_field)))    goto error;
    if (!(config->sen_field    = strdup(orig->sen_field)))    goto error;
    if (!(config->ent_field    = strdup(orig->ent_field)))    goto error;
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
        *value = bson_iter_int32(&iter);
        return 1;
    }

    if (BSON_ITER_HOLDS_INT64(&iter))
    {
        *value = bson_iter_int64(&iter);
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

int tvg_load_graph_from_mongodb(struct tvg *tvg, struct mongodb *mongodb, uint64_t document, float ts)
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

    filter = BCON_NEW(config->doc_field, BCON_INT64(document));
    if (!filter)
        return 0;

    opts = BCON_NEW("sort", "{", config->sen_field, BCON_INT32(1), "}");
    if (!opts)
    {
        bson_destroy(filter);
        return 0;
    }

    cursor = mongoc_collection_find_with_opts(mongodb->entities, filter, opts, NULL);
    bson_destroy(filter);
    bson_destroy(opts);
    if (!cursor)
        return 0;

    if (!(graph = tvg_alloc_graph(tvg, ts)))
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
        if (!bson_parse_integer(doc, config->sen_field, &new_entry.sen))
        {
            fprintf(stderr, "%s: %s field not found or not an integer\n", __func__, config->sen_field);
            continue;
        }
        if (!bson_parse_integer(doc, config->ent_field, &new_entry.ent))
        {
            fprintf(stderr, "%s: %s field not found or not an integer\n", __func__, config->ent_field);
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

            weight = exp((int64_t)(entry->sen - new_entry.sen));
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
    if (!ret) unlink_graph(graph);
    free_graph(graph);
    free_queue(queue);
    return ret;
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

int tvg_load_graph_from_mongodb(struct tvg *tvg, struct mongodb *mongodb, uint64_t document, float ts)
{
    return 0;
}

#endif  /* HAVE_LIBMONGOC */
