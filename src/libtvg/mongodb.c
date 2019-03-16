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

#endif  /* HAVE_LIBMONGOC */
