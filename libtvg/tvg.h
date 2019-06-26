/*
 * Time-varying graph library
 *
 * Copyright (c) 2018-2019 Sebastian Lackner
 */

#ifndef _TVG_H_
#define _TVG_H_

#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "list.h"
#include "tree.h"

#define LIBTVG_API_VERSION  0x00000008ULL

#define TVG_FLAGS_NONZERO   0x00000001U  /* weights are always nonzero */
#define TVG_FLAGS_POSITIVE  0x00000002U  /* weights are always positive */
#define TVG_FLAGS_DIRECTED  0x00000004U  /* edges are directed */
#define TVG_FLAGS_STREAMING 0x00000008U  /* streaming mode */

#define TVG_FLAGS_LOAD_NEXT 0x00010000U
#define TVG_FLAGS_LOAD_PREV 0x00020000U

#define OBJECTID_NONE   0
#define OBJECTID_INT    1
#define OBJECTID_OID    2

/* Note: In contrast to bson_oid_t, we internally swap the byte order, such
 * that comparison works the same for integers and MongoDB identifiers. */

struct objectid
{
    uint64_t    lo;
    uint32_t    hi;
    uint32_t    type;
};

struct event
{
    int value;
};

struct mutex
{
    int value;
};

struct rwlock
{
    struct mutex control;
    struct mutex write;
    int          readers;
};

struct entry1
{
    uint64_t    index;
    float       weight;
    uint32_t    reserved;
};

struct bucket1
{
    uint64_t    num_entries;
    uint64_t    max_entries;
    struct entry1 *entries;
    uint64_t    hint;
};

struct vector_ops;

struct vector
{
    uint64_t    refcount;
    uint32_t    flags;
    uint64_t    revision;
    float       eps;

    /* private: */
    struct query *query;

    const struct vector_ops *ops;
    uint32_t    bits;
    struct bucket1 *buckets;
    uint64_t    optimize;
};

struct entry2
{
    uint64_t    source;
    uint64_t    target;
    float       weight;
    uint32_t    reserved;
};

struct bucket2
{
    uint64_t    num_entries;
    uint64_t    max_entries;
    struct entry2 *entries;
    uint64_t    hint;
};

struct graph_ops;

struct graph
{
    uint64_t    refcount;
    uint32_t    flags;
    uint64_t    revision;
    float       eps;
    uint64_t    ts;
    struct objectid objectid;

    /* private: */
    struct query *query;

    struct tvg *tvg;         /* NULL for disconnected graphs */
    struct avl_entry entry;

    uint64_t    cache;       /* 0 if not cached, otherwise size of graph */
    struct list cache_entry;

    const struct graph_ops *ops;
    uint32_t    bits_source; /* 0...31 */
    uint32_t    bits_target; /* 0...31 */
    struct bucket2 *buckets;
    uint64_t    optimize;
};

struct attribute
{
    struct list entry;
    const char *key;
    const char *value;
    char        buffer[1];
};

struct node
{
    uint64_t    refcount;
    uint64_t    index;

    /* private: */
    struct tvg *tvg;         /* NULL for disconnected nodes */
    struct avl_entry entry_ind;
    struct avl_entry entry_key;
    struct list attributes;
};

struct tvg
{
    uint64_t    refcount;
    uint32_t    flags;
    int         verbosity;

    /* private: */
    struct avl_tree graphs;
    struct list queries;
    struct mongodb *mongodb;
    uint64_t    batch_size;

    struct avl_tree nodes_ind;  /* node by index */
    struct avl_tree nodes_key;  /* node by primary key */
    struct list primary_key;
    uint64_t    next_node;

    struct list graph_cache;
    uint64_t    graph_cache_used;
    uint64_t    graph_cache_size;

    struct list query_cache;
    uint64_t    query_cache_used;
    uint64_t    query_cache_size;
};

struct query_ops;

struct query
{
    /* private: */
    struct tvg *tvg;
    struct list entry;

    uint64_t    cache;       /* 0 if not cached, otherwise size of graph */
    struct list cache_entry;

    struct list todo_entry;

    const struct query_ops *ops;
    uint64_t    ts_min;
    uint64_t    ts_max;
};

struct query_sum_edges
{
    struct query base;
    struct graph *result;
    float       eps;
};

struct query_sum_edges_exp
{
    struct query base;
    struct graph *result;
    float       weight;
    float       log_beta;
    float       eps;
};

struct query_count_edges
{
    struct query base;
    struct graph *result;
};

struct query_count_nodes
{
    struct query base;
    struct vector *result;
};

struct mongodb_config
{
    char       *uri;
    char       *database;

    char       *col_articles;
    char       *article_id;     /* document id */
    char       *article_time;   /* time stamp */

    char       *filter_key;     /* article filter key */
    char       *filter_value;   /* article filter value */

    char       *col_entities;
    char       *entity_doc;     /* document id */
    char       *entity_sen;     /* sentence id */
    char       *entity_ent;     /* entity id */

    int         use_pool;       /* use connection pool */
    int         load_nodes;
    int         sum_weights;

    uint64_t    max_distance;
};

struct mongodb
{
    uint64_t    refcount;

    /* private: */
    struct mongodb_config *config;
    void       *client;     /* mongoc_client_t */
    void       *pool;       /* mongoc_client_pool_t */
};

struct bfs_entry
{
    double   weight;
    uint64_t count;
    uint64_t from;
    uint64_t to;
};

#define _UNIQUE_VARIABLE3(a, b) (a ## b)
#define _UNIQUE_VARIABLE2(a, b) _UNIQUE_VARIABLE3(a, b)
#define _UNIQUE_VARIABLE(a) _UNIQUE_VARIABLE2(a, __COUNTER__)

/* bucket1 macros */

static inline struct entry1 *__bucket1_for_each_entry(struct bucket1 *bucket, struct entry1 **entry)
{
    *entry = &bucket->entries[0];
    return &bucket->entries[bucket->num_entries];
}

#define _BUCKET1_FOR_EACH_ENTRY(_bucket, _entry, _end_entry) \
    for (struct entry1 *(_end_entry) = __bucket1_for_each_entry((_bucket), &(_entry)); \
         (_entry) != (_end_entry); (_entry)++)

#define BUCKET1_FOR_EACH_ENTRY(_bucket, _entry) \
    _BUCKET1_FOR_EACH_ENTRY((_bucket), (_entry), _UNIQUE_VARIABLE(__end_entry_))

static inline struct entry1 *__bucket1_for_each_entry_rev(struct bucket1 *bucket, struct entry1 **entry)
{
    *entry = &bucket->entries[bucket->num_entries];
    return &bucket->entries[0];
}

#define _BUCKET1_FOR_EACH_ENTRY_REV(_bucket, _entry, _end_entry) \
    for (struct entry1 *(_end_entry) = __bucket1_for_each_entry_rev((_bucket), &(_entry)); \
         (_entry)-- != (_end_entry);)

#define BUCKET1_FOR_EACH_ENTRY_REV(_bucket, _entry) \
    _BUCKET1_FOR_EACH_ENTRY_REV((_bucket), (_entry), _UNIQUE_VARIABLE(__end_entry_))

struct _bucket1_iter2
{
    struct entry1 *entry1;
    struct entry1 *end_entry1;

    struct entry1 *entry2;
    struct entry1 *end_entry2;
};

static inline struct _bucket1_iter2 __bucket1_for_each_entry2(struct bucket1 *bucket1, struct bucket1 *bucket2)
{
    struct _bucket1_iter2 iter;

    iter.end_entry1 = __bucket1_for_each_entry(bucket1, &iter.entry1);
    iter.end_entry2 = __bucket1_for_each_entry(bucket2, &iter.entry2);

    return iter;
}

static inline int __bucket1_next_entry2(struct _bucket1_iter2 *iter, struct entry1 **entry1, struct entry1 **entry2)
{
    if (iter->entry1 != iter->end_entry1)
    {
        if (iter->entry2 != iter->end_entry2)
        {
            if (iter->entry1->index < iter->entry2->index)
            {
                *entry1 = iter->entry1++;
                *entry2 = NULL;
            }
            else if (iter->entry1->index > iter->entry2->index)
            {
                *entry1 = NULL;
                *entry2 = iter->entry2++;
            }
            else
            {
                *entry1 = iter->entry1++;
                *entry2 = iter->entry2++;
            }
            return 1;
        }
        else
        {
            *entry1 = iter->entry1++;
            *entry2 = NULL;
            return 1;
        }
    }
    else
    {
        if (iter->entry2 != iter->end_entry2)
        {
            *entry1 = NULL;
            *entry2 = iter->entry2++;
            return 1;
        }
        else
        {
            return 0;
        }
    }
}

#define _BUCKET1_FOR_EACH_ENTRY2(_bucket1, _entry1, _bucket2, _entry2, _iter) \
    for (struct _bucket1_iter2 (_iter) = __bucket1_for_each_entry2((_bucket1), (_bucket2)); \
         __bucket1_next_entry2(&(_iter), &(_entry1), &(_entry2));)

#define BUCKET1_FOR_EACH_ENTRY2(_bucket1, _entry1, _bucket2, _entry2) \
    _BUCKET1_FOR_EACH_ENTRY2((_bucket1), (_entry1), (_bucket2), (_entry2), _UNIQUE_VARIABLE(__iter_))

static inline struct _bucket1_iter2 __bucket1_for_each_entry_rev2(struct bucket1 *bucket1, struct bucket1 *bucket2)
{
    struct _bucket1_iter2 iter;

    iter.end_entry1 = __bucket1_for_each_entry_rev(bucket1, &iter.entry1);
    iter.end_entry2 = __bucket1_for_each_entry_rev(bucket2, &iter.entry2);

    return iter;
}

static inline int __bucket1_prev_entry2(struct _bucket1_iter2 *iter, struct entry1 **entry1, struct entry1 **entry2)
{
    if (iter->entry1 != iter->end_entry1)
    {
        if (iter->entry2 != iter->end_entry2)
        {
            struct entry1 *prev_entry1 = &iter->entry1[-1];
            struct entry1 *prev_entry2 = &iter->entry2[-1];
            if (prev_entry1->index < prev_entry2->index)
            {
                *entry1 = NULL;
                *entry2 = --iter->entry2;
            }
            else if (prev_entry1->index > prev_entry2->index)
            {
                *entry1 = --iter->entry1;
                *entry2 = NULL;
            }
            else
            {
                *entry1 = --iter->entry1;
                *entry2 = --iter->entry2;
            }
            return 1;
        }
        else
        {
            *entry1 = --iter->entry1;
            *entry2 = NULL;
            return 1;
        }
    }
    else
    {
        if (iter->entry2 != iter->end_entry2)
        {
            *entry1 = NULL;
            *entry2 = --iter->entry2;
            return 1;
        }
        else
        {
            return 0;
        }
    }
}

#define _BUCKET1_FOR_EACH_ENTRY_REV2(_bucket1, _entry1, _bucket2, _entry2, _iter) \
    for (struct _bucket1_iter2 (_iter) = __bucket1_for_each_entry_rev2((_bucket1), (_bucket2)); \
         __bucket1_prev_entry2(&(_iter), &(_entry1), &(_entry2));)

#define BUCKET1_FOR_EACH_ENTRY_REV2(_bucket1, _entry1, _bucket2, _entry2) \
    _BUCKET1_FOR_EACH_ENTRY_REV2((_bucket1), (_entry1), (_bucket2), (_entry2), _UNIQUE_VARIABLE(__iter_))

/* bucket2 macros */

static inline struct entry2 *__bucket2_for_each_entry(struct bucket2 *bucket, struct entry2 **entry)
{
    *entry = &bucket->entries[0];
    return &bucket->entries[bucket->num_entries];
}

#define _BUCKET2_FOR_EACH_ENTRY(_bucket, _entry, _end_entry) \
    for (struct entry2 *(_end_entry) = __bucket2_for_each_entry((_bucket), &(_entry)); \
         (_entry) != (_end_entry); (_entry)++)

#define BUCKET2_FOR_EACH_ENTRY(_bucket, _entry) \
    _BUCKET2_FOR_EACH_ENTRY((_bucket), (_entry), _UNIQUE_VARIABLE(__end_entry_))

static inline struct entry2 *__bucket2_for_each_entry_rev(struct bucket2 *bucket, struct entry2 **entry)
{
    *entry = &bucket->entries[bucket->num_entries];
    return &bucket->entries[0];
}

#define _BUCKET2_FOR_EACH_ENTRY_REV(_bucket, _entry, _end_entry) \
    for (struct entry2 *(_end_entry) = __bucket2_for_each_entry_rev((_bucket), &(_entry)); \
         (_entry)-- != (_end_entry);)

#define BUCKET2_FOR_EACH_ENTRY_REV(_bucket, _entry) \
    _BUCKET2_FOR_EACH_ENTRY_REV((_bucket), (_entry), _UNIQUE_VARIABLE(__end_entry_))

struct _bucket2_iter2
{
    struct entry2 *entry1;
    struct entry2 *end_entry1;

    struct entry2 *entry2;
    struct entry2 *end_entry2;
};

static inline struct _bucket2_iter2 __bucket2_for_each_entry2(struct bucket2 *bucket1, struct bucket2 *bucket2)
{
    struct _bucket2_iter2 iter;

    iter.end_entry1 = __bucket2_for_each_entry(bucket1, &iter.entry1);
    iter.end_entry2 = __bucket2_for_each_entry(bucket2, &iter.entry2);

    return iter;
}

static inline int __bucket2_next_entry2(struct _bucket2_iter2 *iter, struct entry2 **entry1, struct entry2 **entry2)
{
    if (iter->entry1 != iter->end_entry1)
    {
        if (iter->entry2 != iter->end_entry2)
        {
            if (iter->entry1->target < iter->entry2->target)
            {
                *entry1 = iter->entry1++;
                *entry2 = NULL;
            }
            else if (iter->entry1->target > iter->entry2->target)
            {
                *entry1 = NULL;
                *entry2 = iter->entry2++;
            }
            else if (iter->entry1->source < iter->entry2->source)
            {
                *entry1 = iter->entry1++;
                *entry2 = NULL;
            }
            else if (iter->entry1->source > iter->entry2->source)
            {
                *entry1 = NULL;
                *entry2 = iter->entry2++;
            }
            else
            {
                *entry1 = iter->entry1++;
                *entry2 = iter->entry2++;
            }
            return 1;
        }
        else
        {
            *entry1 = iter->entry1++;
            *entry2 = NULL;
            return 1;
        }
    }
    else
    {
        if (iter->entry2 != iter->end_entry2)
        {
            *entry1 = NULL;
            *entry2 = iter->entry2++;
            return 1;
        }
        else
        {
            return 0;
        }
    }
}

#define _BUCKET2_FOR_EACH_ENTRY2(_bucket1, _entry1, _bucket2, _entry2, _iter) \
    for (struct _bucket2_iter2 (_iter) = __bucket2_for_each_entry2((_bucket1), (_bucket2)); \
         __bucket2_next_entry2(&(_iter), &(_entry1), &(_entry2));)

#define BUCKET2_FOR_EACH_ENTRY2(_bucket1, _entry1, _bucket2, _entry2) \
    _BUCKET2_FOR_EACH_ENTRY2((_bucket1), (_entry1), (_bucket2), (_entry2), _UNIQUE_VARIABLE(__iter_))

static inline struct _bucket2_iter2 __bucket2_for_each_entry_rev2(struct bucket2 *bucket1, struct bucket2 *bucket2)
{
    struct _bucket2_iter2 iter;

    iter.end_entry1 = __bucket2_for_each_entry_rev(bucket1, &iter.entry1);
    iter.end_entry2 = __bucket2_for_each_entry_rev(bucket2, &iter.entry2);

    return iter;
}

static inline int __bucket2_prev_entry2(struct _bucket2_iter2 *iter, struct entry2 **entry1, struct entry2 **entry2)
{
    if (iter->entry1 != iter->end_entry1)
    {
        if (iter->entry2 != iter->end_entry2)
        {
            struct entry2 *prev_entry1 = &iter->entry1[-1];
            struct entry2 *prev_entry2 = &iter->entry2[-1];
            if (prev_entry1->target < prev_entry2->target)
            {
                *entry1 = NULL;
                *entry2 = --iter->entry2;
            }
            else if (prev_entry1->target > prev_entry2->target)
            {
                *entry1 = --iter->entry1;
                *entry2 = NULL;
            }
            else if (prev_entry1->source < prev_entry2->source)
            {
                *entry1 = NULL;
                *entry2 = --iter->entry2;
            }
            else if (prev_entry1->source > prev_entry2->source)
            {
                *entry1 = --iter->entry1;
                *entry2 = NULL;
            }
            else
            {
                *entry1 = --iter->entry1;
                *entry2 = --iter->entry2;
            }
            return 1;
        }
        else
        {
            *entry1 = --iter->entry1;
            *entry2 = NULL;
            return 1;
        }
    }
    else
    {
        if (iter->entry2 != iter->end_entry2)
        {
            *entry1 = NULL;
            *entry2 = --iter->entry2;
            return 1;
        }
        else
        {
            return 0;
        }
    }
}

#define _BUCKET2_FOR_EACH_ENTRY_REV2(_bucket1, _entry1, _bucket2, _entry2, _iter) \
    for (struct _bucket2_iter2 (_iter) = __bucket2_for_each_entry_rev2((_bucket1), (_bucket2)); \
         __bucket2_prev_entry2(&(_iter), &(_entry1), &(_entry2));)

#define BUCKET2_FOR_EACH_ENTRY_REV2(_bucket1, _entry1, _bucket2, _entry2) \
    _BUCKET2_FOR_EACH_ENTRY_REV2((_bucket1), (_entry1), (_bucket2), (_entry2), _UNIQUE_VARIABLE(__iter_))

/* bucket1 + bucket2 macros */

struct _bucket21_iter
{
    struct entry2 *entry2;
    struct entry2 *end_entry2;

    struct entry1 *entry1;
    struct entry1 *end_entry1;
};

static inline struct _bucket21_iter __bucket21_for_each_entry(struct bucket2 *bucket2, struct bucket1 *bucket1)
{
    struct _bucket21_iter iter;

    iter.end_entry2 = __bucket2_for_each_entry(bucket2, &iter.entry2);
    iter.end_entry1 = __bucket1_for_each_entry(bucket1, &iter.entry1);

    return iter;
}

static inline int __bucket21_next_entry(struct _bucket21_iter *iter, struct entry2 **entry2, struct entry1 **entry1)
{
    if (iter->entry2 == iter->end_entry2)
        return 0;

    while (iter->entry1 != iter->end_entry1)
    {
        if (iter->entry1->index < iter->entry2->target)
        {
            iter->entry1++;
            continue;  /* skip entry1 */
        }
        else if (iter->entry1->index > iter->entry2->target)
        {
            *entry2 = iter->entry2++;
            *entry1 = NULL;
        }
        else
        {
            *entry2 = iter->entry2++;
            *entry1 = iter->entry1;
        }
        return 1;
    }

    *entry2 = iter->entry2++;
    *entry1 = NULL;
    return 1;
}

#define _BUCKET21_FOR_EACH_ENTRY(_bucket2, _entry2, _bucket1, _entry1, _iter) \
    for (struct _bucket21_iter (_iter) = __bucket21_for_each_entry((_bucket2), (_bucket1)); \
         __bucket21_next_entry(&(_iter), &(_entry2), &(_entry1));)

/* NOTE: This macro primarily iterates over bucket2: Entries in
 * bucket1 without corresponding entry in bucket2 are skipped! */
#define BUCKET21_FOR_EACH_ENTRY(_bucket2, _entry2, _bucket1, _entry1) \
    _BUCKET21_FOR_EACH_ENTRY((_bucket2), (_entry2), (_bucket1), (_entry1), _UNIQUE_VARIABLE(__iter_))

/* vector macros */

struct _vector_iter
{
    const struct vector *vector;
    uint64_t       index;
    uint64_t       end_index;
    struct entry1 *entry;
    struct entry1 *end_entry;
};

static inline struct _vector_iter __vector_for_each_entry(const struct vector *vector)
{
    struct _vector_iter iter;

    iter.vector    = vector;
    iter.index     = 0;
    iter.end_index = 1ULL << vector->bits;
    iter.end_entry = __bucket1_for_each_entry(&vector->buckets[0], &iter.entry);

    return iter;
}

static inline int __vector_next_entry(struct _vector_iter *iter, struct entry1 **entry)
{
    if (iter->index >= iter->end_index)
        return 0;

    for (;;)
    {
        if (iter->entry != iter->end_entry)
        {
            *entry = iter->entry++;
            return 1;
        }

        iter->index++;
        if (iter->index >= iter->end_index)
            break;

        iter->end_entry = __bucket1_for_each_entry(&iter->vector->buckets[iter->index], &iter->entry);
    }

    return 0;
}

#define _VECTOR_FOR_EACH_ENTRY(_vector, _edge, _iter) \
    for (struct _vector_iter (_iter) = __vector_for_each_entry((_vector)); __vector_next_entry(&(_iter), &(_edge));)

/* NOTE: Due to the internal bucket structure there is no guarantee about the sort order! */
#define VECTOR_FOR_EACH_ENTRY(_vector, _edge) \
    _VECTOR_FOR_EACH_ENTRY((_vector), (_edge), _UNIQUE_VARIABLE(__iter_))

struct _vector_iter2
{
    const struct vector *vector1;
    const struct vector *vector2;

    uint64_t       index;
    uint64_t       end_index_m1;  /* = mask */

    struct entry1 *entry1;
    struct entry1 *end_entry1;

    struct entry1 *entry2;
    struct entry1 *end_entry2;
};

static inline struct _vector_iter2 __vector_for_each_entry2(const struct vector *vector1, const struct vector *vector2)
{
    struct _vector_iter2 iter;

    iter.vector1    = vector1;
    iter.vector2    = vector2;

    iter.index = 0;
    if (vector1->bits > vector2->bits)
        iter.end_index_m1 = (1ULL << vector1->bits) - 1;
    else
        iter.end_index_m1 = (1ULL << vector2->bits) - 1;

    iter.end_entry1 = __bucket1_for_each_entry(&vector1->buckets[0], &iter.entry1);
    iter.end_entry2 = __bucket1_for_each_entry(&vector2->buckets[0], &iter.entry2);
    return iter;
}

static inline int __vector_next_entry2(struct _vector_iter2 *iter, struct entry1 **entry1, struct entry1 **entry2)
{
    if (iter->index > iter->end_index_m1)
        return 0;

    for (;;)
    {
        if (iter->entry1 != iter->end_entry1)
        {
            if ((iter->entry1->index & iter->end_index_m1) != iter->index)
            {
                iter->entry1++;
                continue;
            }

            if (iter->entry2 != iter->end_entry2)
            {
                if ((iter->entry2->index & iter->end_index_m1) != iter->index)
                {
                    iter->entry2++;
                    continue;
                }

                if (iter->entry1->index < iter->entry2->index)
                {
                    *entry1 = iter->entry1++;
                    *entry2 = NULL;
                }
                else if (iter->entry1->index > iter->entry2->index)
                {
                    *entry1 = NULL;
                    *entry2 = iter->entry2++;
                }
                else
                {
                    *entry1 = iter->entry1++;
                    *entry2 = iter->entry2++;
                }
                return 1;
            }
            else
            {
                *entry1 = iter->entry1++;
                *entry2 = NULL;
                return 1;
            }
        }
        else
        {
            if (iter->entry2 != iter->end_entry2)
            {
                if ((iter->entry2->index & iter->end_index_m1) != iter->index)
                {
                    iter->entry2++;
                    continue;
                }

                *entry1 = NULL;
                *entry2 = iter->entry2++;
                return 1;
            }
        }

        iter->index++;
        if (iter->index > iter->end_index_m1)
            break;

        iter->end_entry1 = __bucket1_for_each_entry(&iter->vector1->buckets[iter->index &
            ((1ULL << iter->vector1->bits) - 1)], &iter->entry1);
        iter->end_entry2 = __bucket1_for_each_entry(&iter->vector2->buckets[iter->index &
            ((1ULL << iter->vector2->bits) - 1)], &iter->entry2);
    }

    return 0;
}

#define _VECTOR_FOR_EACH_ENTRY2(_vector1, _edge1, _vector2, _edge2, _iter) \
    for (struct _vector_iter2 (_iter) = __vector_for_each_entry2((_vector1), (_vector2)); __vector_next_entry2(&(_iter), &(_edge1), &(_edge2));)

/* NOTE: Due to the internal bucket structure there is no guarantee about the sort order! */
#define VECTOR_FOR_EACH_ENTRY2(_vector1, _edge1, _vector2, _edge2) \
    _VECTOR_FOR_EACH_ENTRY2((_vector1), (_edge1), (_vector2), (_edge2), _UNIQUE_VARIABLE(__iter_))

/* graph macros */

struct _graph_iter
{
    const struct graph *graph;
    uint64_t       index;
    uint64_t       end_index;
    struct entry2 *edge;
    struct entry2 *end_edge;
};

static inline struct _graph_iter __graph_for_each_edge(const struct graph *graph)
{
    struct _graph_iter iter;

    iter.graph     = graph;
    iter.index     = 0;
    iter.end_index = 1ULL << (graph->bits_source + graph->bits_target);
    iter.end_edge  = __bucket2_for_each_entry(&graph->buckets[iter.index], &iter.edge);

    return iter;
}

static inline int __graph_next_directed_edge(struct _graph_iter *iter, struct entry2 **edge)
{
    if (iter->index >= iter->end_index)
        return 0;

    for (;;)
    {
        if (iter->edge != iter->end_edge)
        {
            *edge = iter->edge++;
            return 1;
        }
        iter->index++;

        if (iter->index >= iter->end_index)
            break;

        iter->end_edge = __bucket2_for_each_entry(&iter->graph->buckets[iter->index], &iter->edge);
    }

    return 0;
}

#define _GRAPH_FOR_EACH_DIRECTED_EDGE(_graph, _edge, _iter) \
    for (struct _graph_iter (_iter) = __graph_for_each_edge((_graph)); __graph_next_directed_edge(&(_iter), &(_edge));)

/* NOTE: Due to the internal bucket structure there is no guarantee about the sort order! */
#define GRAPH_FOR_EACH_DIRECTED_EDGE(_graph, _edge) \
    _GRAPH_FOR_EACH_DIRECTED_EDGE((_graph), (_edge), _UNIQUE_VARIABLE(__iter_))

static inline int __graph_next_undirected_edge(struct _graph_iter *iter, struct entry2 **edge)
{
    if (iter->index >= iter->end_index)
        return 0;

    for (;;)
    {
        for (; iter->edge != iter->end_edge; iter->edge++)
        {
            if (iter->edge->target >= iter->edge->source)
            {
                *edge = iter->edge++;
                return 1;
            }
        }
        iter->index++;

        if (iter->index >= iter->end_index)
            break;

        iter->end_edge = __bucket2_for_each_entry(&iter->graph->buckets[iter->index], &iter->edge);
    }

    return 0;
}

#define _GRAPH_FOR_EACH_UNDIRECTED_EDGE(_graph, _edge, _iter) \
    for (struct _graph_iter (_iter) = __graph_for_each_edge((_graph)); __graph_next_undirected_edge(&(_iter), &(_edge));)

/* NOTE: Due to the internal bucket structure there is no guarantee about the sort order! */
#define GRAPH_FOR_EACH_UNDIRECTED_EDGE(_graph, _edge) \
    _GRAPH_FOR_EACH_UNDIRECTED_EDGE((_graph), (_edge), _UNIQUE_VARIABLE(__iter_))

static inline int __graph_next_edge(struct _graph_iter *iter, struct entry2 **edge)
{
    if (iter->graph->flags & TVG_FLAGS_DIRECTED)
        return __graph_next_directed_edge(iter, edge);
    else
        return __graph_next_undirected_edge(iter, edge);
}

#define _GRAPH_FOR_EACH_EDGE(_graph, _edge, _iter) \
    for (struct _graph_iter (_iter) = __graph_for_each_edge((_graph)); __graph_next_edge(&(_iter), &(_edge));)

/* NOTE: Due to the internal bucket structure there is no guarantee about the sort order! */
#define GRAPH_FOR_EACH_EDGE(_graph, _edge) \
    _GRAPH_FOR_EACH_EDGE((_graph), (_edge), _UNIQUE_VARIABLE(__iter_))

struct _graph_adjacent_iter
{
    const struct graph *graph;
    uint64_t       index;
    uint64_t       end_index;
    struct entry2 *edge;
    struct entry2 *end_edge;
    uint64_t       source;
};

static inline struct _graph_adjacent_iter __graph_for_each_adjacent_edge(const struct graph *graph, uint64_t source)
{
    struct _graph_adjacent_iter iter;

    iter.graph     = graph;
    iter.index     = source & ((1ULL << graph->bits_source) - 1);
    iter.end_index = 1ULL << (graph->bits_source + graph->bits_target);
    iter.end_edge  = __bucket2_for_each_entry(&graph->buckets[iter.index], &iter.edge);
    iter.source    = source;

    return iter;
}

static inline int __graph_next_adjacent_edge(struct _graph_adjacent_iter *iter, struct entry2 **edge)
{
    if (iter->index >= iter->end_index)
        return 0;

    for (;;)
    {
        for (; iter->edge != iter->end_edge; iter->edge++)
        {
            if (iter->edge->source == iter->source)
            {
                *edge = iter->edge++;
                return 1;
            }
        }
        iter->index += (1ULL << iter->graph->bits_source);

        if (iter->index >= iter->end_index)
            break;

        iter->end_edge = __bucket2_for_each_entry(&iter->graph->buckets[iter->index], &iter->edge);
    }

    return 0;
}

#define _GRAPH_FOR_EACH_ADJACENT_EDGE(_graph, _source, _edge, _iter) \
    for (struct _graph_adjacent_iter (_iter) = __graph_for_each_adjacent_edge((_graph), (_source)); \
         __graph_next_adjacent_edge(&(_iter), &(_edge));)

/* NOTE: Due to the internal bucket structure there is no guarantee about the sort order! */
#define GRAPH_FOR_EACH_ADJACENT_EDGE(_graph, _source, _edge) \
    _GRAPH_FOR_EACH_ADJACENT_EDGE((_graph), (_source), (_edge), _UNIQUE_VARIABLE(__iter_))

/* graph + vector macros */

struct _graph_vector_iter
{
    const struct graph *graph;
    const struct vector *vector;

    uint64_t       index;
    uint64_t       mask;
    uint64_t       end_index;

    struct entry2 *entry2;
    struct entry2 *end_entry2;

    struct entry1 *entry1;
    struct entry1 *end_entry1;
};

static inline struct _graph_vector_iter __graph_vector_for_each_edge(const struct graph *graph, const struct vector *vector)
{
    struct _graph_vector_iter iter;

    iter.graph      = graph;
    iter.vector     = vector;

    iter.index = 0;
    if (graph->bits_target > vector->bits)
    {
        iter.mask       = (1ULL << graph->bits_target) - 1;
        iter.end_index  = 1ULL << (graph->bits_source + graph->bits_target);
    }
    else
    {
        iter.mask       = (1ULL << vector->bits) - 1;
        iter.end_index  = 1ULL << (graph->bits_source + vector->bits);
    }

    iter.end_entry2 = __bucket2_for_each_entry(&graph->buckets[0],  &iter.entry2);
    iter.end_entry1 = __bucket1_for_each_entry(&vector->buckets[0], &iter.entry1);

    return iter;
}

static inline int __graph_vector_next_edge(struct _graph_vector_iter *iter, struct entry2 **entry2, struct entry1 **entry1)
{
    uint64_t target;

    if (iter->index >= iter->end_index)
        return 0;

    target = iter->index >> iter->graph->bits_source;
    for (;;)
    {
        if (iter->entry2 != iter->end_entry2)
        {
            if ((iter->entry2->target & iter->mask) != target)
            {
                iter->entry2++;
                continue;
            }

            while (iter->entry1 != iter->end_entry1)
            {
                if (iter->entry1->index < iter->entry2->target)
                {
                    iter->entry1++;
                    continue;  /* skip entry1 */
                }
                else if (iter->entry1->index > iter->entry2->target)
                {
                    *entry2 = iter->entry2++;
                    *entry1 = NULL;
                }
                else
                {
                    *entry2 = iter->entry2++;
                    *entry1 = iter->entry1;
                }
                return 1;
            }

            *entry2 = iter->entry2++;
            *entry1 = NULL;
            return 1;
        }

        iter->index++;
        if (iter->index >= iter->end_index)
            break;

        target = iter->index >> iter->graph->bits_source;
        iter->end_entry2 = __bucket2_for_each_entry(&iter->graph->buckets[iter->index &
            ((1ULL << (iter->graph->bits_source + iter->graph->bits_target)) - 1)], &iter->entry2);
        iter->end_entry1 = __bucket1_for_each_entry(&iter->vector->buckets[target &
            ((1ULL << iter->vector->bits) - 1)], &iter->entry1);
    }

    return 0;
}

#define _GRAPH_VECTOR_FOR_EACH_EDGE(_graph, _entry2, _vector, _entry1, _iter) \
    for (struct _graph_vector_iter (_iter) = __graph_vector_for_each_edge((_graph), (_vector)); \
         __graph_vector_next_edge(&(_iter), &(_entry2), &(_entry1));)

/* NOTE: This macro primarily iterates over the graph: Entries in
 * the vector without corresponding entry in the graph are skipped! */
#define GRAPH_VECTOR_FOR_EACH_EDGE(_graph, _entry2, _vector, _entry1) \
    _GRAPH_VECTOR_FOR_EACH_EDGE((_graph), (_entry2), (_vector), (_entry1), _UNIQUE_VARIABLE(__iter_))

/* tvg macros */

/* FIXME: Avoid double declaration */
void free_graph(struct graph *graph);
struct graph *prev_graph(struct graph *graph);
struct graph *next_graph(struct graph *graph);

static inline int __tvg_enter_iteration(struct graph **graph)
{
    *graph = NULL;
    return 1;
}

static inline void __tvg_leave_iteration(struct graph **graph)
{
    free_graph(*graph);
    *graph = NULL;
}

static inline struct graph *__tvg_next_graph(struct graph *graph)
{
    struct graph *ret = next_graph(graph);
    free_graph(graph);
    return ret;
}

static inline struct graph *__tvg_prev_graph(struct graph *graph)
{
    struct graph *ret = prev_graph(graph);
    free_graph(graph);
    return ret;
}

#define _TVG_FOR_EACH_GRAPH_GE(_tvg, _graph, _ts, _once) \
    for (int (_once) = __tvg_enter_iteration(&(_graph)); (_once)--; __tvg_leave_iteration(&(_graph))) \
    for ((_graph) = tvg_lookup_graph_ge((_tvg), (_ts)); \
         (_graph) != NULL; (_graph) = __tvg_next_graph((_graph)))

/* NOTE: The caller must release the graph manually when using
 * goto / return within the iteration. */
#define TVG_FOR_EACH_GRAPH_GE(_tvg, _graph, _ts) \
    _TVG_FOR_EACH_GRAPH_GE((_tvg), (_graph), (_ts), _UNIQUE_VARIABLE(__once_))

#define _TVG_FOR_EACH_GRAPH_LE_REV(_tvg, _graph, _ts, _once) \
    for (int (_once) = __tvg_enter_iteration(&(_graph)); (_once)--; __tvg_leave_iteration(&(_graph))) \
    for ((_graph) = tvg_lookup_graph_le((_tvg), (_ts)); \
         (_graph) != NULL; (_graph) = __tvg_prev_graph((_graph)))

/* NOTE: The caller must release the graph manually when using
 * goto / return within the iteration. */
#define TVG_FOR_EACH_GRAPH_LE_REV(_tvg, _graph, _ts) \
    _TVG_FOR_EACH_GRAPH_LE_REV((_tvg), (_graph), (_ts), _UNIQUE_VARIABLE(__once_))

/* node macros */

struct _node_primary_attr_iter
{
    struct attribute *primary;
    struct attribute *end_primary;

    struct attribute *attr;
    struct attribute *end_attr;
};

static inline struct _node_primary_attr_iter __node_for_each_primary_attribute(struct tvg *tvg, struct node *node)
{
    struct _node_primary_attr_iter iter;

    iter.primary     = LIST_ENTRY(tvg->primary_key.next, struct attribute, entry);
    iter.end_primary = LIST_ENTRY(&tvg->primary_key, struct attribute, entry);

    iter.attr     = LIST_ENTRY(node->attributes.next, struct attribute, entry);
    iter.end_attr = LIST_ENTRY(&node->attributes, struct attribute, entry);

    return iter;
}

static inline int __node_next_primary_attribute(struct _node_primary_attr_iter *iter, struct attribute **attr)
{
    int res;

    if (iter->primary == iter->end_primary)
        return 0;

    *attr = NULL;

    for (; iter->attr != iter->end_attr;
         iter->attr = LIST_ENTRY(iter->attr->entry.next, struct attribute, entry))
    {
        res = strcmp(iter->primary->key, iter->attr->key);
        if (res < 0) break;
        if (!res)
        {
            *attr = iter->attr;
            iter->attr = LIST_ENTRY(iter->attr->entry.next, struct attribute, entry);
            break;
        }
    }

    iter->primary = LIST_ENTRY(iter->primary->entry.next, struct attribute, entry);
    return 1;
}

#define _NODE_FOR_EACH_PRIMARY_ATTRIBUTE(_tvg, _node, _attr, _iter) \
    for (struct _node_primary_attr_iter (_iter) = __node_for_each_primary_attribute((_tvg), (_node)); \
         __node_next_primary_attribute(&(_iter), &(_attr));)

#define NODE_FOR_EACH_PRIMARY_ATTRIBUTE(_tvg, _node, _attr) \
    _NODE_FOR_EACH_PRIMARY_ATTRIBUTE((_tvg), (_node), (_attr), _UNIQUE_VARIABLE(__iter_))

struct _node_primary_attr_iter2
{
    struct attribute *primary;
    struct attribute *end_primary;

    struct attribute *attr1;
    struct attribute *end_attr1;

    struct attribute *attr2;
    struct attribute *end_attr2;
};

static inline struct _node_primary_attr_iter2 __node_for_each_primary_attribute2(struct tvg *tvg, struct node *node1, struct node *node2)
{
    struct _node_primary_attr_iter2 iter;

    iter.primary     = LIST_ENTRY(tvg->primary_key.next, struct attribute, entry);
    iter.end_primary = LIST_ENTRY(&tvg->primary_key, struct attribute, entry);

    iter.attr1     = LIST_ENTRY(node1->attributes.next, struct attribute, entry);
    iter.end_attr1 = LIST_ENTRY(&node1->attributes, struct attribute, entry);

    iter.attr2     = LIST_ENTRY(node2->attributes.next, struct attribute, entry);
    iter.end_attr2 = LIST_ENTRY(&node2->attributes, struct attribute, entry);

    return iter;
}

static inline int __node_next_primary_attribute2(struct _node_primary_attr_iter2 *iter, struct attribute **attr1, struct attribute **attr2)
{
    int res;

    if (iter->primary == iter->end_primary)
        return 0;

    *attr1 = NULL;
    *attr2 = NULL;

    for (; iter->attr1 != iter->end_attr1;
         iter->attr1 = LIST_ENTRY(iter->attr1->entry.next, struct attribute, entry))
    {
        res = strcmp(iter->primary->key, iter->attr1->key);
        if (res < 0) break;
        if (!res)
        {
            *attr1 = iter->attr1;
            iter->attr1 = LIST_ENTRY(iter->attr1->entry.next, struct attribute, entry);
            break;
        }
    }

    for (; iter->attr2 != iter->end_attr2;
         iter->attr2 = LIST_ENTRY(iter->attr2->entry.next, struct attribute, entry))
    {
        res = strcmp(iter->primary->key, iter->attr2->key);
        if (res < 0) break;
        if (!res)
        {
            *attr2 = iter->attr2;
            iter->attr2 = LIST_ENTRY(iter->attr2->entry.next, struct attribute, entry);
            break;
        }
    }

    iter->primary = LIST_ENTRY(iter->primary->entry.next, struct attribute, entry);
    return 1;
}

#define _NODE_FOR_EACH_PRIMARY_ATTRIBUTE2(_tvg, _node1, _attr1, _node2, _attr2, _iter) \
    for (struct _node_primary_attr_iter2 (_iter) = __node_for_each_primary_attribute2((_tvg), (_node1), (_node2)); \
         __node_next_primary_attribute2(&(_iter), &(_attr1), &(_attr2));)

#define NODE_FOR_EACH_PRIMARY_ATTRIBUTE2(_tvg, _node1, _attr1, _node2, _attr2) \
    _NODE_FOR_EACH_PRIMARY_ATTRIBUTE2((_tvg), (_node1), (_attr1), (_node2), (_attr2), _UNIQUE_VARIABLE(__iter_))

/* init function */

int init_libtvg(uint64_t api_version);

/* vector functions */

struct vector *alloc_vector(uint32_t flags);
struct vector *grab_vector(struct vector *vector);
void free_vector(struct vector *vector);

struct vector *vector_duplicate(struct vector *source);

uint64_t vector_memory_usage(struct vector *vector);

int vector_inc_bits(struct vector *vector);
int vector_dec_bits(struct vector *vector);
void vector_optimize(struct vector *vector);

int vector_set_eps(struct vector *vector, float eps);

int vector_empty(struct vector *vector);
int vector_clear(struct vector *vector);

int vector_has_entry(struct vector *vector, uint64_t index);

float vector_get_entry(struct vector *vector, uint64_t index);
uint64_t vector_num_entries(struct vector *vector);
uint64_t vector_get_entries(struct vector *vector, uint64_t *indices, float *weights, uint64_t max_entries);

int vector_set_entry(struct vector *vector, uint64_t index, float weight);
int vector_set_entries(struct vector *vector, uint64_t *indices, float *weights, uint64_t num_entries);

int vector_add_entry(struct vector *vector, uint64_t index, float weight);
int vector_add_entries(struct vector *vector, uint64_t *indices, float *weights, uint64_t num_entries);
int vector_add_vector(struct vector *out, struct vector *vector, float weight);

int vector_sub_entry(struct vector *vector, uint64_t index, float weight);
int vector_sub_entries(struct vector *vector, uint64_t *indices, float *weights, uint64_t num_entries);
int vector_sub_vector(struct vector *out, struct vector *vector, float weight);

int vector_del_entry(struct vector *vector, uint64_t index);
int vector_del_entries(struct vector *vector, uint64_t *indices, uint64_t num_entries);

int vector_mul_const(struct vector *vector, float constant);
double vector_norm(const struct vector *vector);
double vector_mul_vector(const struct vector *vector1, const struct vector *vector2);
double vector_sub_vector_norm(const struct vector *vector1, const struct vector *vector2);

/* graph functions */

struct graph *alloc_graph(uint32_t flags);
struct graph *grab_graph(struct graph *graph);
void free_graph(struct graph *graph);
void unlink_graph(struct graph *graph);

struct graph *graph_duplicate(struct graph *source);

void graph_debug(struct graph *graph);
uint64_t graph_memory_usage(struct graph *graph);

struct graph *prev_graph(struct graph *graph);
struct graph *next_graph(struct graph *graph);

int graph_inc_bits_target(struct graph *graph);
int graph_dec_bits_target(struct graph *graph);
int graph_inc_bits_source(struct graph *graph);
int graph_dec_bits_source(struct graph *graph);
void graph_optimize(struct graph *graph);

int graph_set_eps(struct graph *graph, float eps);

int graph_empty(struct graph *graph);
int graph_clear(struct graph *graph);

int graph_has_edge(struct graph *graph, uint64_t source, uint64_t target);

float graph_get_edge(struct graph *graph, uint64_t source, uint64_t target);
uint64_t graph_num_edges(struct graph *graph);
uint64_t graph_get_edges(struct graph *graph, uint64_t *indices, float *weights, uint64_t max_edges);
uint64_t graph_get_top_edges(struct graph *graph, uint64_t *indices, float *weights, uint64_t max_edges);
uint64_t graph_get_adjacent_edges(struct graph *graph, uint64_t source, uint64_t *indices, float *weights, uint64_t max_edges);

int graph_set_edge(struct graph *graph, uint64_t source, uint64_t target, float weight);
int graph_set_edges(struct graph *graph, uint64_t *indices, float *weights, uint64_t num_edges);

int graph_add_edge(struct graph *graph, uint64_t source, uint64_t target, float weight);
int graph_add_edges(struct graph *graph, uint64_t *indices, float *weights, uint64_t num_edges);
int graph_add_graph(struct graph *out, struct graph *graph, float weight);

int graph_sub_edge(struct graph *graph, uint64_t source, uint64_t target, float weight);
int graph_sub_edges(struct graph *graph, uint64_t *indices, float *weights, uint64_t num_edges);
int graph_sub_graph(struct graph *out, struct graph *graph, float weight);

int graph_del_edge(struct graph *graph, uint64_t source, uint64_t target);
int graph_del_edges(struct graph *graph, uint64_t *indices, uint64_t num_edges);

int graph_mul_const(struct graph *graph, float constant);
struct vector *graph_mul_vector(const struct graph *graph, const struct vector *vector);

struct vector *graph_in_degrees(const struct graph *graph);
struct vector *graph_in_weights(const struct graph *graph);

struct vector *graph_out_degrees(const struct graph *graph);
struct vector *graph_out_weights(const struct graph *graph);

struct vector *graph_degree_anomalies(const struct graph *graph);
struct vector *graph_weight_anomalies(const struct graph *graph);

struct vector *graph_power_iteration(const struct graph *graph, struct vector *initial_guess,
                                     uint32_t num_iterations, double tolerance, double *ret_eigenvalue);

struct graph *graph_filter_nodes(const struct graph *graph, struct vector *nodes);

int graph_bfs(struct graph *g, uint64_t source, int use_weights, int (*callback)(struct graph *,
              const struct bfs_entry *, void *), void *userdata);

/* node functions */

struct node *alloc_node(void);
struct node *grab_node(struct node *node);
void free_node(struct node *node);
void unlink_node(struct node *node);

int node_set_attribute(struct node *node, const char *key, const char *value);
const char *node_get_attribute(struct node *node, const char *key);
char **node_get_attributes(struct node *node);

/* tvg functions */

struct tvg *alloc_tvg(uint32_t flags);
struct tvg *grab_tvg(struct tvg *tvg);
void free_tvg(struct tvg *tvg);

void tvg_set_verbosity(struct tvg *tvg, int verbosity);
void tvg_debug(struct tvg *tvg);
uint64_t tvg_memory_usage(struct tvg *tvg);

int tvg_link_graph(struct tvg *tvg, struct graph *graph, uint64_t ts);

int tvg_set_primary_key(struct tvg *tvg, const char *key);
int tvg_link_node(struct tvg *tvg, struct node *node, struct node **ret_node, uint64_t index);
struct node *tvg_get_node_by_index(struct tvg *tvg, uint64_t index);
struct node *tvg_get_node_by_primary_key(struct tvg *tvg, struct node *primary_key);

int tvg_load_graphs_from_file(struct tvg *tvg, const char *filename);
int tvg_load_nodes_from_file(struct tvg *tvg, const char *filename, const char *key);

int tvg_enable_mongodb_sync(struct tvg *tvg, struct mongodb *mongodb,
                            uint64_t batch_size, uint64_t cache_size);
void tvg_disable_mongodb_sync(struct tvg *tvg);

int tvg_enable_query_cache(struct tvg *tvg, uint64_t cache_size);
void tvg_disable_query_cache(struct tvg *tvg);

struct graph *tvg_lookup_graph_ge(struct tvg *tvg, uint64_t ts);
struct graph *tvg_lookup_graph_le(struct tvg *tvg, uint64_t ts);
struct graph *tvg_lookup_graph_near(struct tvg *tvg, uint64_t ts);

int tvg_compress(struct tvg *tvg, uint64_t step, uint64_t offset);

struct graph *tvg_extract(struct tvg *tvg, uint64_t ts, float (*weight_func)(struct graph *,
                          uint64_t, void *), void *userdata);

void tvg_invalidate_queries(struct tvg *tvg, uint64_t ts_min, uint64_t ts_max);

/* Query functions */

struct graph *tvg_sum_edges(struct tvg *tvg, uint64_t ts_min, uint64_t ts_max, float eps);
struct graph *tvg_sum_edges_exp(struct tvg *tvg, uint64_t ts_min, uint64_t ts_max,
                                       float weight, float log_beta, float eps);
struct graph *tvg_count_edges(struct tvg *tvg, uint64_t ts_min, uint64_t ts_max);
struct vector *tvg_count_nodes(struct tvg *tvg, uint64_t ts_min, uint64_t ts_max);
struct graph *tvg_topics(struct tvg *tvg, uint64_t ts_min, uint64_t ts_max);

/* MongoDB functions */

struct mongodb *alloc_mongodb(const struct mongodb_config *config);
struct mongodb *grab_mongodb(struct mongodb *mongodb);
void free_mongodb(struct mongodb *mongodb);

struct graph *mongodb_load_graph(struct tvg *tvg, struct mongodb *mongodb, struct objectid *objectid, uint32_t flags);
int tvg_load_graphs_from_mongodb(struct tvg *tvg, struct mongodb *mongodb);

#endif /* _TVG_H_ */
