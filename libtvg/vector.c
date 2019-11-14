/*
 * Time-varying graph library
 * Vector functions.
 *
 * Copyright (c) 2018-2019 Sebastian Lackner
 */

#include "internal.h"

static const struct vector_ops *get_vector_ops(uint32_t flags)
{
    if (flags & TVG_FLAGS_POSITIVE)
        return &vector_positive_ops;
    else if (flags & TVG_FLAGS_NONZERO)
        return &vector_nonzero_ops;
    else
        return &vector_generic_ops;
}

struct vector *alloc_vector(uint32_t flags)
{
    static const uint32_t bits = 0;
    struct vector *vector;
    struct bucket1 *buckets;
    uint64_t i, num_buckets;

    if (flags & ~(TVG_FLAGS_NONZERO |
                  TVG_FLAGS_POSITIVE))
        return NULL;

    if (flags & TVG_FLAGS_POSITIVE)
        flags |= TVG_FLAGS_NONZERO;  /* positive implies nonzero */

    num_buckets = 1ULL << bits;
    if (!(buckets = malloc(sizeof(*buckets) * num_buckets)))
        return NULL;

    for (i = 0; i < num_buckets; i++)
        init_bucket1(&buckets[i]);

    if (!(vector = malloc(sizeof(*vector))))
    {
        free(buckets);
        return NULL;
    }

    vector->refcount = 1;
    vector->flags    = flags;
    vector->revision = 0;
    vector->eps      = 0.0;
    vector->query    = NULL;
    vector->ops      = get_vector_ops(flags);
    vector->readonly = 0;
    vector->bits     = bits;
    vector->buckets  = buckets;
    vector->optimize = 0;

    /* set a proper 'optimize' value */
    vector_optimize(vector);
    return vector;
}

struct vector *grab_vector(struct vector *vector)
{
    if (vector) __sync_fetch_and_add(&vector->refcount, 1);
    return vector;
}

void free_vector(struct vector *vector)
{
    uint64_t i, num_buckets;

    if (!vector) return;
    if (__sync_sub_and_fetch(&vector->refcount, 1)) return;

    num_buckets = 1ULL << vector->bits;
    for (i = 0; i < num_buckets; i++)
        free_bucket1(&vector->buckets[i]);

    free_query(vector->query);
    free(vector->buckets);
    free(vector);
}

struct vector *vector_duplicate(struct vector *source)
{
    struct vector *vector;
    struct bucket1 *buckets;
    uint64_t i, num_buckets;

    num_buckets = 1ULL << source->bits;
    if (!(buckets = malloc(sizeof(*buckets) * num_buckets)))
        return NULL;

    for (i = 0; i < num_buckets; i++)
    {
        if (!init_bucket1_from(&buckets[i], &source->buckets[i]))
        {
            while (i--)
                free_bucket1(&buckets[i]);

            return NULL;
        }
    }

    if (!(vector = malloc(sizeof(*vector))))
    {
        for (i = 0; i < num_buckets; i++)
            free_bucket1(&buckets[i]);

        free(buckets);
        return NULL;
    }

    vector->refcount = 1;
    vector->flags    = source->flags;
    vector->revision = source->revision;
    vector->eps      = source->eps;
    vector->query    = NULL;
    vector->ops      = get_vector_ops(vector->flags);
    vector->readonly = 0;
    vector->bits     = source->bits;
    vector->buckets  = buckets;
    vector->optimize = source->optimize;
    return vector;
}

uint64_t vector_memory_usage(struct vector *vector)
{
    uint64_t i, num_buckets;
    struct bucket1 *bucket;
    uint64_t size = sizeof(*vector);

    /* In the following, we underestimate the memory usage a bit, since
     * we do not take into account the heap structure itself. */

    num_buckets = 1ULL << vector->bits;
    size += sizeof(*bucket) * num_buckets;

    for (i = 0; i < num_buckets; i++)
    {
        bucket = &vector->buckets[i];
        size += sizeof(bucket->entries) * bucket->max_entries;
    }

    return size;
}

int vector_inc_bits(struct vector *vector)
{
    struct bucket1 *buckets;
    uint64_t i, num_buckets;
    uint64_t mask = 1ULL << vector->bits;

    if (vector->bits >= 31)
        return 0;

    num_buckets = 1ULL << vector->bits;
    if (!(buckets = realloc(vector->buckets, sizeof(*buckets) * 2 * num_buckets)))
        return 0;

    vector->buckets = buckets;

    for (i = 0; i < num_buckets; i++)
    {
        init_bucket1(&buckets[i + num_buckets]);
        if (!bucket1_split(&buckets[i], &buckets[i + num_buckets], mask))
        {
            /* FIXME: Error handling is mostly untested. */

            while (i--)
            {
                bucket1_merge(&buckets[i], &buckets[i + num_buckets]);
                free_bucket1(&buckets[i + num_buckets]);
            }

            if ((buckets = realloc(vector->buckets, sizeof(*buckets) * num_buckets)))
                vector->buckets = buckets;

            return 0;
        }
    }

    for (i = 0; i < 2 * num_buckets; i++)
        bucket1_compress(&buckets[i]);

    vector->bits++;
    return 1;
}

int vector_dec_bits(struct vector *vector)
{
    struct bucket1 *buckets;
    uint64_t i, num_buckets;
    uint64_t mask = 1ULL << (vector->bits - 1);

    if (!vector->bits)
        return 0;

    num_buckets = 1ULL << (vector->bits - 1);
    buckets = vector->buckets;

    for (i = 0; i < num_buckets; i++)
    {
        if (!bucket1_merge(&buckets[i], &buckets[i + num_buckets]))
        {
            /* FIXME: Error handling is mostly untested. */

            while (i--)
                bucket1_split(&buckets[i], &buckets[i + num_buckets], mask);

            return 0;
        }
    }

    for (i = 0; i < num_buckets; i++)
    {
        bucket1_compress(&buckets[i]);
        free_bucket1(&buckets[i + num_buckets]);
    }

    if ((buckets = realloc(vector->buckets, sizeof(*buckets) * num_buckets)))
        vector->buckets = buckets;

    vector->bits--;
    return 1;
}

void vector_optimize(struct vector *vector)
{
    uint64_t i, num_buckets;
    uint64_t num_entries;

    num_buckets = 1ULL << vector->bits;

    num_entries = 0;
    for (i = 0; i < num_buckets; i++)
        num_entries += bucket1_num_entries(&vector->buckets[i]);

    if (num_entries >= num_buckets * 256)
    {
        while (num_entries >= num_buckets * 64)
        {
            if (!vector_inc_bits(vector)) goto error;
            num_buckets *= 2;
        }
    }

    if (num_buckets >= 2 && num_entries < num_buckets * 16)
    {
        while (num_buckets >= 2 && num_entries < num_buckets * 64)
        {
            if (!vector_dec_bits(vector)) goto error;
            num_buckets /= 2;
        }
    }

    vector->optimize = MIN(num_buckets * 256 - num_entries, num_entries - num_buckets * 16);
    vector->optimize = MAX(vector->optimize, 256ULL);
    return;

error:
    fprintf(stderr, "%s: Failed to optimize vector, trying again later.\n", __func__);
    vector->optimize = 1024;
}

int vector_set_eps(struct vector *vector, float eps)
{
    return vector->ops->set_eps(vector, eps);
}

int vector_empty(struct vector *vector)
{
    struct entry1 *entry;

    VECTOR_FOR_EACH_ENTRY(vector, entry)
    {
        return 0;
    }

    return 1;
}

uint64_t vector_num_entries(struct vector *vector)
{
    uint64_t i, num_buckets;
    uint64_t num_entries;

    num_buckets = 1ULL << vector->bits;

    num_entries = 0;
    for (i = 0; i < num_buckets; i++)
        num_entries += bucket1_num_entries(&vector->buckets[i]);

    return num_entries;
}

uint64_t vector_get_entries(struct vector *vector, uint64_t *indices, float *weights, uint64_t max_entries)
{
    uint64_t count = 0;
    struct entry1 *entry;

    VECTOR_FOR_EACH_ENTRY(vector, entry)
    {
        if (count++ >= max_entries) continue;
        if (indices)
        {
            *indices++ = entry->index;
        }
        if (weights)
        {
            *weights++ = entry->weight;
        }
    }

    return count;
}

int vector_set_entries(struct vector *vector, uint64_t *indices, float *weights, uint64_t num_entries)
{
    if (weights)
    {
        while (num_entries--)
        {
            if (!vector_set_entry(vector, indices[0], weights[0]))
                return 0;

            indices++;
            weights++;
        }
    }
    else
    {
        while (num_entries--)
        {
            if (!vector_set_entry(vector, indices[0], 1.0f))
                return 0;

            indices++;
        }
    }

    return 1;
}

int vector_add_entries(struct vector *vector, uint64_t *indices, float *weights, uint64_t num_entries)
{
    if (weights)
    {
        while (num_entries--)
        {
            if (!vector_add_entry(vector, indices[0], weights[0]))
                return 0;

            indices++;
            weights++;
        }
    }
    else
    {
        while (num_entries--)
        {
            if (!vector_add_entry(vector, indices[0], 1.0f))
                return 0;

            indices++;
        }
    }

    return 1;
}

int vector_add_vector(struct vector *out, struct vector *vector, float weight)
{
    struct entry1 *entry;

    VECTOR_FOR_EACH_ENTRY(vector, entry)
    {
        if (!vector_add_entry(out, entry->index, entry->weight * weight))
            return 0;
    }

    /* vector_add_entry already updated the revision */
    return 1;
}

int vector_sub_entry(struct vector *vector, uint64_t index, float weight)
{
    return vector_add_entry(vector, index, -weight);
}

int vector_sub_entries(struct vector *vector, uint64_t *indices, float *weights, uint64_t num_entries)
{
    if (weights)
    {
        while (num_entries--)
        {
            if (!vector_add_entry(vector, indices[0], -weights[0]))
                return 0;

            indices++;
            weights++;
        }
    }
    else
    {
        while (num_entries--)
        {
            if (!vector_add_entry(vector, indices[0], -1.0f))
                return 0;

            indices++;
        }
    }

    return 1;
}

int vector_sub_vector(struct vector *out, struct vector *vector, float weight)
{
    return vector_add_vector(out, vector, -weight);
}

int vector_del_entries(struct vector *vector, uint64_t *indices, uint64_t num_entries)
{
    while (num_entries--)
    {
        if (!vector_del_entry(vector, indices[0]))
            return 0;

        indices++;
    }

    return 1;
}

int vector_mul_const(struct vector *vector, float constant)
{
    return vector->ops->mul_const(vector, constant);
}

double vector_sum_weights(const struct vector *vector)
{
    struct entry1 *entry;
    double sum = 0.0;

    VECTOR_FOR_EACH_ENTRY(vector, entry)
    {
        sum += entry->weight;
    }

    return sum;
}

double vector_norm(const struct vector *vector)
{
    struct entry1 *entry;
    double norm = 0.0;

    VECTOR_FOR_EACH_ENTRY(vector, entry)
    {
        norm += entry->weight * entry->weight;
    }

    return sqrt(norm);
}

double vector_mul_vector(const struct vector *vector1, const struct vector *vector2)
{
    struct entry1 *entry1, *entry2;
    double product = 0.0;

    VECTOR_FOR_EACH_ENTRY2(vector1, entry1, vector2, entry2)
    {
        if (!entry1 || !entry2) continue;
        product += entry1->weight * entry2->weight;
    }

    return product;
}

double vector_sub_vector_norm(const struct vector *vector1, const struct vector *vector2)
{
    struct entry1 *entry1, *entry2;
    double norm = 0.0;
    double weight;

    VECTOR_FOR_EACH_ENTRY2(vector1, entry1, vector2, entry2)
    {
        if (entry1 && entry2)
            weight = entry1->weight - entry2->weight;
        else if (entry1)
            weight = entry1->weight;
        else if (entry2)
            weight = entry2->weight;  /* skip the minus sign */
        else
            assert(0);

        norm += weight * weight;
    }

    return sqrt(norm);
}
