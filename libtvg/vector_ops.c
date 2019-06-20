/*
 * Time-varying graph library
 * Vector functions.
 *
 * Copyright (c) 2018-2019 Sebastian Lackner
 */

#include "tvg.h"
#include "internal.h"

static inline struct bucket1 *_vector_get_bucket(struct vector *vector, uint64_t index)
{
    uint32_t i = (uint32_t)(index & ((1ULL << vector->bits) - 1));
    return &vector->buckets[i];
}

static inline struct entry1 *_vector_get_entry(struct vector *vector, uint64_t index, int allocate)
{
    struct bucket1 *bucket = _vector_get_bucket(vector, index);
    return bucket1_get_entry(bucket, index, allocate);
}

int vector_has_entry(struct vector *vector, uint64_t index)
{
    return _vector_get_entry(vector, index, 0) != NULL;
}

float vector_get_entry(struct vector *vector, uint64_t index)
{
    struct entry1 *entry;

    if (!(entry = _vector_get_entry(vector, index, 0)))
        return 0.0;

    return entry->weight;
}

static int generic_clear(struct vector *vector)
{
    uint64_t i, num_buckets;

    num_buckets = 1ULL << vector->bits;
    for (i = 0; i < num_buckets; i++)
        bucket1_clear(&vector->buckets[i]);

    vector->revision++;
    if (!--vector->optimize)
        vector_optimize(vector);

    return 1;
}

static int generic_set(struct vector *vector, uint64_t index, float weight)
{
    struct entry1 *entry;

    if (!(entry = _vector_get_entry(vector, index, 1)))
        return 0;

    entry->weight = weight;

    vector->revision++;
    if (!--vector->optimize)
        vector_optimize(vector);

    return 1;
}

static int generic_add(struct vector *vector, uint64_t index, float weight)
{
    struct entry1 *entry;

    if (!(entry = _vector_get_entry(vector, index, 1)))
        return 0;

    entry->weight += weight;

    vector->revision++;
    if (!--vector->optimize)
        vector_optimize(vector);

    return 1;
}

static int generic_del(struct vector *vector, uint64_t index)
{
    struct bucket1 *bucket;
    struct entry1 *entry;

    bucket = _vector_get_bucket(vector, index);
    if (!(entry = bucket1_get_entry(bucket, index, 0)))
        return 1;

    bucket1_del_entry(bucket, entry);

    vector->revision++;
    if (!--vector->optimize)
        vector_optimize(vector);

    return 1;
}

static int generic_mul_const(struct vector *vector, float constant)
{
    struct entry1 *entry;

    VECTOR_FOR_EACH_ENTRY(vector, entry)
    {
        entry->weight *= constant;
    }

    vector->revision++;
    return 1;
}

static int generic_set_eps(struct vector *vector, float eps)
{
    vector->eps = (float)fabs(eps);
    return 1;
}

const struct vector_ops vector_generic_ops =
{
    generic_set_eps,
    generic_clear,
    generic_set,
    generic_add,
    generic_del,
    generic_mul_const,
};

static int nonzero_set(struct vector *vector, uint64_t index, float weight)
{
    /* Is the weight filtered? */
    if (fabs(weight) <= vector->eps)
    {
        generic_del(vector, index);
        return 1;
    }

    return generic_set(vector, index, weight);
}

static int nonzero_add(struct vector *vector, uint64_t index, float weight)
{
    struct bucket1 *bucket;
    struct entry1 *entry;
    int allocate;

    /* Only allocate a new entry when the weight is not filtered. */
    allocate = !(fabs(weight) <= vector->eps);
    bucket = _vector_get_bucket(vector, index);
    if (!(entry = bucket1_get_entry(bucket, index, allocate)))
        return !allocate;

    weight += entry->weight;
    if (fabs(weight) <= vector->eps)
    {
        bucket1_del_entry(bucket, entry);
    }
    else
    {
        entry->weight = weight;
    }

    vector->revision++;
    if (!--vector->optimize)
        vector_optimize(vector);

    return 1;
}

static int nonzero_mul_const(struct vector *vector, float constant)
{
    struct bucket1 *bucket;
    struct entry1 *entry, *out;
    uint64_t i, num_buckets;

    num_buckets = 1ULL << vector->bits;
    for (i = 0; i < num_buckets; i++)
    {
        bucket = &vector->buckets[i];
        out = &bucket->entries[0];

        BUCKET1_FOR_EACH_ENTRY(bucket, entry)
        {
            entry->weight *= constant;
            if (fabs(entry->weight) <= vector->eps) continue;
            *out++ = *entry;
        }

        bucket->num_entries = (uint64_t)(out - &bucket->entries[0]);
        assert(bucket->num_entries <= bucket->max_entries);
    }

    vector->revision++;
    /* FIXME: Trigger vector_optimize? */
    return 1;
}

static int nonzero_set_eps(struct vector *vector, float eps)
{
    vector->eps = (float)fabs(eps);
    return nonzero_mul_const(vector, 1.0);
}

const struct vector_ops vector_nonzero_ops =
{
    nonzero_set_eps,
    generic_clear,
    nonzero_set,
    nonzero_add,
    generic_del,
    nonzero_mul_const,
};

static int positive_set(struct vector *vector, uint64_t index, float weight)
{
    /* Is the weight filtered? */
    if (weight <= vector->eps)
    {
        generic_del(vector, index);
        return 1;
    }

    return generic_set(vector, index, weight);
}

static int positive_add(struct vector *vector, uint64_t index, float weight)
{
    struct bucket1 *bucket;
    struct entry1 *entry;
    int allocate;

    /* Only allocate a new entry when the weight is not filtered. */
    allocate = !(weight <= vector->eps);
    bucket = _vector_get_bucket(vector, index);
    if (!(entry = bucket1_get_entry(bucket, index, allocate)))
        return !allocate;

    weight += entry->weight;
    if (weight <= vector->eps)
    {
        bucket1_del_entry(bucket, entry);
    }
    else
    {
        entry->weight = weight;
    }

    vector->revision++;
    if (!--vector->optimize)
        vector_optimize(vector);

    return 1;
}

static int positive_mul_const(struct vector *vector, float constant)
{
    struct bucket1 *bucket;
    struct entry1 *entry, *out;
    uint64_t i, num_buckets;

    num_buckets = 1ULL << vector->bits;
    for (i = 0; i < num_buckets; i++)
    {
        bucket = &vector->buckets[i];
        out = &bucket->entries[0];

        BUCKET1_FOR_EACH_ENTRY(bucket, entry)
        {
            entry->weight *= constant;
            if (entry->weight <= vector->eps) continue;
            *out++ = *entry;
        }

        bucket->num_entries = (uint64_t)(out - &bucket->entries[0]);
        assert(bucket->num_entries <= bucket->max_entries);
    }

    vector->revision++;
    /* FIXME: Trigger vector_optimize? */
    return 1;
}

static int positive_set_eps(struct vector *vector, float eps)
{
    vector->eps = (float)fabs(eps);
    return positive_mul_const(vector, 1.0);
}

const struct vector_ops vector_positive_ops =
{
    positive_set_eps,
    generic_clear,
    positive_set,
    positive_add,
    generic_del,
    positive_mul_const,
};
