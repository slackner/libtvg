/*
 * Time-varying graph library
 * Vector functions.
 *
 * Copyright (c) 2018-2019 Sebastian Lackner
 */

#include "tvg.h"
#include "internal.h"

struct vector *alloc_vector(uint32_t flags)
{
    static const uint32_t bits = 0;
    const struct vector_ops *ops;
    struct vector *vector;
    struct bucket1 *buckets;
    uint64_t i, num_buckets;

    if (flags & ~(TVG_FLAGS_NONZERO |
                  TVG_FLAGS_POSITIVE))
        return NULL;

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

    if (flags & TVG_FLAGS_POSITIVE)
    {
        ops = &vector_positive_ops;
        flags |= TVG_FLAGS_NONZERO;  /* positive implies nonzero */
    }
    else if (flags & TVG_FLAGS_NONZERO)
        ops = &vector_nonzero_ops;
    else
        ops = &vector_generic_ops;

    vector->refcount = 1;
    vector->flags    = flags;
    vector->revision = 0;
    vector->eps      = 0.0;
    vector->ops      = ops;
    vector->bits     = bits;
    vector->buckets  = buckets;
    vector->optimize = 0;

    /* set a proper 'optimize' value */
    vector_optimize(vector);
    return vector;
}

struct vector *grab_vector(struct vector *vector)
{
    if (vector) vector->refcount++;
    return vector;
}

void free_vector(struct vector *vector)
{
    uint64_t i, num_buckets;

    if (!vector) return;
    if (--vector->refcount) return;

    num_buckets = 1ULL << vector->bits;
    for (i = 0; i < num_buckets; i++)
        free_bucket1(&vector->buckets[i]);

    free(vector->buckets);
    free(vector);
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
        num_entries += vector->buckets[i].num_entries;

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

void vector_set_eps(struct vector *vector, float eps)
{
    vector->eps = (float)fabs(eps);
    vector->ops->mul_const(vector, 1.0);
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

int vector_has_entry(struct vector *vector, uint64_t index)
{
    /* keep in sync with _vector_get_bucket! */
    uint32_t i = (uint32_t)(index & ((1ULL << vector->bits) - 1));
    return bucket1_get_entry(&vector->buckets[i], index, 0) != NULL;
}

float vector_get_entry(struct vector *vector, uint64_t index)
{
    return vector->ops->get(vector, index);
}

uint64_t vector_get_entries(struct vector *vector, uint64_t *indices, float *weights, uint64_t max_edges)
{
    uint64_t count = 0;
    struct entry1 *entry;

    VECTOR_FOR_EACH_ENTRY(vector, entry)
    {
        if (count++ >= max_edges) continue;
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

int vector_set_entry(struct vector *vector, uint64_t index, float weight)
{
    return vector->ops->set(vector, index, weight);
}

int vector_set_entries(struct vector *vector, uint64_t *indices, float *weights, uint64_t num_entries)
{
    if (weights)
    {
        while (num_entries--)
        {
            if (!vector->ops->set(vector, indices[0], weights[0]))
                return 0;

            indices++;
            weights++;
        }
    }
    else
    {
        while (num_entries--)
        {
            if (!vector->ops->set(vector, indices[0], 1.0f))
                return 0;

            indices++;
        }
    }

    return 1;
}

int vector_add_entry(struct vector *vector, uint64_t index, float weight)
{
    return vector->ops->add(vector, index, weight);
}

int vector_add_entries(struct vector *vector, uint64_t *indices, float *weights, uint64_t num_entries)
{
    if (weights)
    {
        while (num_entries--)
        {
            if (!vector->ops->add(vector, indices[0], weights[0]))
                return 0;

            indices++;
            weights++;
        }
    }
    else
    {
        while (num_entries--)
        {
            if (!vector->ops->add(vector, indices[0], 1.0f))
                return 0;

            indices++;
        }
    }

    return 1;
}

int vector_sub_entry(struct vector *vector, uint64_t index, float weight)
{
    return vector->ops->add(vector, index, -weight);
}

int vector_sub_entries(struct vector *vector, uint64_t *indices, float *weights, uint64_t num_entries)
{
    if (weights)
    {
        while (num_entries--)
        {
            if (!vector->ops->add(vector, indices[0], -weights[0]))
                return 0;

            indices++;
            weights++;
        }
    }
    else
    {
        while (num_entries--)
        {
            if (!vector->ops->add(vector, indices[0], -1.0f))
                return 0;

            indices++;
        }
    }

    return 1;
}

void vector_del_entry(struct vector *vector, uint64_t index)
{
    vector->ops->del(vector, index);
}

void vector_del_entries(struct vector *vector, uint64_t *indices, uint64_t num_entries)
{
    while (num_entries--)
    {
        vector->ops->del(vector, indices[0]);
        indices++;
    }
}

void vector_mul_const(struct vector *vector, float constant)
{
    vector->ops->mul_const(vector, constant);
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
