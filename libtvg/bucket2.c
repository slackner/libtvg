/*
 * Time-varying graph library
 * Matrix bucket functions.
 *
 * Copyright (c) 2018-2019 Sebastian Lackner
 */

#include "internal.h"

void init_bucket2(struct bucket2 *bucket)
{
    bucket->num_entries = 0;
    bucket->max_entries = 0;
    bucket->entries     = NULL;
    bucket->hint        = ~0ULL;
}

int init_bucket2_from(struct bucket2 *bucket, struct bucket2 *source)
{
    uint64_t num_entries = source->num_entries;
    uint64_t max_entries;
    struct entry2 *entries;

    if (!num_entries)
    {
        init_bucket2(bucket);
        return 1;
    }

    max_entries = MAX(num_entries, 2ULL);
    if (!(entries = malloc(sizeof(*entries) * max_entries)))
        return 0;

    memcpy(entries, source->entries, sizeof(*entries) * num_entries);

    bucket->num_entries = num_entries;
    bucket->max_entries = max_entries;
    bucket->entries     = entries;
    bucket->hint        = ~0ULL;
    return 1;
}

void free_bucket2(struct bucket2 *bucket)
{
    free(bucket->entries);
    init_bucket2(bucket);
}

void bucket2_clear(struct bucket2 *bucket)
{
    bucket->num_entries = 0;
}

void bucket2_compress(struct bucket2 *bucket)
{
    struct entry2 *entries;

    if (bucket->num_entries >= bucket->max_entries)
        return;

    if (bucket->num_entries)
    {
        if (!(entries = realloc(bucket->entries, sizeof(*entries) * bucket->num_entries)))
            return;

        bucket->max_entries = bucket->num_entries;
        bucket->entries     = entries;
    }
    else
    {
        free_bucket2(bucket);
    }
}

static int bucket2_reserve(struct bucket2 *bucket, uint64_t new_entries)
{
    uint64_t max_entries;
    struct entry2 *entries;

    if (!bucket->entries)
    {
        max_entries = MAX(new_entries, 2ULL);
        if (!(entries = malloc(sizeof(*entries) * max_entries)))
            return 0;

        bucket->num_entries = 0;
        bucket->max_entries = max_entries;
        bucket->entries     = entries;
    }
    else if (bucket->num_entries + new_entries > bucket->max_entries)
    {
        max_entries = MAX(bucket->num_entries + new_entries, bucket->max_entries * 2);
        if (!(entries = realloc(bucket->entries, sizeof(*entries) * max_entries)))
            return 0;

        bucket->max_entries = max_entries;
        bucket->entries     = entries;
    }

    return 1;
}

int bucket2_split(struct bucket2 *bucket1, struct bucket2 *bucket2,
                  uint64_t source_mask, uint64_t target_mask)
{
    struct entry2 *out1, *out2;
    struct entry2 *entry;
    uint64_t num_entries = 0;

    assert(!bucket2->num_entries);

    BUCKET2_FOR_EACH_ENTRY(bucket1, entry)
    {
        if ((source_mask && !(entry->source & source_mask)) ||
            (target_mask && !(entry->target & target_mask))) continue;
        num_entries++;
    }

    if (!bucket2_reserve(bucket2, num_entries))
        return 0;

    out1 = &bucket1->entries[0];
    out2 = &bucket2->entries[0];

    BUCKET2_FOR_EACH_ENTRY(bucket1, entry)
    {
        if ((source_mask && !(entry->source & source_mask)) ||
            (target_mask && !(entry->target & target_mask)))
        {
            *out1++ = *entry;
            continue;
        }
        *out2++ = *entry;
    }

    bucket1->num_entries -= num_entries;
    bucket2->num_entries = num_entries;
    return 1;
}

int bucket2_merge(struct bucket2 *bucket1, struct bucket2 *bucket2)
{
    struct entry2 *entry1, *entry2;
    struct entry2 *out;
    uint64_t num_entries;

    if (!bucket2_reserve(bucket1, bucket2->num_entries))
        return 0;

    num_entries = bucket1->num_entries + bucket2->num_entries;
    out = &bucket1->entries[num_entries];

    BUCKET2_FOR_EACH_ENTRY_REV2(bucket1, entry1, bucket2, entry2)
    {
        if (entry1) *--out = *entry1;
        if (entry2) *--out = *entry2;
    }

    assert(out == &bucket1->entries[0]);
    bucket1->num_entries = num_entries;
    return 1;
}

struct entry2 *bucket2_get_entry(struct bucket2 *bucket, uint64_t source, uint64_t target, int allocate)
{
    struct entry2 *entry;
    uint64_t insert = 0;

    if (bucket->num_entries)
    {
        /* Note that we have to use signed numbers here. */
        int64_t min = 0;
        int64_t max = bucket->num_entries - 1;
        int64_t i;

        if (bucket->hint < bucket->num_entries)
        {
            i = bucket->hint;
            entry = &bucket->entries[i];
            if (target < entry->target) max = i - 1;
            else if (target > entry->target) min = i + 1;
            else if (source < entry->source) max = i - 1;
            else if (source > entry->source) min = i + 1;
            else
            {
                bucket->hint++;
                return entry;
            }
        }

        while (min <= max)
        {
            i = (min + max) / 2;
            entry = &bucket->entries[i];
            if (target < entry->target) max = i - 1;
            else if (target > entry->target) min = i + 1;
            else if (source < entry->source) max = i - 1;
            else if (source > entry->source) min = i + 1;
            else
            {
                bucket->hint = i + 1;
                return entry;
            }
        }

        insert = min;
    }

    if (!allocate) return NULL;
    if (!bucket2_reserve(bucket, 1)) return NULL;

    entry = &bucket->entries[insert];
    memmove(&entry[1], entry, (size_t)((char *)&bucket->entries[bucket->num_entries] - (char *)entry));
    bucket->num_entries++;

    entry->source = source;
    entry->target = target;
    entry->weight = 0.0;
    entry->reserved = 0;
    return entry;
}

void bucket2_del_entry(struct bucket2 *bucket, struct entry2 *entry)
{
    if (bucket->hint < bucket->num_entries && entry <= &bucket->entries[bucket->hint])
    {
        if (entry < &bucket->entries[bucket->hint]) bucket->hint--;
        else bucket->hint = ~0ULL;
    }
    memmove(entry, &entry[1], (size_t)((char *)&bucket->entries[bucket->num_entries] - (char *)&entry[1]));
    bucket->num_entries--;
}
