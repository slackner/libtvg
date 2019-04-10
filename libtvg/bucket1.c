/*
 * Time-varying graph library
 * Vector bucket functions.
 *
 * Copyright (c) 2018-2019 Sebastian Lackner
 */

#include "tvg.h"
#include "internal.h"

void init_bucket1(struct bucket1 *bucket)
{
    bucket->num_entries = 0;
    bucket->max_entries = 0;
    bucket->entries     = NULL;
    bucket->hint        = ~0ULL;
}

void free_bucket1(struct bucket1 *bucket)
{
    free(bucket->entries);
    init_bucket1(bucket);
}

void bucket1_compress(struct bucket1 *bucket)
{
    struct entry1 *entries;

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
        free_bucket1(bucket);
    }
}

static int bucket1_reserve(struct bucket1 *bucket, uint64_t new_entries)
{
    uint64_t max_entries;
    struct entry1 *entries;

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

int bucket1_split(struct bucket1 *bucket1, struct bucket1 *bucket2, uint64_t mask)
{
    struct entry1 *out1, *out2;
    struct entry1 *entry;
    uint64_t num_entries = 0;

    assert(!bucket2->num_entries);

    BUCKET1_FOR_EACH_ENTRY(bucket1, entry)
    {
        if (!(entry->index & mask)) continue;
        num_entries++;
    }

    if (!bucket1_reserve(bucket2, num_entries))
        return 0;

    out1 = &bucket1->entries[0];
    out2 = &bucket2->entries[0];

    BUCKET1_FOR_EACH_ENTRY(bucket1, entry)
    {
        if (!(entry->index & mask))
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

int bucket1_merge(struct bucket1 *bucket1, struct bucket1 *bucket2)
{
    struct entry1 *entry1, *entry2;
    struct entry1 *out;
    uint64_t num_entries;

    if (!bucket1_reserve(bucket1, bucket2->num_entries))
        return 0;

    num_entries = bucket1->num_entries + bucket2->num_entries;
    out = &bucket1->entries[num_entries];

    BUCKET1_FOR_EACH_ENTRY_REV2(bucket1, entry1, bucket2, entry2)
    {
        if (entry1) *--out = *entry1;
        if (entry2) *--out = *entry2;
    }

    assert(out == &bucket1->entries[0]);
    bucket1->num_entries = num_entries;
    return 1;
}

struct entry1 *bucket1_get_entry(struct bucket1 *bucket, uint64_t index, int allocate)
{
    struct entry1 *entry;
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
            if (index < entry->index) max = i - 1;
            else if (index > entry->index) min = i + 1;
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
            if (index < entry->index) max = i - 1;
            else if (index > entry->index) min = i + 1;
            else
            {
                bucket->hint = i + 1;
                return entry;
            }
        }

        insert = min;
    }

    if (!allocate) return NULL;
    if (!bucket1_reserve(bucket, 1)) return NULL;

    entry = &bucket->entries[insert];
    memmove(&entry[1], entry, (size_t)((char *)&bucket->entries[bucket->num_entries] - (char *)entry));
    bucket->num_entries++;

    entry->index  = index;
    entry->weight = 0.0;
    return entry;
}

void bucket1_del_entry(struct bucket1 *bucket, struct entry1 *entry)
{
    if (bucket->hint < bucket->num_entries && entry <= &bucket->entries[bucket->hint])
    {
        if (entry < &bucket->entries[bucket->hint]) bucket->hint--;
        else bucket->hint = ~0ULL;
    }
    memmove(entry, &entry[1], (size_t)((char *)&bucket->entries[bucket->num_entries] - (char *)&entry[1]));
    bucket->num_entries--;
}
