/*
 * Time-varying graph library
 * Ranges functions.
 *
 * Copyright (c) 2019 Sebastian Lackner
 */

#include "internal.h"

struct costs
{
    uint64_t len;
    int64_t  weight;
};

static int _sort_costs_by_weight(const void *a, const void *b, void *userdata)
{
    const struct costs *ca = a, *cb = b;
    return COMPARE(ca->weight, cb->weight);
}

static int _range_compar(const void *a, const void *b, void *userdata)
{
    const struct range *ra = AVL_ENTRY(a, struct range, entry);
    const struct range *rb = AVL_ENTRY(b, struct range, entry);

    if (ra->pos + ra->len <= rb->pos)
        return -1;
    if (rb->pos + rb->len <= ra->pos)
        return 1;

    return 0;
}

static int _range_lookup(const void *a, const void *b, void *userdata)
{
    const struct range *ra = AVL_ENTRY(a, struct range, entry);
    const uint64_t *pos = b;

    if (ra->pos > *pos)
        return 1;
    if (ra->pos + ra->len <= *pos)
        return -1;

    return 0;
}

struct ranges *alloc_ranges(void)
{
    struct ranges *ranges;

    if (!(ranges = malloc(sizeof(*ranges))))
        return NULL;

    avl_init(&ranges->tree, _range_compar, _range_lookup, NULL);
    return ranges;
}

void free_ranges(struct ranges *ranges)
{
    struct range *range, *next_range;

    if (!ranges) return;

    AVL_FOR_EACH_SAFE(range, next_range, &ranges->tree, struct range, entry)
    {
        avl_remove(&range->entry);
        free(range);
    }

    free(ranges);
}

void ranges_debug(struct ranges *ranges)
{
    struct range *range;

    fprintf(stderr, "Ranges %p\n", ranges);

    AVL_FOR_EACH(range, &ranges->tree, struct range, entry)
    {
        fprintf(stderr, "-> Range [%llu, %llu], weight = %lld\n",
                (long long unsigned int)range->pos,
                (long long unsigned int)(range->pos + range->len - 1),
                (long long int)range->weight);
    }
}

void ranges_assert_valid(struct ranges *ranges)
{
    struct range *range, *next_range;

    AVL_FOR_EACH_SAFE(range, next_range, &ranges->tree, struct range, entry)
    {
        assert(range->len > 0);
        assert(range->weight != 0);

        if (next_range)
        {
            assert(next_range->pos >= range->pos + range->len);
            if (next_range->pos == range->pos + range->len)
                assert(next_range->weight != range->weight);
        }
    }
}

int ranges_empty(struct ranges *ranges)
{
    return avl_empty(&ranges->tree);
}

int ranges_add_range(struct ranges *ranges, uint64_t pos, uint64_t len, int64_t weight)
{
    struct range *range, *new_range;
    struct range *next_range;
    struct range *prev_range;

    if (!weight || !len)
        return 1;  /* nothing to do */

    if (pos + len < pos)
        return 0;  /* overflow */

    range = AVL_LOOKUP_GE(&ranges->tree, &pos, struct range, entry);
    while (len)
    {
        /* If pos > range->pos, we are only updating part of a range. Split
         * the existing range and only proceed with the second part. */
        if (range && range->pos < pos)
        {
            assert(range->pos + range->len > pos);

            if (!(new_range = malloc(sizeof(*new_range))))
                return 0;

            new_range->pos    = pos;
            new_range->len    = range->pos + range->len - pos;
            new_range->weight = range->weight;

            range->len = pos - range->pos;
            avl_add_after(&ranges->tree, &range->entry, &new_range->entry);

            range = new_range;
            /* fall-through */
        }

        assert(!range || range->pos >= pos);

        if (range && range->pos == pos)
        {
            /* If the existing region goes past the region we are updating,
             * split the current region and only proceed with the first part. */
            if (range->pos + range->len > pos + len)
            {
                if (!(new_range = malloc(sizeof(*new_range))))
                    return 0;

                new_range->pos = pos + len;
                new_range->len = range->pos + range->len - new_range->pos;
                new_range->weight = range->weight;

                range->len = new_range->pos - range->pos;
                avl_add_after(&ranges->tree, &range->entry, &new_range->entry);

                /* fall-through */
            }

            assert(range->pos + range->len <= pos + len);

            next_range = AVL_NEXT(range, &ranges->tree, struct range, entry);
            pos += range->len;
            len -= range->len;

            if (!(range->weight += weight))
            {
                avl_remove(&range->entry);
                free(range);
            }
            else
            {
                prev_range = AVL_PREV(range, &ranges->tree, struct range, entry);
                if (prev_range && prev_range->pos + prev_range->len == range->pos &&
                    prev_range->weight == range->weight)
                {
                    prev_range->len += range->len;
                    avl_remove(&range->entry);
                    free(range);
                }
            }

            range = next_range;
            continue;
        }

        assert(!range || range->pos > pos);

        prev_range = range ? AVL_PREV(range, &ranges->tree, struct range, entry) : NULL;
        if (prev_range && prev_range->pos + prev_range->len == pos &&
            prev_range->weight == weight)
        {
            if (!range || range->pos >= pos + len)
            {
                prev_range->len += len;
                len = 0;
            }
            else
            {
                prev_range->len += (range->pos - pos);
                len -= (range->pos - pos);
                pos = range->pos;
            }
        }
        else
        {
            if (!(new_range = malloc(sizeof(*new_range))))
                return 0;

            new_range->pos    = pos;
            new_range->weight = weight;

            if (!range || range->pos >= pos + len)
            {
                new_range->len = len;
                len = 0;
            }
            else
            {
                new_range->len = (range->pos - pos);
                len -= (range->pos - pos);
                pos = range->pos;
            }

            avl_add_before(&ranges->tree, &range->entry, &new_range->entry);
        }

        /* Reconsider 'range' the next time around. */
    }

    if (range)
    {
        prev_range = AVL_PREV(range, &ranges->tree, struct range, entry);
        if (prev_range && prev_range->pos + prev_range->len == range->pos &&
            prev_range->weight == range->weight)
        {
            prev_range->len += range->len;
            avl_remove(&range->entry);
            free(range);
        }
    }

    return 1;
}

int64_t ranges_get_weight(struct ranges *ranges, uint64_t pos)
{
    struct range *range;

    if ((range = AVL_LOOKUP(&ranges->tree, &pos, struct range, entry)))
        return range->weight;

    return 0;
}

uint64_t ranges_get_length(struct ranges *ranges)
{
    struct range *range;
    uint64_t result = 0;

    AVL_FOR_EACH(range, &ranges->tree, struct range, entry)
    {
        result += range->len;
    }

    return result;
}

int64_t ranges_get_delta_length(struct ranges *ranges, uint64_t pos, uint64_t len, int64_t *weight)
{
    struct costs current, max_len;
    struct minheap *queue;
    struct range *range;
    struct costs costs;
    int64_t result = len;

    if (weight)
        *weight = 0;

    if (!len)
        return 0;
    if (!(range = AVL_LOOKUP_GE(&ranges->tree, &pos, struct range, entry)))
        return 0;
    if (!(queue = alloc_minheap(sizeof(struct costs), _sort_costs_by_weight, NULL)))
        return 0;  /* FIXME: Return error */

    /* Go through the overlapping ranges and push them to a minheap, so
     * we can iterate over ranges in order sorted by weights. */

    while (range)
    {
        if (range->pos >= pos + len) break;

        costs.len    = MIN(range->pos + range->len, pos + len) - MAX(range->pos, pos);
        costs.weight = range->weight;

        result -= costs.len;

        if (!minheap_push(queue, &costs))
        {
            free_minheap(queue);
            return 0;  /* FIXME: Return error */
        }

        range = AVL_NEXT(range, &ranges->tree, struct range, entry);
    }

    /* Go through the sorted ranges and determine which 'weight' appears
     * most frequently. */

    max_len.len    = 0;
    max_len.weight = 0;

    current.len    = 0;
    current.weight = 0;

    while (minheap_pop(queue, &costs))
    {
        if (costs.weight != current.weight)
        {
            if (current.len && current.len > max_len.len)
                max_len = current;

            current.len    = 0;
            current.weight = costs.weight;
        }

        current.len += costs.len;
    }

    if (current.len && current.len > max_len.len)
        max_len = current;

    free_minheap(queue);

    if (!max_len.weight)
        return 0;

    if (weight)
        *weight = -max_len.weight;

    result -= max_len.len;
    return result;
}
