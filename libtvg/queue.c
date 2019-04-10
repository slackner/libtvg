/*
 * Time-varying graph library
 * Queue / Ring buffer implementation.
 *
 * Copyright (c) 2019 Sebastian Lackner
 */

#include "tvg.h"
#include "internal.h"

struct queue
{
    size_t first_entry;
    size_t num_entries;
    size_t max_entries;
    size_t entry_size;
    char  *entries;
};

struct queue *alloc_queue(size_t entry_size)
{
    struct queue *q;

    if (!(q = malloc(sizeof(*q))))
        return NULL;

    q->first_entry  = 0;
    q->num_entries  = 0;
    q->max_entries  = 0;
    q->entry_size   = entry_size;
    q->entries      = NULL;
    return q;
}

void free_queue(struct queue *q)
{
    if (!q) return;
    free(q->entries);
    free(q);
}

int queue_put(struct queue *q, const void *element)
{
    size_t max_entries;
    char *entries;
    size_t i;

    if (q->num_entries >= q->max_entries)
    {
        max_entries = MAX(q->num_entries + 1, q->max_entries * 2);
        if (!(entries = realloc(q->entries, q->entry_size * max_entries)))
            return 0;

        i = MIN(q->first_entry, max_entries - q->max_entries);
        memcpy(entries + q->max_entries * q->entry_size, entries, q->entry_size * i);
        if (i < q->first_entry)
            memcpy(entries, entries + i * q->entry_size, q->entry_size * (q->first_entry - i));

        q->max_entries = max_entries;
        q->entries = entries;
    }

    i = (q->first_entry + q->num_entries++) % q->max_entries;
    memcpy(q->entries + i * q->entry_size, element, q->entry_size);
    return 1;
}

int queue_get(struct queue *q, void *element)
{
    if (!q->num_entries) return 0;
    if (element) memcpy(element, q->entries + q->first_entry * q->entry_size, q->entry_size);
    if (!--q->num_entries) q->first_entry = 0;  /* queue empty, reset */
    else q->first_entry = (q->first_entry + 1) % q->max_entries;
    return 1;
}

const void *queue_ptr(struct queue *q, size_t index)
{
    size_t i;
    if (index >= q->num_entries) return NULL;
    i = (q->first_entry + index) % q->max_entries;
    return q->entries + i * q->entry_size;
}
