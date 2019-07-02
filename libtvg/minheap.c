/*
 * Time-varying graph library
 * Min Heap implementation.
 *
 * Copyright (c) 2017-2018 Sebastian Lackner
 */

#include "tvg.h"
#include "internal.h"

#define PARENT(x) (((x) - 1)/2)
#define LCHILD(x) (2 * (x) + 1)
#define RCHILD(x) (2 * (x) + 2)

struct minheap
{
    size_t num_entries;
    size_t max_entries;
    size_t entry_size;
    char  *entries;
    int  (*compar)(const void *, const void *, void *);
    void  *userdata;
};

struct minheap *alloc_minheap(size_t entry_size, int (*compar)(const void *, const void *, void *), void *userdata)
{
    struct minheap *h;

    if (!(h = malloc(sizeof(*h))))
        return NULL;

    h->num_entries  = 0;
    h->max_entries  = 0;
    h->entry_size   = entry_size;
    h->entries      = NULL;
    h->compar       = compar;
    h->userdata     = userdata;
    return h;
}

void free_minheap(struct minheap *h)
{
    if (!h) return;
    free(h->entries);
    free(h);
}

int minheap_push(struct minheap *h, const void *element)
{
    size_t max_entries;
    char *entries;
    size_t i;

    if (h->num_entries >= h->max_entries)
    {
        max_entries = MAX(h->num_entries + 1, h->max_entries * 2);
        if (!(entries = realloc(h->entries, h->entry_size * max_entries)))
            return 0;

        h->max_entries = max_entries;
        h->entries = entries;
    }

    i = h->num_entries++;
    while (i && h->compar(element, h->entries + PARENT(i) * h->entry_size, h->userdata) < 0)
    {
        memcpy(h->entries + i * h->entry_size, h->entries + PARENT(i) * h->entry_size, h->entry_size);
        i = PARENT(i);
    }

    memcpy(h->entries + i * h->entry_size, element, h->entry_size);
    return 1;
}

void minheap_heapify(struct minheap *h, size_t i)
{
    for (;;)
    {
        size_t left = LCHILD(i);
        size_t right = RCHILD(i);
        size_t smallest = i;

        if (left < h->num_entries && h->compar(h->entries + left * h->entry_size,
            h->entries + smallest * h->entry_size, h->userdata) < 0) smallest = left;

        if (right < h->num_entries && h->compar(h->entries + right * h->entry_size,
            h->entries + smallest * h->entry_size, h->userdata) < 0) smallest = right;

        if (smallest == i) break;
        SWAP_BYTES(h->entries + i * h->entry_size, h->entries + smallest * h->entry_size, h->entry_size);
        i = smallest;
    }
}

int minheap_pop(struct minheap *h, void *element)
{
    if (!h->num_entries) return 0;
    if (element) memcpy(element, h->entries, h->entry_size);
    memmove(h->entries, h->entries + (--h->num_entries) * h->entry_size, h->entry_size);
    minheap_heapify(h, 0);
    return 1;
}

size_t minheap_count(struct minheap *h)
{
    return h->num_entries;
}
