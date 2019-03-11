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
    size_t num_members;
    size_t max_members;
    char *members;

    size_t size;
    int (*compar)(const void *, const void *, void *);
    void *userdata;
};

struct minheap *alloc_minheap(size_t size, int (*compar)(const void *, const void *, void *), void *userdata)
{
    struct minheap *h;

    if (!(h = malloc(sizeof(*h))))
        return NULL;

    h->num_members  = 0;
    h->max_members  = 0;
    h->members      = NULL;
    h->size         = size;
    h->compar       = compar;
    h->userdata     = userdata;
    return h;
}

void free_minheap(struct minheap *h)
{
    free(h->members);
    free(h);
}

int minheap_push(struct minheap *h, const void *element)
{
    size_t max_members;
    char *members;
    size_t i;

    if (h->num_members >= h->max_members)
    {
        max_members = h->max_members ? (h->max_members * 2) : 4;
        if (!(members = realloc(h->members, h->size * max_members)))
            return 0;

        h->max_members = max_members;
        h->members = members;
    }

    i = h->num_members++;
    while (i && h->compar(element, h->members + PARENT(i) * h->size, h->userdata) < 0)
    {
        memcpy(h->members + i * h->size, h->members + PARENT(i) * h->size, h->size);
        i = PARENT(i);
    }

    memcpy(h->members + i * h->size, element, h->size);
    return 1;
}

void minheap_heapify(struct minheap *h, size_t i)
{
    for (;;)
    {
        size_t left = LCHILD(i);
        size_t right = RCHILD(i);
        size_t smallest = i;

        if (left < h->num_members && h->compar(h->members + left * h->size,
            h->members + smallest * h->size, h->userdata) < 0) smallest = left;

        if (right < h->num_members && h->compar(h->members + right * h->size,
            h->members + smallest * h->size, h->userdata) < 0) smallest = right;

        if (smallest == i) break;
        SWAP(h->members + i * h->size, h->members + smallest * h->size, h->size);
        i = smallest;
    }
}

int minheap_pop(struct minheap *h, void *element)
{
    if (!h->num_members) return 0;
    memcpy(element, h->members, h->size);
    memmove(h->members, h->members + (--h->num_members) * h->size, h->size);
    minheap_heapify(h, 0);
    return 1;
}
