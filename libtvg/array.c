/*
 * Time-varying graph library
 * Array implementation.
 *
 * Copyright (c) 2019 Sebastian Lackner
 */

#define _GNU_SOURCE
#include "internal.h"

struct array
{
    size_t num_entries;
    size_t max_entries;
    size_t entry_size;
    char  *entries;
};

struct array *alloc_array(size_t entry_size)
{
    struct array *a;

    if (!(a = malloc(sizeof(*a))))
        return NULL;

    a->num_entries  = 0;
    a->max_entries  = 0;
    a->entry_size   = entry_size;
    a->entries      = NULL;
    return a;
}

void free_array(struct array *a)
{
    if (!a) return;
    free(a->entries);
    free(a);
}

void *array_append_empty(struct array *a)
{
    size_t max_entries;
    char *entries;
    size_t i;

    if (a->num_entries >= a->max_entries)
    {
        max_entries = MAX(a->num_entries + 1, a->max_entries * 2);
        if (!(entries = realloc(a->entries, a->entry_size * max_entries)))
            return NULL;

        a->max_entries = max_entries;
        a->entries = entries;
    }

    i = a->num_entries++;
    return a->entries + i * a->entry_size;
}

int array_append(struct array *a, const void *element)
{
    char *ptr;

    if (!(ptr = array_append_empty(a)))
        return 0;

    memcpy(ptr, element, a->entry_size);
    return 1;
}

int array_remove(struct array *a, void *element)
{
    if (!a->num_entries) return 0;
    a->num_entries--;
    if (element) memcpy(element, a->entries + a->num_entries * a->entry_size, a->entry_size);
    return 1;
}

#ifdef __linux__

void array_sort(struct array *a, int (*compar)(const void *, const void *, void *), void *userdata)
{
    qsort_r(a->entries, a->num_entries, a->entry_size, compar, userdata);
}

#else   /* __linux__ */

/* On macOS / BSD, the order of parameters of the qsort_r() function is swapped.
 * We use a wrapper function to call the original compar callback. */

struct qsort_context
{
    int (*compar)(const void *, const void *, void *);
    void *userdata;
};

static int _qsort_callback(void *userdata, const void *a, const void *b)
{
    struct qsort_context *context = userdata;
    return context->compar(a, b, context->userdata);
}

void array_sort(struct array *a, int (*compar)(const void *, const void *, void *), void *userdata)
{
    struct qsort_context context;
    context.compar = compar;
    context.userdata = userdata;
    qsort_r(a->entries, a->num_entries, a->entry_size, &context, _qsort_callback);
}

#endif  /* __linux__ */

const void *array_ptr(struct array *a, size_t index)
{
    if (index >= a->num_entries) return NULL;
    return a->entries + index * a->entry_size;
}

size_t array_count(struct array *a)
{
    return a->num_entries;
}
