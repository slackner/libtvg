/*
 * Time-varying graph library
 *
 * Copyright (c) 2017-2019 Sebastian Lackner
 */

#ifndef _INTERNAL_H_
#define _INTERNAL_H_

#include "tvg.h"

#define MIN(a, b) \
    ({ __typeof__ (a) __a = (a); \
       __typeof__ (b) __b = (b); \
       __a < __b ? __a : __b; })

#define MAX(a, b) \
    ({ __typeof__ (a) __a = (a); \
       __typeof__ (b) __b = (b); \
       __a > __b ? __a : __b; })

#define COMPARE(a, b) \
    ({ __typeof__ (a) __a = (a); \
       __typeof__ (b) __b = (b); \
       (__a > __b) - (__a < __b); })

#define SWAP(a, b, size)                        \
    do                                          \
    {                                           \
        register size_t __size = (size);        \
        register char *__a = (a), *__b = (b);   \
        while (__size--)                        \
        {                                       \
            char __tmp = *__a;                  \
            *__a++ = *__b;                      \
            *__b++ = __tmp;                     \
        }                                       \
    }                                           \
    while (0)

#define LIKELY(x)   __builtin_expect((x), 1)
#define UNLIKELY(x) __builtin_expect((x), 0)
#define DECL_INTERNAL __attribute__((__visibility__("hidden")))
#define C_ASSERT(e) extern void __C_ASSERT__(int [(e) ? 1 : -1])

struct vector_ops
{
    float     (*get)(struct vector *, uint64_t);
    int       (*set)(struct vector *, uint64_t, float);
    int       (*add)(struct vector *, uint64_t, float);
    void      (*del)(struct vector *, uint64_t);
    void      (*mul_const)(struct vector *, float);
};

struct graph_ops
{
    float     (*get)(struct graph *, uint64_t, uint64_t);
    int       (*set)(struct graph *, uint64_t, uint64_t, float);
    int       (*add)(struct graph *, uint64_t, uint64_t, float);
    void      (*del)(struct graph *, uint64_t, uint64_t);
    void      (*mul_const)(struct graph *, float);
};

struct window_ops
{
    int       (*add)(struct window *, struct graph *);
    int       (*sub)(struct window *, struct graph *);
    int       (*mov)(struct window *, float ts);
};

void progress(const char *format, ...) __attribute__((format (printf,1,2))) DECL_INTERNAL;
uint64_t clock_monotonic(void) DECL_INTERNAL;
uint64_t count_lines(FILE *fp) DECL_INTERNAL;

void random_bytes(uint8_t *buffer, size_t length) DECL_INTERNAL;
float random_float(void) DECL_INTERNAL;

void init_bucket1(struct bucket1 *bucket) DECL_INTERNAL;
void free_bucket1(struct bucket1 *bucket) DECL_INTERNAL;
void bucket1_compress(struct bucket1 *bucket) DECL_INTERNAL;
int bucket1_split(struct bucket1 *bucket1, struct bucket1 *bucket2, uint64_t mask) DECL_INTERNAL;
int bucket1_merge(struct bucket1 *bucket1, struct bucket1 *bucket2) DECL_INTERNAL;
struct entry1 *bucket1_get_entry(struct bucket1 *bucket, uint64_t index, int allocate) DECL_INTERNAL;
void bucket1_del_entry(struct bucket1 *bucket, struct entry1 *entry) DECL_INTERNAL;

extern const struct vector_ops vector_generic_ops DECL_INTERNAL;
extern const struct vector_ops vector_nonzero_ops DECL_INTERNAL;
extern const struct vector_ops vector_positive_ops DECL_INTERNAL;

void init_bucket2(struct bucket2 *bucket) DECL_INTERNAL;
void free_bucket2(struct bucket2 *bucket) DECL_INTERNAL;
void bucket2_compress(struct bucket2 *bucket) DECL_INTERNAL;
int bucket2_split(struct bucket2 *bucket1, struct bucket2 *bucket2, uint64_t source_mask, uint64_t target_mask) DECL_INTERNAL;
int bucket2_merge(struct bucket2 *bucket1, struct bucket2 *bucket2) DECL_INTERNAL;
struct entry2 *bucket2_get_entry(struct bucket2 *bucket, uint64_t source, uint64_t target, int allocate) DECL_INTERNAL;
void bucket2_del_entry(struct bucket2 *bucket, struct entry2 *entry) DECL_INTERNAL;

extern const struct graph_ops graph_generic_ops DECL_INTERNAL;
extern const struct graph_ops graph_nonzero_ops DECL_INTERNAL;
extern const struct graph_ops graph_positive_ops DECL_INTERNAL;

struct window *alloc_window(struct tvg *tvg, const struct window_ops *ops, float window_l,
                            float window_r, float weight, float log_beta) DECL_INTERNAL;

extern const struct window_ops window_rect_ops DECL_INTERNAL;
extern const struct window_ops window_decay_ops DECL_INTERNAL;
extern const struct window_ops window_smooth_ops DECL_INTERNAL;

struct minheap *alloc_minheap(size_t entry_size, int (*compar)(const void *, const void *, void *), void *userdata) DECL_INTERNAL;
void free_minheap(struct minheap *h) DECL_INTERNAL;
int minheap_push(struct minheap *h, const void *element) DECL_INTERNAL;
void minheap_heapify(struct minheap *h, size_t i) DECL_INTERNAL;
int minheap_pop(struct minheap *h, void *element) DECL_INTERNAL;

#endif /* _INTERNAL_H_ */
