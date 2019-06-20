/*
 * Time-varying graph library
 *
 * Copyright (c) 2018-2019 Sebastian Lackner
 */

#include <time.h>
#include <math.h>

#include "tvg.h"
#include "internal.h"
#include "tree.h"

/* replace random_bytes() to avoid export from libtvg */
static void _random_bytes(uint8_t *buffer, size_t length) { while (length--) *buffer++ = (uint8_t)rand(); }
#define random_bytes _random_bytes

/* replace random_float() to avoid export from libtvg */
static float _random_float(void) { return (float)rand() / (float)(RAND_MAX); }
#define random_float _random_float

static uint64_t random_uint64(void)
{
    uint64_t value;
    random_bytes((uint8_t *)&value, sizeof(value));
    return value;
}

static uint64_t abs_sub_uint64(uint64_t a, uint64_t b)
{
    return (a < b) ? (b - a) : (a - b);
}

/* helper for test_next_graph and test_prev_graph */
static struct tvg *alloc_random_tvg(uint32_t flags, uint32_t count)
{
    uint32_t graph_flags;
    struct graph *graph;
    struct tvg *tvg;
    uint32_t i;
    int ret;

    tvg = alloc_tvg(flags);
    assert(tvg != NULL);

    graph_flags = flags & (TVG_FLAGS_NONZERO |
                           TVG_FLAGS_POSITIVE |
                           TVG_FLAGS_DIRECTED);

    for (i = 0; i < count; i++)
    {
        graph = alloc_graph(graph_flags);
        assert(graph != NULL);
        ret = tvg_link_graph(tvg, graph, random_uint64());
        assert(ret);
        free_graph(graph);
    }

    return tvg;
}

static void test_alloc_vector(void)
{
    struct vector *vector;

    vector = alloc_vector(0);
    assert(vector != NULL);
    assert(vector->flags == 0);
    free_vector(vector);

    vector = alloc_vector(TVG_FLAGS_NONZERO);
    assert(vector != NULL);
    assert(vector->flags == TVG_FLAGS_NONZERO);
    free_vector(vector);

    vector = alloc_vector(TVG_FLAGS_POSITIVE);
    assert(vector != NULL);
    assert(vector->flags == (TVG_FLAGS_NONZERO | TVG_FLAGS_POSITIVE));
    free_vector(vector);

    vector = alloc_vector(TVG_FLAGS_NONZERO | TVG_FLAGS_POSITIVE);
    assert(vector != NULL);
    assert(vector->flags == (TVG_FLAGS_NONZERO | TVG_FLAGS_POSITIVE));
    free_vector(vector);

    vector = alloc_vector(0x80000000);
    assert(vector == NULL);
}

static void test_alloc_graph(void)
{
    struct graph *graph;

    graph = alloc_graph(0);
    assert(graph != NULL);
    assert(graph->flags == 0);
    free_graph(graph);

    graph = alloc_graph(TVG_FLAGS_NONZERO);
    assert(graph != NULL);
    assert(graph->flags == TVG_FLAGS_NONZERO);
    free_graph(graph);

    graph = alloc_graph(TVG_FLAGS_POSITIVE);
    assert(graph != NULL);
    assert(graph->flags == (TVG_FLAGS_NONZERO | TVG_FLAGS_POSITIVE));
    free_graph(graph);

    graph = alloc_graph(TVG_FLAGS_NONZERO | TVG_FLAGS_POSITIVE);
    assert(graph != NULL);
    assert(graph->flags == (TVG_FLAGS_NONZERO | TVG_FLAGS_POSITIVE));
    free_graph(graph);

    graph = alloc_graph(0x80000000);
    assert(graph == NULL);
}

static void test_alloc_tvg(void)
{
    struct tvg *tvg, *tvg2;

    tvg = alloc_tvg(0);
    assert(tvg != NULL);
    tvg2 = grab_tvg(tvg);
    assert(tvg == tvg2);
    free_tvg(tvg);
    free_tvg(tvg);

    tvg = alloc_tvg(TVG_FLAGS_DIRECTED);
    assert(tvg != NULL);
    tvg2 = grab_tvg(tvg);
    assert(tvg == tvg2);
    free_tvg(tvg);
    free_tvg(tvg);

    tvg = alloc_tvg(TVG_FLAGS_STREAMING);
    assert(tvg != NULL);
    tvg2 = grab_tvg(tvg);
    assert(tvg == tvg2);
    free_tvg(tvg);
    free_tvg(tvg);

    tvg = alloc_tvg(0x80000000);
    assert(tvg == NULL);
}

static void test_lookup_graph(void)
{
    struct graph *other_graph;
    struct graph *graph;
    uint64_t max_ts = 0, min_ts = ~0ULL;
    uint64_t ts, delta;
    struct tvg *tvg;
    uint32_t i;
    int ret;

    tvg = alloc_tvg(0);
    assert(tvg != NULL);

    for (i = 0; i < 100; i++)
    {
        graph = alloc_graph(0);
        assert(graph != NULL);
        ts = random_uint64();
        ret = tvg_link_graph(tvg, graph, ts);
        assert(ret);
        free_graph(graph);

        min_ts = MIN(min_ts, ts);
        max_ts = MAX(max_ts, ts);
    }

    for (i = 0; i < 10000; i++)
    {
        ts = random_uint64();

        graph = tvg_lookup_graph_ge(tvg, ts);
        if (ts > max_ts) assert(graph == NULL);
        else
        {
            assert(graph != NULL && graph->ts >= ts);
            if ((other_graph = prev_graph(graph)))
            {
                assert(other_graph->ts < ts);
                free_graph(other_graph);
            }
        }
        free_graph(graph);

        graph = tvg_lookup_graph_le(tvg, ts);
        if (ts < min_ts) assert(graph == NULL);
        else
        {
            assert(graph != NULL && graph->ts <= ts);
            if ((other_graph = next_graph(graph)))
            {
                assert(other_graph->ts > ts);
                free_graph(other_graph);
            }
        }
        free_graph(graph);

        graph = tvg_lookup_graph_near(tvg, ts);
        assert(graph != NULL);
        delta = abs_sub_uint64(ts, graph->ts);
        if ((other_graph = prev_graph(graph)))
        {
            assert(delta <= abs_sub_uint64(ts, other_graph->ts));
            free_graph(other_graph);
        }
        if ((other_graph = next_graph(graph)))
        {
            assert(delta <= abs_sub_uint64(ts, other_graph->ts));
            free_graph(other_graph);
        }
        free_graph(graph);
    }

    free_tvg(tvg);
}

static struct graph *next_free_graph(struct graph *graph)
{
    struct graph *next = next_graph(graph);
    free_graph(graph);
    return next;
}

static void test_next_graph(void)
{
    struct graph *graph;
    struct tvg *tvg;
    uint32_t count;
    uint64_t ts;

    tvg = alloc_random_tvg(0, 100);
    assert(tvg != NULL);

    graph = tvg_lookup_graph_ge(tvg, 0);
    assert(graph != NULL);
    assert(prev_graph(graph) == NULL);
    ts = graph->ts;
    count = 1;
    while ((graph = next_free_graph(graph)))
    {
        assert(graph->ts >= ts);
        ts = graph->ts;
        count++;
    }
    assert(count == 100);

    graph = alloc_graph(0);
    assert(graph != NULL);
    assert(next_graph(graph) == NULL);
    free_graph(graph);

    free_tvg(tvg);
}

static struct graph *prev_free_graph(struct graph *graph)
{
    struct graph *prev = prev_graph(graph);
    free_graph(graph);
    return prev;
}

static void test_prev_graph(void)
{
    struct graph *graph;
    struct tvg *tvg;
    uint32_t count;
    uint64_t ts;

    tvg = alloc_random_tvg(0, 100);
    assert(tvg != NULL);

    graph = tvg_lookup_graph_le(tvg, ~0ULL);
    assert(graph != NULL);
    assert(next_graph(graph) == NULL);
    ts = graph->ts;
    count = 1;
    while ((graph = prev_free_graph(graph)))
    {
        assert(graph->ts <= ts);
        ts = graph->ts;
        count++;
    }
    assert(count == 100);

    graph = alloc_graph(0);
    assert(graph != NULL);
    assert(prev_graph(graph) == NULL);
    free_graph(graph);

    free_tvg(tvg);
}

static void test_graph_get_edge(void)
{
    struct graph *graph = alloc_graph(TVG_FLAGS_DIRECTED);
    struct graph *graph2;
    float weight;
    uint64_t i;
    int ret;

    graph_add_edge(graph, 0, 2, 1.0);
    graph_add_edge(graph, 0, 8, 4.0);
    graph_add_edge(graph, 0, 4, 2.0);
    graph_add_edge(graph, 0, 6, 3.0);

    graph2 = graph_duplicate(graph);
    assert(graph2 != NULL);

    for (i = 0; i < 11; i++)
    {
        weight = graph_get_edge(graph, 0, i);
        if (i < 2 || i > 8 || (i & 1)) assert(weight == 0.0);
        else assert(weight == i / 2.0);

        weight = graph_get_edge(graph2, 0, i);
        if (i < 2 || i > 8 || (i & 1)) assert(weight == 0.0);
        else assert(weight == i / 2.0);
    }

    for (i = 11; i < 21; i++)
    {
        assert(!graph_has_edge(graph, 0, i));
        assert(!graph_has_edge(graph, i, 0));
        assert(graph_get_edge(graph, 0, i) == 0.0);
        assert(graph_get_edge(graph, i, 0) == 0.0);

        assert(!graph_has_edge(graph2, 0, i));
        assert(!graph_has_edge(graph2, i, 0));
        assert(graph_get_edge(graph2, 0, i) == 0.0);
        assert(graph_get_edge(graph2, i, 0) == 0.0);
    }

    for (i = 0; i < 11; i++)
    {
        ret = graph_del_edge(graph, 0, i);
        assert(ret);
        assert(!graph_has_edge(graph, 0, i));
        assert(!graph_has_edge(graph, i, 0));
        assert(graph_get_edge(graph, 0, i) == 0.0);
        assert(graph_get_edge(graph, i, 0) == 0.0);
    }


    for (i = 0; i < 11; i++)
    {
        weight = graph_get_edge(graph2, 0, i);
        if (i < 2 || i > 8 || (i & 1)) assert(weight == 0.0);
        else assert(weight == i / 2.0);
    }

    for (i = 0; i < 11; i++)
    {
        ret = graph_del_edge(graph2, 0, i);
        assert(ret);
        assert(!graph_has_edge(graph2, 0, i));
        assert(!graph_has_edge(graph2, i, 0));
        assert(graph_get_edge(graph2, 0, i) == 0.0);
        assert(graph_get_edge(graph2, i, 0) == 0.0);
    }

    for (i = 11; i < 21; i++)
    {
        ret = graph_del_edge(graph, 0, i);
        assert(ret);
        ret = graph_del_edge(graph, i, 0);
        assert(ret);

        ret = graph_del_edge(graph2, 0, i);
        assert(ret);
        ret = graph_del_edge(graph2, i, 0);
        assert(ret);
    }

    free_graph(graph);
    free_graph(graph2);
}

static void test_vector_get_entry(void)
{
    struct vector *vector = alloc_vector(0);
    struct vector *vector2;
    float weight;
    uint64_t i;
    int ret;

    vector_add_entry(vector, 2, 1.0);
    vector_add_entry(vector, 8, 4.0);
    vector_add_entry(vector, 4, 2.0);
    vector_add_entry(vector, 6, 3.0);

    vector2 = vector_duplicate(vector);
    assert(vector2 != NULL);

    for (i = 0; i < 11; i++)
    {
        weight = vector_get_entry(vector, i);
        if (i < 2 || i > 8 || (i & 1)) assert(weight == 0.0);
        else assert(weight == i / 2.0);

        weight = vector_get_entry(vector2, i);
        if (i < 2 || i > 8 || (i & 1)) assert(weight == 0.0);
        else assert(weight == i / 2.0);
    }

    for (i = 11; i < 21; i++)
    {
        assert(!vector_has_entry(vector, i));
        assert(vector_get_entry(vector, i) == 0.0);

        assert(!vector_has_entry(vector2, i));
        assert(vector_get_entry(vector2, i) == 0.0);
    }

    for (i = 0; i < 11; i++)
    {
        ret = vector_del_entry(vector, i);
        assert(ret);
        assert(!vector_has_entry(vector, i));
        assert(vector_get_entry(vector, i) == 0.0);
    }

    for (i = 0; i < 11; i++)
    {
        weight = vector_get_entry(vector2, i);
        if (i < 2 || i > 8 || (i & 1)) assert(weight == 0.0);
        else assert(weight == i / 2.0);
    }

    for (i = 0; i < 11; i++)
    {
        ret = vector_del_entry(vector2, i);
        assert(ret);
        assert(!vector_has_entry(vector2, i));
        assert(vector_get_entry(vector2, i) == 0.0);
    }


    for (i = 11; i < 21; i++)
    {
        ret = vector_del_entry(vector, i);
        assert(ret);

        ret = vector_del_entry(vector2, i);
        assert(ret);
    }

    free_vector(vector);
    free_vector(vector2);
}

static void test_graph_bits_target(void)
{
    struct graph *graph = alloc_graph(TVG_FLAGS_DIRECTED);
    uint64_t i, j;
    float weight;
    int ret;

    for (i = 0; i < 100 * 100; i++)
        graph_add_edge(graph, i / 100, i % 100, 1.0 + i);

    while (graph_dec_bits_target(graph)) {}

    for (j = 0; j < 12; j++)
    {
        for (i = 0; i < 100 * 100; i++)
        {
            weight = graph_get_edge(graph, i / 100, i % 100);
            assert(weight == 1.0 + i);
        }
        ret = graph_inc_bits_target(graph);
        assert(ret);
    }

    for (j = 0; j < 12; j++)
    {
        for (i = 0; i < 100 * 100; i++)
        {
            weight = graph_get_edge(graph, i / 100, i % 100);
            assert(weight == 1.0 + i);
        }
        ret = graph_dec_bits_target(graph);
        assert(ret);
    }

    ret = graph_dec_bits_target(graph);
    assert(!ret);

    free_graph(graph);
}

static void test_graph_bits_source(void)
{
    struct graph *graph = alloc_graph(TVG_FLAGS_DIRECTED);
    uint64_t i, j;
    float weight;
    int ret;

    for (i = 0; i < 100 * 100; i++)
        graph_add_edge(graph, i / 100, i % 100, 1.0 + i);

    while (graph_dec_bits_source(graph)) {}

    for (j = 0; j < 12; j++)
    {
        for (i = 0; i < 100 * 100; i++)
        {
            weight = graph_get_edge(graph, i / 100, i % 100);
            assert(weight == 1.0 + i);
        }
        ret = graph_inc_bits_source(graph);
        assert(ret);
    }

    for (j = 0; j < 12; j++)
    {
        for (i = 0; i < 100 * 100; i++)
        {
            weight = graph_get_edge(graph, i / 100, i % 100);
            assert(weight == 1.0 + i);
        }
        ret = graph_dec_bits_source(graph);
        assert(ret);
    }

    ret = graph_dec_bits_source(graph);
    assert(!ret);

    free_graph(graph);
}

static void test_graph_optimize(void)
{
    struct graph *graph;
    uint64_t i;

    /* 4 x 4 */
    graph = alloc_graph(TVG_FLAGS_DIRECTED);

    for (i = 0; i < 4 * 4; i++)
        graph_add_edge(graph, i / 4, i % 4, 1.0 + i);

    assert(graph->bits_source == 0);
    assert(graph->bits_target == 0);
    free_graph(graph);

    /* 20 x 20 */
    graph = alloc_graph(TVG_FLAGS_DIRECTED);

    for (i = 0; i < 20 * 20; i++)
        graph_add_edge(graph, i / 20, i % 20, 1.0 + i);

    assert(graph->bits_source == 2);
    assert(graph->bits_target == 1);
    free_graph(graph);

    /* 100 x 100 */
    graph = alloc_graph(TVG_FLAGS_DIRECTED);

    for (i = 0; i < 100 * 100; i++)
        graph_add_edge(graph, i / 100, i % 100, 1.0 + i);

    assert(graph->bits_source == 3);
    assert(graph->bits_target == 3);
    free_graph(graph);
}

static void test_vector_bits(void)
{
    struct vector *vector = alloc_vector(0);
    uint64_t i, j;
    float weight;
    int ret;

    for (i = 0; i < 1000; i++)
        vector_add_entry(vector, i, 1.0 + i);

    while (vector_dec_bits(vector)) {}

    for (j = 0; j < 12; j++)
    {
        for (i = 0; i < 1000; i++)
        {
            weight = vector_get_entry(vector, i);
            assert(weight == 1.0 + i);
        }
        ret = vector_inc_bits(vector);
        assert(ret);
    }

    for (j = 0; j < 12; j++)
    {
        for (i = 0; i < 1000; i++)
        {
            weight = vector_get_entry(vector, i);
            assert(weight == 1.0 + i);
        }
        ret = vector_dec_bits(vector);
        assert(ret);
    }

    ret = vector_dec_bits(vector);
    assert(!ret);

    free_vector(vector);
}

static void test_vector_optimize(void)
{
    struct vector *vector;
    uint64_t i;

    /* 16 */
    vector = alloc_vector(0);

    for (i = 0; i < 16; i++)
        vector_add_entry(vector, i, 1.0 + i);

    assert(vector->bits == 0);
    free_vector(vector);

    /* 128 */
    vector = alloc_vector(0);

    for (i = 0; i < 256; i++)
        vector_add_entry(vector, i, 1.0 + i);

    assert(vector->bits == 3);
    free_vector(vector);

    /* 1024 */
    vector = alloc_vector(0);

    for (i = 0; i < 4096; i++)
        vector_add_entry(vector, i, 1.0 + i);

    assert(vector->bits == 6);
    free_vector(vector);
}

static float weight_func(struct graph *graph, uint64_t ts, void *userdata)
{
    assert(ts == 123);
    assert(userdata == (void *)0xdeadbeef);
    return (float)graph->ts;
}

static void test_extract(void)
{
    struct graph *graph;
    struct tvg *tvg;
    int ret;

    tvg = alloc_tvg(0);
    assert(tvg != NULL);

    graph = alloc_graph(0);
    assert(graph != NULL);
    graph_add_edge(graph, 0, 0, 1.0);
    ret = tvg_link_graph(tvg, graph, 100);
    assert(ret);
    free_graph(graph);

    graph = alloc_graph(0);
    assert(graph != NULL);
    graph_add_edge(graph, 0, 1, 2.0);
    ret = tvg_link_graph(tvg, graph, 200);
    assert(ret);
    free_graph(graph);

    graph = alloc_graph(0);
    assert(graph != NULL);
    graph_add_edge(graph, 0, 2, 3.0);
    ret = tvg_link_graph(tvg, graph, 300);
    assert(ret);
    free_graph(graph);

    graph = tvg_extract(tvg, 123, weight_func, (void *)0xdeadbeef);
    assert(graph != NULL);

    assert(graph_get_edge(graph, 0, 0) == 100.0);
    assert(graph_get_edge(graph, 0, 1) == 400.0);
    assert(graph_get_edge(graph, 0, 2) == 900.0);
    free_graph(graph);

    free_tvg(tvg);
}

static void test_window_sum_edges(void)
{
    struct window *window;
    struct metric *metric;
    struct graph *graph;
    struct tvg *tvg;
    uint32_t i;
    uint64_t ts;
    int ret;

    tvg = alloc_tvg(0);
    assert(tvg != NULL);

    graph = alloc_graph(0);
    assert(graph != NULL);
    graph_add_edge(graph, 0, 0, 1.0);
    ret = tvg_link_graph(tvg, graph, 200);
    assert(ret);
    free_graph(graph);

    graph = alloc_graph(0);
    assert(graph != NULL);
    graph_add_edge(graph, 0, 1, 2.0);
    ret = tvg_link_graph(tvg, graph, 300);
    assert(ret);
    free_graph(graph);

    graph = alloc_graph(0);
    assert(graph != NULL);
    graph_add_edge(graph, 0, 2, 3.0);
    ret = tvg_link_graph(tvg, graph, 400);
    assert(ret);
    free_graph(graph);

    window = tvg_alloc_window(tvg, -100, 100);
    assert(window != NULL);
    metric = window_alloc_metric_sum_edges(window, 0.0);
    assert(metric != NULL);

    for (ts = 0; ts <= 600; ts += 50)
    {
        ret = window_update(window, ts);
        assert(ret);
        graph = metric_sum_edges_get_result(metric);
        assert(graph != NULL);

        if (ts < 100 || ts >= 300) assert(!graph_has_edge(graph, 0, 0));
        else assert(graph_get_edge(graph, 0, 0) == 1.0);

        if (ts < 200 || ts >= 400) assert(!graph_has_edge(graph, 0, 1));
        else assert(graph_get_edge(graph, 0, 1) == 2.0);

        if (ts < 300 || ts >= 500) assert(!graph_has_edge(graph, 0, 2));
        else assert(graph_get_edge(graph, 0, 2) == 3.0);

        free_graph(graph);
    }

    for (i = 0; i < 10000; i++)
    {
        ts = random_uint64() % 700;
        ret = window_update(window, ts);
        assert(ret);
        graph = metric_sum_edges_get_result(metric);
        assert(graph != NULL);

        if (ts < 100 || ts >= 300) assert(!graph_has_edge(graph, 0, 0));
        else assert(graph_get_edge(graph, 0, 0) == 1.0);

        if (ts < 200 || ts >= 400) assert(!graph_has_edge(graph, 0, 1));
        else assert(graph_get_edge(graph, 0, 1) == 2.0);

        if (ts < 300 || ts >= 500) assert(!graph_has_edge(graph, 0, 2));
        else assert(graph_get_edge(graph, 0, 2) == 3.0);

        free_graph(graph);
    }

    free_metric(metric);
    free_window(window);
    free_tvg(tvg);
}

static void test_window_sum_edges_exp(void)
{
    static float beta = 0.9930924954370359;
    struct window *window;
    struct metric *metric;
    struct graph *graph;
    struct tvg *tvg;
    uint64_t ts;
    uint32_t i;
    int ret;

    tvg = alloc_tvg(0);
    assert(tvg != NULL);

    graph = alloc_graph(0);
    assert(graph != NULL);
    graph_add_edge(graph, 0, 0, 1.0);
    ret = tvg_link_graph(tvg, graph, 200);
    assert(ret);
    free_graph(graph);

    graph = alloc_graph(0);
    assert(graph != NULL);
    graph_add_edge(graph, 0, 1, 2.0);
    ret = tvg_link_graph(tvg, graph, 300);
    assert(ret);
    free_graph(graph);

    graph = alloc_graph(0);
    assert(graph != NULL);
    graph_add_edge(graph, 0, 2, 3.0);
    ret = tvg_link_graph(tvg, graph, 400);
    assert(ret);
    free_graph(graph);

    window = tvg_alloc_window(tvg, -1000, 0);
    assert(window != NULL);
    metric = window_alloc_metric_sum_edges_exp(window, 1.0, log(beta), 0.0);
    assert(metric != NULL);

    for (ts = 0; ts <= 600; ts += 50)
    {
        ret = window_update(window, ts);
        assert(ret);
        graph = metric_sum_edges_exp_get_result(metric);
        assert(graph != NULL);

        if (ts < 200) assert(!graph_has_edge(graph, 0, 0));
        else assert(fabs(graph_get_edge(graph, 0, 0) - 1.0 * pow(beta, ts - 200)) < 1e-6);

        if (ts < 300) assert(!graph_has_edge(graph, 0, 1));
        else assert(fabs(graph_get_edge(graph, 0, 1) - 2.0 * pow(beta, ts - 300)) < 1e-6);

        if (ts < 400) assert(!graph_has_edge(graph, 0, 2));
        else assert(fabs(graph_get_edge(graph, 0, 2) - 3.0 * pow(beta, ts - 400)) < 1e-6);

        free_graph(graph);
    }

    /* Seeking back is not really numerically stable. To avoid test
     * failures, manually delete edges that shouldn't exist. */

    for (i = 0; i < 10000; i++)
    {
        ts = random_uint64() % 700;
        ret = window_update(window, ts);
        assert(ret);
        graph = metric_sum_edges_exp_get_result(metric);
        assert(graph != NULL);

        if (ts < 200)
        {
            if (graph_has_edge(graph, 0, 0))
            {
                assert(fabs(graph_get_edge(graph, 0, 0)) < 1e-5);
                graph_del_edge(graph, 0, 0);
            }
        }
        else assert(fabs(graph_get_edge(graph, 0, 0) - 1.0 * pow(beta, ts - 200)) < 1e-6);

        if (ts < 300)
        {
            if (graph_has_edge(graph, 0, 1))
            {
                assert(fabs(graph_get_edge(graph, 0, 1)) < 1e-5);
                graph_del_edge(graph, 0, 1);
            }
        }
        else assert(fabs(graph_get_edge(graph, 0, 1) - 2.0 * pow(beta, ts - 300)) < 1e-6);

        if (ts < 400)
        {
            if (graph_has_edge(graph, 0, 2))
            {
                assert(fabs(graph_get_edge(graph, 0, 2)) < 1e-5);
                graph_del_edge(graph, 0, 2);
            }
        }
        else assert(fabs(graph_get_edge(graph, 0, 2) - 3.0 * pow(beta, ts - 400)) < 1e-6);

        free_graph(graph);
    }

    free_metric(metric);
    free_window(window);
    free_tvg(tvg);
}

static void test_graph_mul_vector(void)
{
    struct vector *vector, *out;
    struct graph *graph;
    uint64_t i, j, k;
    float weight;
    int ret;

    graph = alloc_graph(TVG_FLAGS_DIRECTED);
    vector = alloc_vector(0);

    for (i = 0; i < 100 * 100; i++)
        graph_add_edge(graph, i / 100, i % 100, 1.0 + i);

    for (i = 0; i < 100; i++)
        vector_add_entry(vector, i, 100.0 - i);

    while (graph_dec_bits_source(graph)) {}
    while (graph_dec_bits_target(graph)) {}
    while (vector_dec_bits(vector)) {}

    for (i = 0; i < 6; i++)
    {
        for (j = 0; j < 12; j++)
        {
            out = graph_mul_vector(graph, vector);
            assert(out != NULL);
            for (k = 0; k < 100; k++)
            {
                weight = vector_get_entry(out, k);
                /* expected: sum (100*i+k)*(101-k) from k=1 to 100 */
                assert(fabs(weight - 10100.0 * (50.0 * k + 17.0))/weight < 1e-6);
            }
            free_vector(out);

            ret = vector_inc_bits(vector);
            assert(ret);
        }

        while (vector_dec_bits(vector)) {}
        ret = graph_inc_bits_source(graph);
        assert(ret);
        ret = graph_inc_bits_target(graph);
        assert(ret);
    }

    free_vector(vector);
    free_graph(graph);
}

static void test_graph_vector_for_each_entry(void)
{
    uint64_t i, j, count;
    struct vector *vector;
    struct entry1 *entry;
    struct entry2 *edge;
    struct graph *graph;
    uint64_t num_edges = 0;
    int ret;

    graph = alloc_graph(TVG_FLAGS_DIRECTED);
    vector = alloc_vector(0);

    for (i = 0; i < 100 * 100; i++)
    {
        if (random_float() < sqrt(0.75)) continue;
        graph_add_edge(graph, i / 100, i % 100, 1.0);
        num_edges++;
    }

    for (i = 0; i < 100; i++)
    {
        if (random_float() < sqrt(0.75)) continue;
        vector_add_entry(vector, i, 1.0);
    }

    while (graph_dec_bits_source(graph)) {}
    while (graph_dec_bits_target(graph)) {}
    while (vector_dec_bits(vector)) {}

    for (i = 0; i < 6; i++)
    {
        for (j = 0; j < 12; j++)
        {
            count = 0;
            GRAPH_VECTOR_FOR_EACH_EDGE(graph, edge, vector, entry)
            {
                assert(edge != NULL);
                if (vector_has_entry(vector, edge->target))
                {
                    assert(entry != NULL);
                    assert(entry->index == edge->target);
                }
                else
                {
                    assert(entry == NULL);
                }
                count++;
            }
            assert(count == num_edges);

            ret = vector_inc_bits(vector);
            assert(ret);
        }

        while (vector_dec_bits(vector)) {}
        ret = graph_inc_bits_source(graph);
        assert(ret);
        ret = graph_inc_bits_target(graph);
        assert(ret);
    }

    free_vector(vector);
    free_graph(graph);
}

static void test_power_iteration(void)
{
    const double invsqrt2 = 1.0 / sqrt(2.0);
    struct vector *vector;
    struct graph *graph;
    double eigenvalue;
    int ret;

    graph = alloc_graph(TVG_FLAGS_DIRECTED);
    graph_add_edge(graph, 0, 0, 0.5);
    graph_add_edge(graph, 0, 1, 0.5);
    graph_add_edge(graph, 1, 0, 0.2);
    graph_add_edge(graph, 1, 1, 0.8);

    vector = graph_power_iteration(graph, NULL, 0, 0.0, &eigenvalue);
    assert(vector != NULL);
    assert(fabs(eigenvalue - 1.0) < 1e-7);
    assert(fabs(vector_get_entry(vector, 0) - invsqrt2) < 1e-7);
    assert(fabs(vector_get_entry(vector, 1) - invsqrt2) < 1e-7);
    free_vector(vector);

    ret = graph_mul_const(graph, -1.0);
    assert(ret);

    vector = graph_power_iteration(graph, NULL, 0, 0.0, &eigenvalue);
    assert(vector != NULL);
    assert(fabs(eigenvalue + 1.0) < 1e-7);
    assert(fabs(vector_get_entry(vector, 0) - invsqrt2) < 1e-7);
    assert(fabs(vector_get_entry(vector, 1) - invsqrt2) < 1e-7);
    free_vector(vector);

    free_graph(graph);
}

static void test_load_graphs_from_file(void)
{
    struct tvg *tvg;
    int ret;

    tvg = alloc_tvg(0);
    assert(tvg != NULL);

    ret = tvg_load_graphs_from_file(tvg, "../datasets/example/example-tvg.graph");
    assert(ret);

    free_tvg(tvg);
}

static void test_load_nodes_from_file(void)
{
    struct tvg *tvg;
    int ret;

    tvg = alloc_tvg(0);
    assert(tvg != NULL);

    ret = tvg_set_primary_key(tvg, "a;b;c");
    assert(ret);

    ret = tvg_set_primary_key(tvg, "text");
    assert(ret);

    ret = tvg_load_nodes_from_file(tvg, "../datasets/example/example-tvg.nodes");
    assert(ret);

    free_tvg(tvg);
}

static void test_vector_mul_vector(void)
{
    struct vector *vector1, *vector2;
    double product;
    uint64_t i, j;
    int ret;

    vector1 = alloc_vector(0);
    vector2 = alloc_vector(0);

    for (i = 0; i < 1000; i++)
    {
        vector_add_entry(vector1, i, i);
        vector_add_entry(vector2, i, 1000.0 - i);
    }

    while (vector_dec_bits(vector1)) {}
    while (vector_dec_bits(vector2)) {}

    for (i = 0; i < 12; i++)
    {
        for (j = 0; j < 12; j++)
        {
            product = vector_mul_vector(vector1, vector2);
            /* expected: sum i*(1000-i) from i=0 to 999 */
            assert(product == 166666500.0);

            ret = vector_inc_bits(vector2);
            assert(ret);
        }

        while (vector_dec_bits(vector2)) {}
        ret = vector_inc_bits(vector1);
        assert(ret);
    }

    free_vector(vector1);
    free_vector(vector2);
}

static void test_vector_for_each_entry2(void)
{
    struct vector *vector1, *vector2;
    struct entry1 *entry1, *entry2;
    uint64_t num_entries = 0;
    uint64_t i, j, count;
    int c1, c2, ret;

    vector1 = alloc_vector(0);
    vector2 = alloc_vector(0);

    for (i = 0; i < 1000; i++)
    {
        if ((c1 = (random_float() < sqrt(0.5))))
            vector_add_entry(vector1, i, 1.0);
        if ((c2 = (random_float() < sqrt(0.5))))
            vector_add_entry(vector2, i, 1.0);
        num_entries += (c1 || c2);
    }

    while (vector_dec_bits(vector1)) {}
    while (vector_dec_bits(vector2)) {}

    for (i = 0; i < 12; i++)
    {
        for (j = 0; j < 12; j++)
        {
            count = 0;
            VECTOR_FOR_EACH_ENTRY2(vector1, entry1, vector2, entry2)
            {
                if (entry1 && entry2) assert(entry1->index == entry2->index);
                else if (entry1) assert(!vector_has_entry(vector2, entry1->index));
                else if (entry2) assert(!vector_has_entry(vector1, entry2->index));
                else assert(0);
                count++;
            }
            assert(count == num_entries);

            ret = vector_inc_bits(vector2);
            assert(ret);
        }

        while (vector_dec_bits(vector2)) {}
        ret = vector_inc_bits(vector1);
        assert(ret);
    }

    free_vector(vector1);
    free_vector(vector2);
}

static void test_tvg_for_each_graph(void)
{
    static const uint64_t mid = (1ULL << 63);
    struct graph *graph;
    struct tvg *tvg;
    uint32_t i;
    int ret;

    tvg = alloc_tvg(0);
    assert(tvg != NULL);

    for (i = 0; i < 100; i++)
    {
        graph = alloc_graph(0);
        assert(graph != NULL);
        ret = tvg_link_graph(tvg, graph, random_uint64());
        assert(ret);
        free_graph(graph);
    }

    TVG_FOR_EACH_GRAPH_GE(tvg, graph, mid)
    {
        assert(graph->ts >= mid);
    }
    assert(!graph);

    TVG_FOR_EACH_GRAPH_GE(tvg, graph, mid)
    {
        assert(graph->ts >= mid);
        break;  /* test for possible leaks */
    }
    assert(!graph);

    TVG_FOR_EACH_GRAPH_LE_REV(tvg, graph, mid)
    {
        assert(graph->ts <= mid);
    }
    assert(!graph);

    TVG_FOR_EACH_GRAPH_LE_REV(tvg, graph, mid)
    {
        assert(graph->ts <= mid);
        break;  /* test for possible leaks */
    }
    assert(!graph);

    free_tvg(tvg);
}

static int _bfs_callback(struct graph *graph, const struct bfs_entry *entry, void *userdata)
{
    static const struct bfs_entry expected[] =
    {
        {0.0, 0, ~0ULL, 0},
        {1.0, 1,     0, 1},
        {2.0, 2,     1, 2},
        {3.0, 3,     2, 3},
        {3.5, 3,     2, 4},
    };
    size_t *state = userdata;
    assert(*state < sizeof(expected)/sizeof(expected[0]));
    assert(!memcmp(&expected[*state], entry, sizeof(*entry)));
    (*state)++;
    return 0;
}

static void test_graph_bfs(void)
{
    struct graph *graph;
    size_t state;
    int ret;

    graph = alloc_graph(TVG_FLAGS_DIRECTED);

    graph_set_edge(graph, 0, 1, 1.0);
    graph_set_edge(graph, 1, 2, 1.0);
    graph_set_edge(graph, 2, 3, 1.0);
    graph_set_edge(graph, 3, 4, 1.5);
    graph_set_edge(graph, 2, 4, 1.5);

    state = 0;
    ret = graph_bfs(graph, 0, 0, _bfs_callback, &state);
    assert(ret == 1);
    assert(state == 5);

    state = 0;
    ret = graph_bfs(graph, 0, 1, _bfs_callback, &state);
    assert(ret == 1);
    assert(state == 5);

    free_graph(graph);
}

struct sample
{
    struct avl_entry entry;
    uint64_t ts;
};

static int _sample_compar(const void *a, const void *b, void *userdata)
{
    const struct sample *sa = AVL_ENTRY(a, struct sample, entry);
    const struct sample *sb = AVL_ENTRY(b, struct sample, entry);
    assert(userdata == (void *)0xdeadbeef);
    return COMPARE(sa->ts, sb->ts);
}

static int _sample_lookup(const void *a, const void *b, void *userdata)
{
    const struct sample *sa = AVL_ENTRY(a, struct sample, entry);
    const uint64_t *b_ts = b;
    assert(userdata == (void *)0xdeadbeef);
    return COMPARE(sa->ts, *b_ts);
}

static void test_avl_tree(void)
{
    uint64_t max_ts = 0, min_ts = ~0ULL;
    struct sample *sample, *next_sample;
    struct avl_tree tree;
    uint64_t ts;
    uint32_t i;

    avl_init(&tree, _sample_compar, _sample_lookup, (void *)0xdeadbeef);
    avl_assert_valid(&tree);

    ts = 0;
    sample = AVL_LOOKUP_GE(&tree, &ts, struct sample, entry);
    assert(!sample);

    ts = ~0ULL;
    sample = AVL_LOOKUP_LE(&tree, &ts, struct sample, entry);
    assert(!sample);

    i = 0;
    AVL_FOR_EACH(sample, &tree, struct sample, entry)
    {
        i++;
    }
    assert(i == 0);

    i = 0;
    AVL_FOR_EACH_SAFE(sample, next_sample, &tree, struct sample, entry)
    {
        i++;
    }
    assert(i == 0);

    i = 0;
    AVL_FOR_EACH_POSTORDER(sample, &tree, struct sample, entry)
    {
        i++;
    }
    assert(i == 0);

    i = 0;
    AVL_FOR_EACH_POSTORDER_SAFE(sample, next_sample, &tree, struct sample, entry)
    {
        i++;
    }
    assert(i == 0);

    for (i = 0; i < 100; i++)
    {
        ts = random_uint64();
        sample = malloc(sizeof(*sample));
        sample->ts = ts;
        avl_insert(&tree, &sample->entry, 1);
        avl_assert_valid(&tree);
        min_ts = MIN(min_ts, ts);
        max_ts = MAX(max_ts, ts);
    }

    ts = min_ts;
    i = 0;
    AVL_FOR_EACH(sample, &tree, struct sample, entry)
    {
        assert(sample->ts >= ts);
        ts = sample->ts;
        i++;
    }
    assert(ts == max_ts);
    assert(i == 100);

    i = 0;
    AVL_FOR_EACH_POSTORDER(sample, &tree, struct sample, entry)
    {
        assert(sample->ts >= min_ts);
        assert(sample->ts <= max_ts);
        i++;
    }
    assert(i == 100);

    ts = min_ts;
    sample = AVL_LOOKUP(&tree, &ts, struct sample, entry);
    assert(sample != NULL);
    assert(sample->ts == min_ts);

    ts = max_ts;
    sample = AVL_LOOKUP(&tree, &ts, struct sample, entry);
    assert(sample != NULL);
    assert(sample->ts == max_ts);

    for (i = 0; i < 10000; i++)
    {
        ts = random_uint64();

        sample = AVL_LOOKUP_GE(&tree, &ts, struct sample, entry);
        if (ts > max_ts) assert(sample == NULL);
        else
        {
            assert(sample != NULL);
            assert(sample->ts >= ts);
            if ((sample = AVL_PREV(sample, &tree, struct sample, entry)))
                assert(sample->ts < ts);
        }

        sample = AVL_LOOKUP_LE(&tree, &ts, struct sample, entry);
        if (ts < min_ts) assert(sample == NULL);
        else
        {
            assert(sample != NULL);
            assert(sample->ts <= ts);
            if ((sample = AVL_NEXT(sample, &tree, struct sample, entry)))
                assert(sample->ts > ts);
        }
    }

    for (i = 0; i < 50; i++)
    {
        ts = random_uint64();

        sample = AVL_LOOKUP_GE(&tree, &ts, struct sample, entry);
        if (sample)
        {
            next_sample = AVL_NEXT(sample, &tree, struct sample, entry);
            avl_remove(&sample->entry);
            free(sample);
            avl_assert_valid(&tree);
            if (next_sample && (sample = AVL_PREV(next_sample, &tree, struct sample, entry)))
                assert(sample->ts < ts);
        }
    }

    for (i = 0; i < 50; i++)
    {
        ts = random_uint64();

        sample = AVL_LOOKUP_LE(&tree, &ts, struct sample, entry);
        assert(!sample || sample->ts <= ts);
        next_sample = malloc(sizeof(*next_sample));
        next_sample->ts = ts;
        avl_add_after(&tree, sample ? &sample->entry : NULL, &next_sample->entry);
        avl_assert_valid(&tree);

        ts = random_uint64();

        sample = AVL_LOOKUP_GE(&tree, &ts, struct sample, entry);
        assert(!sample || sample->ts >= ts);
        next_sample = malloc(sizeof(*next_sample));
        next_sample->ts = ts;
        avl_add_before(&tree, sample ? &sample->entry : NULL, &next_sample->entry);
        avl_assert_valid(&tree);
    }

    AVL_FOR_EACH_SAFE(sample, next_sample, &tree, struct sample, entry)
    {
        avl_remove(&sample->entry);
        free(sample);
    }
}

int main(void)
{
    srand((unsigned int)time(NULL));

    if (!init_libtvg(LIBTVG_API_VERSION))
    {
        fprintf(stderr, "Incompatible libtvg library! Try to run 'make'.\n");
        exit(1);
    }

    test_alloc_vector();
    test_alloc_graph();
    test_alloc_tvg();
    test_lookup_graph();
    test_next_graph();
    test_prev_graph();
    test_graph_get_edge();
    test_vector_get_entry();
    test_graph_bits_target();
    test_graph_bits_source();
    test_graph_optimize();
    test_vector_bits();
    test_vector_optimize();
    test_extract();
    test_window_sum_edges();
    test_window_sum_edges_exp();
    test_graph_mul_vector();
    test_graph_vector_for_each_entry();
    test_power_iteration();
    test_load_graphs_from_file();
    test_load_nodes_from_file();
    test_vector_mul_vector();
    test_vector_for_each_entry2();
    test_tvg_for_each_graph();
    test_graph_bfs();
    test_avl_tree();

    fprintf(stderr, "No test failures found\n");
}
