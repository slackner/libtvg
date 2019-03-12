/*
 * Time-varying graph library
 *
 * Copyright (c) 2018-2019 Sebastian Lackner
 */

#include <time.h>
#include <math.h>

#include "tvg.h"
#include "internal.h"

/* replace random_float() to avoid export from libtvg */
static float _random_float(void) { return (float)rand() / (float)(RAND_MAX); }
#define random_float _random_float

/* helper for test_next_graph and test_prev_graph */
static struct tvg *alloc_random_tvg(uint32_t flags, uint32_t count)
{
    struct graph *graph;
    struct tvg *tvg;
    uint32_t i;

    tvg = alloc_tvg(flags);
    assert(tvg != NULL);

    for (i = 0; i < count; i++)
    {
        graph = tvg_alloc_graph(tvg, random_float());
        assert(graph != NULL);
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
    float max_ts = 0.0, min_ts = 1.0;
    float ts, delta;
    struct tvg *tvg;
    uint32_t i;

    tvg = alloc_tvg(0);
    assert(tvg != NULL);

    for (i = 0; i < 100; i++)
    {
        ts = random_float();
        graph = tvg_alloc_graph(tvg, ts);
        assert(graph != NULL);
        min_ts = MIN(min_ts, ts);
        max_ts = MAX(max_ts, ts);
        free_graph(graph);
    }

    for (i = 0; i < 10000; i++)
    {
        ts = random_float();

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
        delta = fabs(graph->ts - ts);
        if ((other_graph = prev_graph(graph)))
        {
            assert(delta <= fabs(other_graph->ts - ts));
            free_graph(other_graph);
        }
        if ((other_graph = next_graph(graph)))
        {
            assert(delta <= fabs(other_graph->ts - ts));
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
    float ts;

    tvg = alloc_random_tvg(0, 100);
    assert(tvg != NULL);

    graph = tvg_lookup_graph_ge(tvg, 0.0);
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
    float ts;

    tvg = alloc_random_tvg(0, 100);
    assert(tvg != NULL);

    graph = tvg_lookup_graph_le(tvg, 1.0);
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
    struct tvg *tvg = alloc_tvg(TVG_FLAGS_DIRECTED);
    struct graph *graph = tvg_alloc_graph(tvg, 0.0);
    float weight;
    uint64_t i;

    graph_add_edge(graph, 0, 2, 1.0);
    graph_add_edge(graph, 0, 8, 4.0);
    graph_add_edge(graph, 0, 4, 2.0);
    graph_add_edge(graph, 0, 6, 3.0);

    for (i = 0; i < 11; i++)
    {
        weight = graph_get_edge(graph, 0, i);
        if (i < 2 || i > 8 || (i & 1)) assert(weight == 0.0);
        else assert(weight == i / 2.0);
    }

    for (i = 11; i < 21; i++)
    {
        assert(!graph_has_edge(graph, 0, i));
        assert(!graph_has_edge(graph, i, 0));
        assert(graph_get_edge(graph, 0, i) == 0.0);
        assert(graph_get_edge(graph, i, 0) == 0.0);
    }

    for (i = 0; i < 11; i++)
    {
        graph_del_edge(graph, 0, i);
        assert(!graph_has_edge(graph, 0, i));
        assert(!graph_has_edge(graph, i, 0));
        assert(graph_get_edge(graph, 0, i) == 0.0);
        assert(graph_get_edge(graph, i, 0) == 0.0);
    }

    for (i = 11; i < 21; i++)
    {
        graph_del_edge(graph, 0, i);
        graph_del_edge(graph, i, 0);
    }

    free_graph(graph);
    free_tvg(tvg);
}

static void test_vector_get_entry(void)
{
    struct vector *vector = alloc_vector(0);
    float weight;
    uint64_t i;

    vector_add_entry(vector, 2, 1.0);
    vector_add_entry(vector, 8, 4.0);
    vector_add_entry(vector, 4, 2.0);
    vector_add_entry(vector, 6, 3.0);

    for (i = 0; i < 11; i++)
    {
        weight = vector_get_entry(vector, i);
        if (i < 2 || i > 8 || (i & 1)) assert(weight == 0.0);
        else assert(weight == i / 2.0);
    }

    for (i = 11; i < 21; i++)
    {
        assert(!vector_has_entry(vector, i));
        assert(vector_get_entry(vector, i) == 0.0);
    }

    for (i = 0; i < 11; i++)
    {
        vector_del_entry(vector, i);
        assert(!vector_has_entry(vector, i));
        assert(vector_get_entry(vector, i) == 0.0);
    }

    for (i = 11; i < 21; i++)
    {
        vector_del_entry(vector, i);
    }

    free_vector(vector);
}

static void test_graph_bits_target(void)
{
    struct tvg *tvg = alloc_tvg(TVG_FLAGS_DIRECTED);
    struct graph *graph = tvg_alloc_graph(tvg, 0.0);
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
    free_tvg(tvg);
}

static void test_graph_bits_source(void)
{
    struct tvg *tvg = alloc_tvg(TVG_FLAGS_DIRECTED);
    struct graph *graph = tvg_alloc_graph(tvg, 0.0);
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
    free_tvg(tvg);
}

static void test_graph_optimize(void)
{
    struct tvg *tvg = alloc_tvg(TVG_FLAGS_DIRECTED);
    struct graph *graph;
    uint64_t i;

    /* 4 x 4 */
    graph = tvg_alloc_graph(tvg, 0.0);

    for (i = 0; i < 4 * 4; i++)
        graph_add_edge(graph, i / 4, i % 4, 1.0 + i);

    assert(graph->bits_source == 0);
    assert(graph->bits_target == 0);
    free_graph(graph);

    /* 20 x 20 */
    graph = tvg_alloc_graph(tvg, 0.0);

    for (i = 0; i < 20 * 20; i++)
        graph_add_edge(graph, i / 20, i % 20, 1.0 + i);

    assert(graph->bits_source == 2);
    assert(graph->bits_target == 1);
    free_graph(graph);

    /* 100 x 100 */
    graph = tvg_alloc_graph(tvg, 0.0);

    for (i = 0; i < 100 * 100; i++)
        graph_add_edge(graph, i / 100, i % 100, 1.0 + i);

    assert(graph->bits_source == 3);
    assert(graph->bits_target == 3);
    free_graph(graph);

    free_tvg(tvg);
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

static float weight_func(struct graph *graph, float ts, void *userdata)
{
    assert(ts == 123.0);
    assert(userdata == (void *)0xdeadbeef);
    return graph->ts;
}

static void test_extract(void)
{
    struct graph *graph;
    struct tvg *tvg;

    tvg = alloc_tvg(0);
    assert(tvg != NULL);

    graph = tvg_alloc_graph(tvg, 100.0);
    assert(graph != NULL);
    graph_add_edge(graph, 0, 0, 1.0);
    free_graph(graph);

    graph = tvg_alloc_graph(tvg, 200.0);
    assert(graph != NULL);
    graph_add_edge(graph, 0, 1, 2.0);
    free_graph(graph);

    graph = tvg_alloc_graph(tvg, 300.0);
    assert(graph != NULL);
    graph_add_edge(graph, 0, 2, 3.0);
    free_graph(graph);

    graph = tvg_extract(tvg, 123.0, weight_func, (void *)0xdeadbeef);
    assert(graph != NULL);

    assert(graph_get_edge(graph, 0, 0) == 100.0);
    assert(graph_get_edge(graph, 0, 1) == 400.0);
    assert(graph_get_edge(graph, 0, 2) == 900.0);
    free_graph(graph);

    free_tvg(tvg);
}

static void test_window_rect(void)
{
    struct window *window;
    struct graph *graph;
    struct tvg *tvg;
    uint32_t i;
    float ts;

    tvg = alloc_tvg(0);
    assert(tvg != NULL);

    graph = tvg_alloc_graph(tvg, 100.0);
    assert(graph != NULL);
    graph_add_edge(graph, 0, 0, 1.0);
    free_graph(graph);

    graph = tvg_alloc_graph(tvg, 200.0);
    assert(graph != NULL);
    graph_add_edge(graph, 0, 1, 2.0);
    free_graph(graph);

    graph = tvg_alloc_graph(tvg, 300.0);
    assert(graph != NULL);
    graph_add_edge(graph, 0, 2, 3.0);
    free_graph(graph);

    window = tvg_alloc_window_rect(tvg, -100.0, 100.0);
    assert(window != NULL);

    for (ts = -100.0; ts <= 500.0; ts += 50.0)
    {
        graph = window_update(window, ts);
        assert(graph != NULL);

        if (ts < 0.0 || ts >= 200.0) assert(!graph_has_edge(graph, 0, 0));
        else assert(graph_get_edge(graph, 0, 0) == 1.0);

        if (ts < 100.0 || ts >= 300.0) assert(!graph_has_edge(graph, 0, 1));
        else assert(graph_get_edge(graph, 0, 1) == 2.0);

        if (ts < 200.0 || ts >= 400.0) assert(!graph_has_edge(graph, 0, 2));
        else assert(graph_get_edge(graph, 0, 2) == 3.0);

        free_graph(graph);
    }

    for (i = 0; i < 10000; i++)
    {
        ts = random_float() * 600.0 - 100.0;

        graph = window_update(window, ts);
        assert(graph != NULL);

        if (ts < 0.0 || ts >= 200.0) assert(!graph_has_edge(graph, 0, 0));
        else assert(graph_get_edge(graph, 0, 0) == 1.0);

        if (ts < 100.0 || ts >= 300.0) assert(!graph_has_edge(graph, 0, 1));
        else assert(graph_get_edge(graph, 0, 1) == 2.0);

        if (ts < 200.0 || ts >= 400.0) assert(!graph_has_edge(graph, 0, 2));
        else assert(graph_get_edge(graph, 0, 2) == 3.0);

        free_graph(graph);
    }

    free_window(window);
    free_tvg(tvg);
}

static void test_window_decay(void)
{
    static float beta = 0.9930924954370359;
    struct window *window;
    struct graph *graph;
    struct tvg *tvg;
    uint32_t i;
    float ts;

    tvg = alloc_tvg(0);
    assert(tvg != NULL);

    graph = tvg_alloc_graph(tvg, 100.0);
    assert(graph != NULL);
    graph_add_edge(graph, 0, 0, 1.0);
    free_graph(graph);

    graph = tvg_alloc_graph(tvg, 200.0);
    assert(graph != NULL);
    graph_add_edge(graph, 0, 1, 2.0);
    free_graph(graph);

    graph = tvg_alloc_graph(tvg, 300.0);
    assert(graph != NULL);
    graph_add_edge(graph, 0, 2, 3.0);
    free_graph(graph);

    window = tvg_alloc_window_decay(tvg, 1000.0, log(beta));
    assert(window != NULL);

    for (ts = -100.0; ts <= 500.0; ts += 50.0)
    {
        graph = window_update(window, ts);
        assert(graph != NULL);

        if (ts < 100.0) assert(!graph_has_edge(graph, 0, 0));
        else assert(fabs(graph_get_edge(graph, 0, 0) - 1.0 * pow(beta, ts - 100.0)) < 1e-6);

        if (ts < 200.0) assert(!graph_has_edge(graph, 0, 1));
        else assert(fabs(graph_get_edge(graph, 0, 1) - 2.0 * pow(beta, ts - 200.0)) < 1e-6);

        if (ts < 300.0) assert(!graph_has_edge(graph, 0, 2));
        else assert(fabs(graph_get_edge(graph, 0, 2) - 3.0 * pow(beta, ts - 300.0)) < 1e-6);

        free_graph(graph);
    }

    /* Seeking back is not really numerically stable. To avoid test
     * failures, manually delete edges that shouldn't exist. */

    for (i = 0; i < 10000; i++)
    {
        ts = random_float() * 600.0 - 100.0;

        graph = window_update(window, ts);
        assert(graph != NULL);

        if (ts < 100.0)
        {
            if (graph_has_edge(graph, 0, 0))
            {
                assert(fabs(graph_get_edge(graph, 0, 0)) < 1e-5);
                graph_del_edge(window->result, 0, 0);
            }
        }
        else assert(fabs(graph_get_edge(graph, 0, 0) - 1.0 * pow(beta, ts - 100.0)) < 1e-6);

        if (ts < 200.0)
        {
            if (graph_has_edge(graph, 0, 1))
            {
                assert(fabs(graph_get_edge(graph, 0, 1)) < 1e-5);
                graph_del_edge(window->result, 0, 1);
            }
        }
        else assert(fabs(graph_get_edge(graph, 0, 1) - 2.0 * pow(beta, ts - 200.0)) < 1e-6);

        if (ts < 300.0)
        {
            if (graph_has_edge(graph, 0, 2))
            {
                assert(fabs(graph_get_edge(graph, 0, 2)) < 1e-5);
                graph_del_edge(window->result, 0, 2);
            }
        }
        else assert(fabs(graph_get_edge(graph, 0, 2) - 3.0 * pow(beta, ts - 300.0)) < 1e-6);

        free_graph(graph);
    }

    free_window(window);
    free_tvg(tvg);
}

static void test_graph_mul_vector(void)
{
    struct vector *vector, *out;
    struct graph *graph;
    float weight;
    uint64_t i, j;
    int ret;

    graph = alloc_graph(TVG_FLAGS_DIRECTED);
    vector = alloc_vector(0);

    for (i = 0; i < 100 * 100; i++)
        graph_add_edge(graph, i / 100, i % 100, 1.0 + i);

    for (i = 0; i < 100; i++)
        vector_add_entry(vector, i, 100.0 - i);

    while (graph_dec_bits_source(graph)) {}
    while (graph_dec_bits_target(graph)) {}

    for (j = 0; j < 6; j++)
    {
        out = graph_mul_vector(graph, vector);
        assert(out != NULL);

        for (i = 0; i < 100; i++)
        {
            weight = vector_get_entry(out, i);
            /* expected: sum (100*i+k)*(101-k) from k=1 to 100 */
            assert(fabs(weight - 10100.0 * (50.0 * i + 17.0))/weight < 1e-6);
        }

        free_vector(out);

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

    graph = alloc_graph(TVG_FLAGS_DIRECTED);
    graph_add_edge(graph, 0, 0, 0.5);
    graph_add_edge(graph, 0, 1, 0.5);
    graph_add_edge(graph, 1, 0, 0.2);
    graph_add_edge(graph, 1, 1, 0.8);

    vector = graph_power_iteration(graph, 0, &eigenvalue);
    assert(vector != NULL);
    assert(fabs(eigenvalue - 1.0) < 1e-7);
    assert(fabs(vector_get_entry(vector, 0) - invsqrt2) < 1e-7);
    assert(fabs(vector_get_entry(vector, 1) - invsqrt2) < 1e-7);
    free_vector(vector);

    graph_mul_const(graph, -1.0);

    vector = graph_power_iteration(graph, 0, &eigenvalue);
    assert(vector != NULL);
    assert(fabs(eigenvalue + 1.0) < 1e-7);
    assert(fabs(vector_get_entry(vector, 0) - invsqrt2) < 1e-7);
    assert(fabs(vector_get_entry(vector, 1) - invsqrt2) < 1e-7);
    free_vector(vector);

    free_graph(graph);
}

static void test_load_graphs(void)
{
    struct tvg *tvg;
    int ret;

    tvg = alloc_tvg(0);
    assert(tvg != NULL);

    ret = tvg_load_graphs(tvg, "../data/example-tvg.graph");
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

int main(void)
{
    srand((unsigned int)time(NULL));

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
    test_window_rect();
    test_window_decay();
    test_graph_mul_vector();
    test_power_iteration();
    test_load_graphs();
    test_vector_mul_vector();

    fprintf(stderr, "No test failures found\n");
}
