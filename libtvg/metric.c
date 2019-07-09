/*
 * Time-varying graph library
 * Metric functions.
 *
 * Copyright (c) 2019 Sebastian Lackner
 */

#include "tvg.h"
#include "internal.h"

struct edge_stability
{
    struct list entry;
    uint64_t source;
    uint64_t target;
    float    value1;
    float    value2;
};

struct node_stability
{
    struct list entry;
    uint64_t index;
    float    value1;
    float    value2;
};

static int _sort_edge_stability(const void *a, const void *b, void *userdata)
{
    const struct edge_stability *sa = a, *sb = b;
    int res;

    if ((res = COMPARE(sa->value1, sb->value1))) return res;
    return COMPARE(sa->value2, sb->value2);
}

static int _sort_node_stability(const void *a, const void *b, void *userdata)
{
    const struct node_stability *sa = a, *sb = b;
    int res;

    if ((res = COMPARE(sa->value1, sb->value1))) return res;
    return COMPARE(sa->value2, sb->value2);
}

struct graph *metric_graph_avg(struct graph **graphs, uint64_t num_graphs)
{
    struct graph *result;
    uint32_t graph_flags;
    uint64_t i;

    if (!num_graphs)
        return NULL;

    graph_flags = graphs[0]->flags & TVG_FLAGS_DIRECTED;
    for (i = 1; i < num_graphs; i++)
    {
        if ((graph_flags ^ graphs[i]->flags) & TVG_FLAGS_DIRECTED)
            return NULL;
    }

    if (!(result = alloc_graph(graph_flags)))
        return NULL;

    for (i = 0; i < num_graphs; i++)
    {
        if (!graph_add_graph(result, graphs[i], 1.0))
            goto error;
    }

    if (!graph_mul_const(result, 1.0 / num_graphs))
        goto error;

    return result;

error:
    free_graph(result);
    return NULL;
}

struct vector *metric_vector_avg(struct vector **vectors, uint64_t num_vectors)
{
    struct vector *result;
    uint64_t i;

    if (!num_vectors)
        return NULL;

    if (!(result = alloc_vector(0)))
        return NULL;

    for (i = 0; i < num_vectors; i++)
    {
        if (!vector_add_vector(result, vectors[i], 1.0))
            goto error;
    }

    if (!vector_mul_const(result, 1.0 / num_vectors))
        goto error;

    return result;

error:
    free_vector(result);
    return NULL;
}

struct graph *metric_graph_std(struct graph **graphs, uint64_t num_graphs)
{
    struct entry2 *edge1, *edge2;
    struct graph *average;
    struct graph *result;
    float weight;
    uint64_t i;

    if (num_graphs < 2)
        return NULL;

    if (!(average = metric_graph_avg(graphs, num_graphs)))
        return NULL;

    if (!(result = alloc_graph(average->flags & TVG_FLAGS_DIRECTED)))
    {
        free_graph(average);
        return NULL;
    }

    for (i = 0; i < num_graphs; i++)
    {
        GRAPH_FOR_EACH_EDGE2(graphs[i], edge1, average, edge2)
        {
            if (edge1 && edge2)
            {
                weight = edge1->weight - edge2->weight;
                if (!graph_add_edge(result, edge1->source, edge1->target, weight * weight))
                    goto error;
            }
            else if (edge1)
            {
                if (!graph_add_edge(result, edge1->source, edge1->target,
                                    edge1->weight * edge1->weight))
                    goto error;
            }
            else
            {
                if (!graph_add_edge(result, edge2->source, edge2->target,
                                    edge2->weight * edge2->weight))
                    goto error;
            }
        }
    }

    if (!graph_mul_const(result, 1.0 / (num_graphs - 1)))
        goto error;

    GRAPH_FOR_EACH_EDGE(result, edge1)
    {
        edge1->weight = sqrt(edge1->weight);
    }

    free_graph(average);
    return result;

error:
    free_graph(average);
    free_graph(result);
    return NULL;
}

struct vector *metric_vector_std(struct vector **vectors, uint64_t num_vectors)
{
    struct entry1 *entry1, *entry2;
    struct vector *average;
    struct vector *result;
    float weight;
    uint64_t i;

    if (num_vectors < 2)
        return NULL;

    if (!(average = metric_vector_avg(vectors, num_vectors)))
        return NULL;

    if (!(result = alloc_vector(0)))
    {
        free_vector(average);
        return NULL;
    }

    for (i = 0; i < num_vectors; i++)
    {
        VECTOR_FOR_EACH_ENTRY2(vectors[i], entry1, average, entry2)
        {
            if (entry1 && entry2)
            {
                weight = entry1->weight - entry2->weight;
                if (!vector_add_entry(result, entry1->index, weight * weight))
                    goto error;
            }
            else if (entry1)
            {
                if (!vector_add_entry(result, entry1->index, entry1->weight * entry1->weight))
                    goto error;
            }
            else
            {
                if (!vector_add_entry(result, entry2->index, entry2->weight * entry2->weight))
                    goto error;
            }
        }
    }

    if (!vector_mul_const(result, 1.0 / (num_vectors - 1)))
        goto error;

    VECTOR_FOR_EACH_ENTRY(result, entry1)
    {
        entry1->weight = sqrt(entry1->weight);
    }

    free_vector(average);
    return result;

error:
    free_vector(average);
    free_vector(result);
    return NULL;
}

struct graph *metric_graph_pareto(struct graph *graph1, struct graph *graph2,
                                  int maximize1, int maximize2, float base)
{
    struct edge_stability *next_stability;
    struct edge_stability *stability;
    struct edge_stability *best;
    struct entry2 *edge1, *edge2;
    struct graph *result = NULL;
    uint32_t graph_flags;
    struct array *array;
    float weight = 1.0;
    struct list queue;
    uint64_t i;
    int ret = 0;

    if (!(array = alloc_array(sizeof(struct edge_stability))))
        return NULL;

    GRAPH_FOR_EACH_EDGE2(graph1, edge1, graph2, edge2)
    {
        if (!(stability = array_append_empty(array)))
            goto error;

        if (edge1)
        {
            stability->source = edge1->source;
            stability->target = edge1->target;
            stability->value1 = edge1->weight;
        }
        else
        {
            stability->source = edge2->source;
            stability->target = edge2->target;
            stability->value1 = 0.0;
        }

        stability->value2 = edge2 ? edge2->weight : 0.0;

        if (maximize1) stability->value1 = -stability->value1;
        if (maximize2) stability->value2 = -stability->value2;
    }

    graph_flags = (graph1->flags | graph2->flags) & TVG_FLAGS_DIRECTED;
    if (!(result = alloc_graph(graph_flags | TVG_FLAGS_POSITIVE)))
        goto error;

    array_sort(array, _sort_edge_stability, NULL);

    list_init(&queue);
    for (i = 0; (stability = (struct edge_stability *)array_ptr(array, i)); i++)
    {
        list_add_tail(&queue, &stability->entry);
    }

    while (!list_empty(&queue))
    {
        best = NULL;

        LIST_FOR_EACH_SAFE(stability, next_stability, &queue, struct edge_stability, entry)
        {
            if (!best || stability->value2 < best->value2 ||
                (stability->value1 == best->value1 && stability->value2 == best->value2))
            {
                if (!graph_set_edge(result, stability->source, stability->target, weight))
                    goto error;

                list_remove(&stability->entry);
                best = stability;
            }
        }

        if (base == 0.0)
            weight += 1.0;
        else
            weight *= base;
    }

    ret = 1;

error:
    if (!ret)
    {
        free_graph(result);
        result = NULL;
    }
    free_array(array);
    return result;
}

struct vector *metric_vector_pareto(struct vector *vector1, struct vector *vector2,
                                    int maximize1, int maximize2, float base)
{
    struct node_stability *next_stability;
    struct node_stability *stability;
    struct node_stability *best;
    struct entry1 *entry1, *entry2;
    struct vector *result = NULL;
    struct array *array;
    float weight = 1.0;
    struct list queue;
    uint64_t i;
    int ret = 0;

    if (!(array = alloc_array(sizeof(struct node_stability))))
        return NULL;

    VECTOR_FOR_EACH_ENTRY2(vector1, entry1, vector2, entry2)
    {
        if (!(stability = array_append_empty(array)))
            goto error;

        if (entry1)
        {
            stability->index  = entry1->index;
            stability->value1 = entry1->weight;
        }
        else
        {
            stability->index  = entry2->index;
            stability->value1 = 0.0;
        }

        stability->value2 = entry2 ? entry2->weight : 0.0;

        if (maximize1) stability->value1 = -stability->value1;
        if (maximize2) stability->value2 = -stability->value2;
    }

    if (!(result = alloc_vector(TVG_FLAGS_POSITIVE)))
        goto error;

    array_sort(array, _sort_node_stability, NULL);

    list_init(&queue);
    for (i = 0; (stability = (struct node_stability *)array_ptr(array, i)); i++)
    {
        list_add_tail(&queue, &stability->entry);
    }

    while (!list_empty(&queue))
    {
        best = NULL;

        LIST_FOR_EACH_SAFE(stability, next_stability, &queue, struct node_stability, entry)
        {
            if (!best || stability->value2 < best->value2 ||
                (stability->value1 == best->value1 && stability->value2 == best->value2))
            {
                if (!vector_set_entry(result, stability->index, weight))
                    goto error;

                list_remove(&stability->entry);
                best = stability;
            }
        }

        if (base == 0.0)
            weight += 1.0;
        else
            weight *= base;
    }

    ret = 1;

error:
    if (!ret)
    {
        free_vector(result);
        result = NULL;
    }
    free_array(array);
    return result;
}
