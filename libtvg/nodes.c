/*
 * Time-varying graph library
 * Node functions.
 *
 * Copyright (c) 2019 Sebastian Lackner
 */

#include "tvg.h"
#include "internal.h"

struct node *alloc_node(void)
{
    struct node *node;

    if (!(node = malloc(sizeof(*node))))
        return NULL;

    node->refcount    = 1;
    node->index       = ~0ULL;
    node->tvg         = NULL;
    list_init(&node->entry_ind);
    list_init(&node->entry_key);
    list_init(&node->attributes);

    return node;
}

struct node *grab_node(struct node *node)
{
    if (node) node->refcount++;
    return node;
}

void free_node(struct node *node)
{
    struct attribute *attr, *next_attr;

    if (!node) return;
    if (--node->refcount) return;

    LIST_FOR_EACH_SAFE(attr, next_attr, &node->attributes, struct attribute, entry)
    {
        list_remove(&attr->entry);
        free(attr);
    }

    assert(!node->tvg);
    free(node);
}

void unlink_node(struct node *node)
{
    if (!node || !node->tvg)
        return;

    list_remove(&node->entry_ind);
    list_remove(&node->entry_key);
    node->tvg = NULL;
}

int node_set_attribute_internal(struct node *node, const char *key, size_t keylen, const char *value)
{
    struct attribute *attr, *other_attr;
    struct tvg *tvg;
    int res;

    if ((tvg = node->tvg))
    {
        /* Don't allow to change primary key after adding to a TVG object. */
        LIST_FOR_EACH(attr, &tvg->primary_key, struct attribute, entry)
        {
            if (!strcmp(attr->key, key)) return 0;
        }
    }

    if (memchr(key, ';', keylen) != NULL)
        return 0;

    if (!(attr = malloc(offsetof(struct attribute, buffer[keylen + strlen(value) + 2]))))
        return 0;

    attr->key     = attr->buffer;
    attr->value   = &attr->buffer[keylen + 1];
    memcpy((char *)attr->key, key, keylen);
    ((char *)attr->key)[keylen] = 0;
    strcpy((char *)attr->value, value);

    LIST_FOR_EACH(other_attr, &node->attributes, struct attribute, entry)
    {
        if ((res = strcmp(other_attr->key, attr->key)) >= 0) break;
    }

    list_add_before(&other_attr->entry, &attr->entry);
    if (!res)  /* Attribute keys should be unique */
    {
        list_remove(&other_attr->entry);
        free(other_attr);
    }
    return 1;
}

int node_set_attribute(struct node *node, const char *key, const char *value)
{
    return node_set_attribute_internal(node, key, strlen(key), value);
}

const char *node_get_attribute(struct node *node, const char *key)
{
    struct attribute *attr;

    LIST_FOR_EACH(attr, &node->attributes, struct attribute, entry)
    {
        if (!strcmp(attr->key, key))
            return attr->value;
    }

    return NULL;
}

char **node_get_attributes(struct node *node)
{
    struct attribute *attr;
    size_t len_ptr = sizeof(char *);
    size_t len_str = 0;
    char *buf, **ptr, *str;

    LIST_FOR_EACH(attr, &node->attributes, struct attribute, entry)
    {
        len_str += strlen(attr->key) + 1;
        len_str += strlen(attr->value) + 1;
        len_ptr += 2 * sizeof(char *);
    }

    if (!(buf = malloc(len_ptr + len_str)))
        return NULL;

    ptr = (char **)buf;
    str = buf + len_ptr;

    LIST_FOR_EACH(attr, &node->attributes, struct attribute, entry)
    {
        *ptr++ = str;
        strcpy(str, attr->key);
        str += strlen(str) + 1;

        *ptr++ = str;
        strcpy(str, attr->value);
        str += strlen(str) + 1;
    }

    *ptr = NULL;
    return (char **)buf;
}

int node_equal_key(struct tvg *tvg, struct node *node1, struct node *node2)
{
    struct attribute *attr1, *attr2;

    NODE_FOR_EACH_PRIMARY_ATTRIBUTE2(tvg, node1, attr1, node2, attr2)
    {
        if (attr1 && attr2)
        {
            if (strcmp(attr1->value, attr2->value)) return 0;
        }
        else if (attr1 || attr2)
        {
            return 0;
        }
    }

    return 1;
}

uint32_t node_hash_index(struct tvg *tvg, uint64_t index)
{
    return index % ARRAY_SIZE(tvg->nodes_ind);
}

uint32_t node_hash_primary_key(struct tvg *tvg, struct node *node)
{
    struct attribute *attr;
    uint32_t hash = 5381;
    const char *str;

    if (list_empty(&tvg->primary_key))
        return ~0U;

    NODE_FOR_EACH_PRIMARY_ATTRIBUTE(tvg, node, attr)
    {
        if (!attr) return ~0U;

        str = attr->value;
        while (*str)
        {
            hash = (hash << 5) + hash;
            hash += *str++;
        }

        /* terminating '\0' */
        hash = (hash << 5) + hash;
    }

    return hash % ARRAY_SIZE(tvg->nodes_key);
}