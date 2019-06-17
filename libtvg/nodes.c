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
    list_init(&node->attributes);

    return node;
}

struct node *grab_node(struct node *node)
{
    if (node) __sync_fetch_and_add(&node->refcount, 1);
    return node;
}

void free_node(struct node *node)
{
    struct attribute *attr, *next_attr;

    if (!node) return;
    if (__sync_sub_and_fetch(&node->refcount, 1)) return;

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

    avl_remove(&node->entry_ind);
    avl_remove(&node->entry_key);  /* only if primary key is set */
    node->tvg = NULL;
    free_node(node);
}

int node_set_attribute_internal(struct node *node, const char *key, size_t keylen, const char *value)
{
    struct attribute *attr, *other_attr;
    struct tvg *tvg;
    int res;

    if (!keylen)
        return 0;

    if ((tvg = node->tvg))
    {
        /* Don't allow to change primary key after adding to a TVG object. */
        LIST_FOR_EACH(attr, &tvg->primary_key, struct attribute, entry)
        {
            if (!strncmp(attr->key, key, keylen) && attr->key[keylen] == '\0')
                return 0;
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
