/*
 * Time-varying graph library
 * Linked lists.
 *
 * Copyright (c) 2018-2019 Sebastian Lackner
 */

#ifndef _LIST_H_
#define _LIST_H_

#include <stddef.h>

struct list
{
    struct list *next;
    struct list *prev;
};

/* get struct from embedded 'struct list' datatype */
#define LIST_ENTRY(elem, type, field) ({ \
    const struct list *__ptr = (elem); \
    (type *)((char *)__ptr - offsetof(type, field)); })

/* retrieve the next element in a linked list */
#define LIST_NEXT(cursor, list, type, field) ({ \
    typeof(((type *)0)) __ret = LIST_ENTRY((cursor) ? (cursor)->field.next : (list)->next, type, field); \
    if (&__ret->field == (list)) __ret = NULL; \
    __ret; })

/* retrieve the first element in a linked list */
#define LIST_HEAD(list, type, field) ({ \
    typeof(((type *)0)) __ret = LIST_ENTRY((list)->next, type, field); \
    if (&__ret->field == (list)) __ret = NULL; \
    __ret; })

/* retrieve the previous element in a linked list */
#define LIST_PREV(cursor, list, type, field) ({ \
    typeof(((type *)0)) __ret = LIST_ENTRY((cursor) ? (cursor)->field.prev : (list)->prev, type, field); \
    if (&__ret->field == (list)) __ret = NULL; \
    __ret; })

/* retrieve the last element in a linked list */
#define LIST_TAIL(list, type, field)  ({ \
    typeof(((type *)0)) __ret = LIST_ENTRY((list)->prev, type, field); \
    if (&__ret->field == (list)) __ret = NULL; \
    __ret; })

/* loop over list elements */
#define LIST_FOR_EACH(cursor, list, type, field) \
    for ((cursor) = LIST_ENTRY((list)->next, type, field); \
         &(cursor)->field != (list); \
         (cursor) = LIST_ENTRY((cursor)->field.next, type, field))

/* loop over list elements in reverse order */
#define LIST_FOR_EACH_REV(cursor, list, type, field) \
    for ((cursor) = LIST_ENTRY((list)->prev, type, field); \
         &(cursor)->field != (list); \
         (cursor) = LIST_ENTRY((cursor)->field.prev, type, field))

/* loop over list elements while ensuring that elements can be deleted */
#define LIST_FOR_EACH_SAFE(cursor, cursor2, list, type, field) \
    for ((cursor) = LIST_ENTRY((list)->next, type, field); \
         &(cursor)->field != (list) && \
             ({ (cursor2) = LIST_ENTRY((cursor)->field.next, type, field); 1; }); \
         (cursor) = (cursor2))

/* initialize a list */
static inline void list_init(struct list *list)
{
    list->next = list;
    list->prev = list;
}

/* check if a list is empty */
static inline int list_empty(const struct list *list)
{
    return list->next == list;
}

/* add a new element after the cursor position */
#define list_add_head list_add_after
static inline void list_add_after(struct list *cursor, struct list *entry)
{
    entry->next         = cursor->next;
    entry->prev         = cursor;
    cursor->next->prev  = entry;
    cursor->next        = entry;
}

/* add a new element before the cursor position */
#define list_add_tail list_add_before
static inline void list_add_before(struct list *cursor, struct list *entry)
{
    entry->next         = cursor;
    entry->prev         = cursor->prev;
    cursor->prev->next  = entry;
    cursor->prev        = entry;
}

static inline void list_remove(struct list *cursor)
{
    cursor->next->prev = cursor->prev;
    cursor->prev->next = cursor->next;
}

#endif  /* _LIST_H_ */
