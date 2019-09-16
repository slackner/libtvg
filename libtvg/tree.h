/*
 * Time-varying graph library
 * AVL Tree.
 *
 * Copyright (c) 2019 Sebastian Lackner
 */

#ifndef _TREE_H_
#define _TREE_H_

#include <stddef.h>

struct avl_entry
{
    struct avl_entry *parent;
    struct avl_entry *left;
    struct avl_entry *right;
    int balance;
};

struct avl_tree
{
    struct avl_entry root;
    int (*compar)(const void *, const void *, void *);
    int (*lookup)(const void *, const void *, void *);
    void *userdata;
};

/* get struct from embedded 'struct avl_entry' datatype */
#define AVL_ENTRY(elem, type, field) ({ \
    const struct avl_entry *__ptr = (elem); \
    (type *)((char *)__ptr - offsetof(type, field)); })

/* retrieve the next element in an avl tree */
#define AVL_NEXT(cursor, tree, type, field) ({ \
    typeof(((type *)0)) __ret = AVL_ENTRY((cursor) ? avl_next(&(cursor)->field) : \
                                          avl_head((tree)->root.right), type, field); \
    if (__ret == AVL_ENTRY(NULL, type, field)) __ret = NULL; \
    __ret; })

/* retrieve the first element in an avl tree */
#define AVL_HEAD(tree, type, field) ({ \
    typeof(((type *)0)) __ret = AVL_ENTRY(avl_head((tree)->root.right), type, field); \
    if (__ret == AVL_ENTRY(NULL, type, field)) __ret = NULL; \
    __ret; })

/* retrieve the previous element in an avl tree */
#define AVL_PREV(cursor, tree, type, field) ({ \
    typeof(((type *)0)) __ret = AVL_ENTRY((cursor) ? avl_prev(&(cursor)->field) : \
                                          avl_tail((tree)->root.right), type, field); \
    if (__ret == AVL_ENTRY(NULL, type, field)) __ret = NULL; \
    __ret; })

/* retrieve the last element in an avl tree */
#define AVL_TAIL(tree, type, field)  ({ \
    typeof(((type *)0)) __ret = AVL_ENTRY(avl_tail((tree)->root.right), type, field); \
    if (__ret == AVL_ENTRY(NULL, type, field)) __ret = NULL; \
    __ret; })

/* lookup an element from the avl tree */
#define AVL_LOOKUP(tree, data, type, field) ({ \
    typeof(((type *)0)) __ret = AVL_ENTRY(avl_lookup((tree), data), type, field); \
    if (__ret == AVL_ENTRY(NULL, type, field)) __ret = NULL; \
    __ret; })

/* lookup the first element >= data in an avl tree */
#define AVL_LOOKUP_GE(tree, data, type, field) ({ \
    typeof(((type *)0)) __ret = AVL_ENTRY(avl_lookup_ge((tree), data), type, field); \
    if (__ret == AVL_ENTRY(NULL, type, field)) __ret = NULL; \
    __ret; })

/* lookup the last element <= data in an avl tree */
#define AVL_LOOKUP_LE(tree, data, type, field) ({ \
    typeof(((type *)0)) __ret = AVL_ENTRY(avl_lookup_le((tree), data), type, field); \
    if (__ret == AVL_ENTRY(NULL, type, field)) __ret = NULL; \
    __ret; })

/* loop over avl tree elements */
#define AVL_FOR_EACH(cursor, tree, type, field) \
    for ((cursor) = AVL_ENTRY(avl_head((tree)->root.right), type, field); \
         (cursor) != AVL_ENTRY(NULL, type, field); \
         (cursor) = AVL_ENTRY(avl_next(&(cursor)->field), type, field))

/* loop over avl tree elements while ensuring that elements can be deleted */
#define AVL_FOR_EACH_SAFE(cursor, cursor2, tree, type, field) \
    for ((cursor) = AVL_ENTRY(avl_head((tree)->root.right), type, field); \
         (cursor) != AVL_ENTRY(NULL, type, field) && \
             ({ (cursor2) = AVL_ENTRY(avl_next(&(cursor)->field), type, field); 1; }); \
         (cursor) = (cursor2))

/* loop over avl tree elements in post-order */
#define AVL_FOR_EACH_POSTORDER(cursor, tree, type, field) \
    for ((cursor) = AVL_ENTRY(avl_postorder_head((tree)->root.right), type, field); \
         (cursor) != AVL_ENTRY(NULL, type, field); \
         (cursor) = AVL_ENTRY(avl_postorder_next(&(cursor)->field), type, field))

/* loop over avl tree elements in post-order while ensuring that elements can be deleted */
#define AVL_FOR_EACH_POSTORDER_SAFE(cursor, cursor2, tree, type, field) \
    for ((cursor) = AVL_ENTRY(avl_postorder_head((tree)->root.right), type, field); \
         (cursor) != AVL_ENTRY(NULL, type, field) && \
             ({ (cursor2) = AVL_ENTRY(avl_postorder_next(&(cursor)->field), type, field); 1; }); \
         (cursor) = (cursor2))

/* initialize an avl tree */
static inline void avl_init(struct avl_tree *tree, int (*compar)(const void *, const void *, void *),
                            int (*lookup)(const void *, const void *, void *), void *userdata)
{
    tree->root.parent   = NULL;
    tree->root.left     = (void *)~0UL;
    tree->root.right    = NULL;
    tree->root.balance  = 0;
    tree->compar        = compar;
    tree->lookup        = lookup;
    tree->userdata      = userdata;
}

/* check if an avl tree is empty */
static inline int avl_empty(const struct avl_tree *tree)
{
    return !tree->root.right;
}

static inline int avl_entry_assert_valid(const struct avl_tree *tree, struct avl_entry *entry,
                                         const struct avl_entry *parent, struct avl_entry *entry_min,
                                         struct avl_entry *entry_max)
{
    int left = 0, right = 0;
    int balance;

    if (!entry)
        return 0;

    assert(entry->parent == parent);

    if (entry_min)
        assert(tree->compar(entry, entry_min, tree->userdata) >= 0);

    if (entry_max)
        assert(tree->compar(entry, entry_max, tree->userdata) <= 0);

    if (entry->left)
    {
        assert(tree->compar(entry, entry->left, tree->userdata) >= 0);
        left = avl_entry_assert_valid(tree, entry->left, entry, entry_min, entry);
    }

    if (entry->right)
    {
        assert(tree->compar(entry, entry->right, tree->userdata) <= 0);
        right = avl_entry_assert_valid(tree, entry->right, entry, entry, entry_max);
    }

    balance = right - left;
    assert(entry->balance == balance);
    assert(balance >= -1 && balance <= 1);

    return 1 + ((right > left) ? right : left);
}

/* validate the order and balance of an avl tree */
static inline void avl_assert_valid(const struct avl_tree *tree)
{
    assert(!tree->root.parent);
    assert(tree->root.left == (void *)~0UL);
    assert(!tree->root.balance);
    avl_entry_assert_valid(tree, tree->root.right, &tree->root, NULL, NULL);
}

/* retrieve the first element in an avl tree */
static inline struct avl_entry *avl_head(struct avl_entry *entry)
{
    if (!entry) return NULL;
    while (entry->left) entry = entry->left;
    return entry;
}

/* retrieve the last element in an avl tree */
static inline struct avl_entry *avl_tail(struct avl_entry *entry)
{
    if (!entry) return NULL;
    while (entry->right) entry = entry->right;
    return entry;
}

/* retrieve the next element in an avl tree */
static inline struct avl_entry *avl_next(struct avl_entry *entry)
{
    struct avl_entry *parent;

    if (entry->right) return avl_head(entry->right);
    for (;;)
    {
        parent = entry->parent;
        if (!parent->parent) return NULL;
        if (parent->left == entry) return parent;
        entry = parent;
    }
}

/* retrieve the previous element in an avl tree */
static inline struct avl_entry *avl_prev(struct avl_entry *entry)
{
    struct avl_entry *parent;

    if (entry->left) return avl_tail(entry->left);
    for (;;)
    {
        parent = entry->parent;
        if (!parent->parent) return NULL;
        if (parent->right == entry) return parent;
        entry = parent;
    }
}

/* begin enumerating elements of avl tree in postorder */
static inline struct avl_entry *avl_postorder_head(struct avl_entry *entry)
{
    if (!entry) return NULL;

    for (;;)
    {
        while (entry->left) entry = entry->left;
        if (!entry->right) return entry;
        entry = entry->right;
    }
}

/* get next element of avl tree in postorder */
static inline struct avl_entry *avl_postorder_next(struct avl_entry *entry)
{
    struct avl_entry *parent = entry->parent;

    if (!parent->parent) return NULL;
    if (entry == parent->right || !parent->right) return parent;
    return avl_postorder_head(parent->right);
}

/* lookup an element from the avl tree */
static inline struct avl_entry *avl_lookup(struct avl_tree *tree, const void *data)
{
    struct avl_entry *entry = tree->root.right;
    int res;

    while (entry)
    {
        res = tree->lookup(entry, data, tree->userdata);
        if (!res) break;
        entry = (res < 0) ? entry->right : entry->left;
    }

    return entry;
}

/* lookup the first element >= data in an avl tree */
static inline struct avl_entry *avl_lookup_ge(struct avl_tree *tree, const void *data)
{
    struct avl_entry *entry = tree->root.right;
    struct avl_entry *other;
    int res;

    while (entry)
    {
        res = tree->lookup(entry, data, tree->userdata);
        if (!res) break;
        if (res < 0)
        {
            if (!entry->right) return avl_next(entry);
            entry = entry->right;
        }
        else
        {
            if (!entry->left) return entry;
            entry = entry->left;
        }
    }

    while (entry)
    {
        if (!(other = avl_prev(entry))) break;
        if (tree->lookup(other, data, tree->userdata) < 0) break;
        entry = other;
    }

    return entry;
}

/* lookup the last element <= data in an avl tree */
static inline struct avl_entry *avl_lookup_le(struct avl_tree *tree, const void *data)
{
    struct avl_entry *entry = tree->root.right;
    struct avl_entry *other;
    int res;

    while (entry)
    {
        res = tree->lookup(entry, data, tree->userdata);
        if (!res) break;
        if (res > 0)
        {
            if (!entry->left) return avl_prev(entry);
            entry = entry->left;
        }
        else
        {
            if (!entry->right) return entry;
            entry = entry->right;
        }
    }

    while (entry)
    {
        if (!(other = avl_next(entry))) break;
        if (tree->lookup(other, data, tree->userdata) > 0) break;
        entry = other;
    }

    return entry;
}

static inline struct avl_entry *avl_rotate_left(struct avl_entry *entry)
{
    struct avl_entry *right = entry->right;
    struct avl_entry *right_left = right->left;
    struct avl_entry *parent = entry->parent;

    right->parent   = parent;
    right->left     = entry;
    entry->right    = right_left;
    entry->parent   = right;

    if (right_left)
        right_left->parent = entry;

    if (parent->left == entry)
        parent->left = right;
    else
        parent->right = right;

    right->balance--;
    entry->balance = -right->balance;
    return right;
}

static inline struct avl_entry *avl_rotate_right(struct avl_entry *entry)
{
    struct avl_entry *left = entry->left;
    struct avl_entry *left_right = left->right;
    struct avl_entry *parent = entry->parent;

    left->parent    = parent;
    left->right     = entry;
    entry->left     = left_right;
    entry->parent   = left;

    if (left_right)
        left_right->parent = entry;

    if (parent->left == entry)
        parent->left = left;
    else
        parent->right = left;

    left->balance++;
    entry->balance = -left->balance;
    return left;
}

static inline struct avl_entry *avl_rotate_left_right(struct avl_entry *entry)
{
    struct avl_entry *left = entry->left;
    struct avl_entry *left_right = left->right;
    struct avl_entry *parent = entry->parent;
    struct avl_entry *left_right_right = left_right->right;
    struct avl_entry *left_right_left = left_right->left;

    left_right->parent  = parent;
    entry->left         = left_right_right;
    left->right         = left_right_left;
    left_right->left    = left;
    left_right->right   = entry;
    left->parent        = left_right;
    entry->parent       = left_right;

    if (left_right_right)
        left_right_right->parent = entry;

    if (left_right_left)
        left_right_left->parent = left;

    if (parent->left == entry)
        parent->left = left_right;
    else
        parent->right = left_right;

    if (left_right->balance == 1)
    {
        entry->balance = 0;
        left->balance = -1;
    }
    else if (!left_right->balance)
    {
        entry->balance = 0;
        left->balance = 0;
    }
    else
    {
        entry->balance = 1;
        left->balance = 0;
    }

    left_right->balance = 0;
    return left_right;
}

static inline struct avl_entry *avl_rotate_right_left(struct avl_entry *entry)
{
    struct avl_entry *right = entry->right;
    struct avl_entry *right_left = right->left;
    struct avl_entry *parent = entry->parent;
    struct avl_entry *right_left_left = right_left->left;
    struct avl_entry *right_left_right = right_left->right;

    right_left->parent  = parent;
    entry->right        = right_left_left;
    right->left         = right_left_right;
    right_left->right   = right;
    right_left->left    = entry;
    right->parent       = right_left;
    entry->parent       = right_left;

    if (right_left_left)
        right_left_left->parent = entry;

    if (right_left_right)
        right_left_right->parent = right;

    if (parent->left == entry)
        parent->left = right_left;
    else
        parent->right = right_left;

    if (right_left->balance == -1)
    {
        entry->balance = 0;
        right->balance = 1;
    }
    else if (!right_left->balance)
    {
        entry->balance = 0;
        right->balance = 0;
    }
    else
    {
        entry->balance = -1;
        right->balance = 0;
    }

    right_left->balance = 0;
    return right_left;
}

static inline void avl_insert_balance(struct avl_entry *entry, int balance)
{
    struct avl_entry *parent;

    for (;;)
    {
        balance = (entry->balance += balance);
        if (!balance) break;

        if (balance == -2)
        {
            if (entry->left->balance == -1)
                avl_rotate_right(entry);
            else
                avl_rotate_left_right(entry);
            break;
        }

        if (balance == 2)
        {
            if (entry->right->balance == 1)
                avl_rotate_left(entry);
            else
                avl_rotate_right_left(entry);
            break;
        }

        parent = entry->parent;
        if (!parent->parent) break;
        balance = (parent->left == entry) ? -1 : 1;
        entry = parent;
    }
}

static inline struct avl_entry *avl_insert(struct avl_tree *tree, struct avl_entry *entry, int allow_duplicates)
{
    struct avl_entry *other;
    int res;

    entry->left    = NULL;
    entry->right   = NULL;
    entry->balance = 0;

    if (!tree->root.right)
    {
        tree->root.right = entry;
        entry->parent = &tree->root;
        return NULL;
    }

    other = tree->root.right;
    for (;;)
    {
        res = tree->compar(other, entry, tree->userdata);
        if (!res && !allow_duplicates)
        {
            entry->parent = NULL;
            return other;
        }
        if (res < 0)
        {
            if (!other->right)
            {
                other->right = entry;
                entry->parent = other;
                avl_insert_balance(other, 1);
                return NULL;
            }

            other = other->right;
        }
        else
        {
            if (!other->left)
            {
                other->left = entry;
                entry->parent = other;
                avl_insert_balance(other, -1);
                return NULL;
            }

            other = other->left;
        }
    }
}

static inline void avl_add_after(struct avl_tree *tree, struct avl_entry *cursor, struct avl_entry *entry)
{
    entry->left    = NULL;
    entry->right   = NULL;
    entry->balance = 0;

    if (!cursor)
    {
        if (!tree->root.right)
        {
            tree->root.right = entry;
            entry->parent = &tree->root;
            return;
        }

        cursor = tree->root.right;
        while (cursor->left)
            cursor = cursor->left;

        cursor->left = entry;
        entry->parent = cursor;
        avl_insert_balance(cursor, -1);
    }
    else if (!cursor->right)
    {
        cursor->right = entry;
        entry->parent = cursor;
        avl_insert_balance(cursor, 1);
    }
    else
    {
        cursor = cursor->right;
        while (cursor->left)
            cursor = cursor->left;

        cursor->left = entry;
        entry->parent = cursor;
        avl_insert_balance(cursor, -1);
    }
}

static inline void avl_add_before(struct avl_tree *tree, struct avl_entry *cursor, struct avl_entry *entry)
{
    entry->left    = NULL;
    entry->right   = NULL;
    entry->balance = 0;

    if (!cursor)
    {
        if (!tree->root.right)
        {
            tree->root.right = entry;
            entry->parent = &tree->root;
            return;
        }

        cursor = tree->root.right;
        while (cursor->right)
            cursor = cursor->right;

        cursor->right = entry;
        entry->parent = cursor;
        avl_insert_balance(cursor, 1);
    }
    else if (!cursor->left)
    {
        cursor->left = entry;
        entry->parent = cursor;
        avl_insert_balance(cursor, -1);
    }
    else
    {
        cursor = cursor->left;
        while (cursor->right)
            cursor = cursor->right;

        cursor->right = entry;
        entry->parent = cursor;
        avl_insert_balance(cursor, 1);
    }
}

static inline void avl_remove_balance(struct avl_entry *entry, int balance)
{
    struct avl_entry *parent;

    for (;;)
    {
        balance = (entry->balance += balance);

        if (balance == -2)
        {
            if (entry->left->balance <= 0)
            {
                entry = avl_rotate_right(entry);
                if (entry->balance == 1) break;
            }
            else
            {
                entry = avl_rotate_left_right(entry);
            }
        }
        else if (balance == 2)
        {
            if (entry->right->balance >= 0)
            {
                entry = avl_rotate_left(entry);
                if (entry->balance == -1) break;
            }
            else
            {
                entry = avl_rotate_right_left(entry);
            }
        }
        else if (balance != 0)
            break;

        parent = entry->parent;
        if (!parent->parent) break;
        balance = (parent->left == entry) ? 1 : -1;
        entry = parent;
    }
}

static inline void avl_remove(struct avl_entry *entry)
{
    struct avl_entry *left = entry->left;
    struct avl_entry *right = entry->right;
    struct avl_entry *parent = entry->parent;

    if (!parent)
        return;

    if (!left)
    {
        if (!right)
        {
            if (!parent->parent)
            {
                assert(parent->right == entry);
                parent->right = NULL;
            }
            else if (parent->left == entry)
            {
                parent->left = NULL;
                avl_remove_balance(parent, 1);
            }
            else
            {
                parent->right = NULL;
                avl_remove_balance(parent, -1);
            }
        }
        else
        {
            if (!parent->parent)
            {
                assert(parent->right == entry);
                parent->right = right;
                right->parent = parent;
            }
            else if (parent->left == entry)
            {
                parent->left = right;
                right->parent = parent;
            }
            else
            {
                parent->right = right;
                right->parent = parent;
            }

            avl_remove_balance(right, 0);
        }
    }
    else if (!right)
    {
        if (!parent->parent)
        {
            assert(parent->right == entry);
            parent->right = left,
            left->parent = parent;
        }
        else if (parent->left == entry)
        {
            parent->left = left;
            left->parent = parent;
        }
        else
        {
            parent->right = left;
            left->parent = parent;
        }

        avl_remove_balance(left, 0);
    }
    else if (!right->left)
    {
        right->parent   = parent;
        right->left     = left;
        right->balance  = entry->balance;

        if (left)
            left->parent = right;

        if (parent->left == entry)
            parent->left = right;
        else
            parent->right = right;

        avl_remove_balance(right, -1);
    }
    else
    {
        struct avl_entry *successor_parent;
        struct avl_entry *successor_right;
        struct avl_entry *successor;

        successor = right;
        while (successor->left)
            successor = successor->left;

        successor_parent = successor->parent;
        successor_right  = successor->right;

        if (successor_parent->left == successor)
            successor_parent->left = successor_right;
        else
            successor_parent->right = successor_right;

        if (successor_right)
            successor_right->parent = successor_parent;

        successor->parent   = parent;
        successor->left     = left;
        successor->right    = right;
        successor->balance  = entry->balance;
        right->parent       = successor;

        if (left)
            left->parent = successor;

        if (parent->left == entry)
            parent->left = successor;
        else
            parent->right = successor;

        avl_remove_balance(successor_parent, 1);
    }

    entry->parent = NULL;
}

#endif  /* _TREE_H_ */
