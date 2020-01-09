/*
 * Time-varying graph library
 * Random functions
 *
 * Copyright (c) 2018 Sebastian Lackner
 */

#define _GNU_SOURCE

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/syscall.h>

#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

#include "internal.h"

#ifdef __linux__

void random_bytes(uint8_t *buffer, size_t length)
{
    while (length)
    {
        ssize_t res = syscall(SYS_getrandom, buffer, length, 0);
        if (res < 0)
        {
            assert(errno == EINTR);
            continue;
        }

        assert(res <= length);
        buffer += res;
        length -= (size_t)res;
    }
}

#else   /* __linux__ */

void random_bytes(uint8_t *buffer, size_t length)
{
    static int dev_urandom = -1;

    if (dev_urandom < 0 && (dev_urandom = open("/dev/urandom", O_RDONLY)) < 0)
    {
        fprintf(stderr, "%s: Failed to open /dev/urandom\n", __func__);
        assert(0);
        return;
    }

    while (length)
    {
        ssize_t res = read(dev_urandom, buffer, length);
        if (res < 0)
        {
            assert(errno == EINTR);
            continue;
        }

        assert(res <= length);
        buffer += res;
        length -= (size_t)res;
    }
}

#endif  /* __linux__ */

float random_float(void)
{
    uint32_t value;
    random_bytes((uint8_t *)&value, sizeof(value));
    return (float)value / (float)(1ULL << (8 * sizeof(value)));
}

void random_pool_init(struct random_pool *pool)
{
    pool->pos = sizeof(pool->buffer);
}

void random_pool_bytes(struct random_pool *pool, uint8_t *buffer, size_t length)
{
    size_t remaining;

    /* For sufficiently large requests, directly forward to kernel. */
    if (length >= sizeof(pool->buffer) / 8)
        return random_bytes(buffer, length);

    assert(pool->pos <= sizeof(pool->buffer));

    /* Try to fulfill request by doing partial copy from the existing
     * buffer. Most of the time this will be sufficient. */
    remaining = MIN(sizeof(pool->buffer) - pool->pos, length);
    memcpy(buffer, &pool->buffer[pool->pos], remaining);
    pool->pos += remaining;

    /* If this did not work (or was not sufficient) refill the buffer,
     * and copy over a part of it. */
    length -= remaining;
    if (length > 0)
    {
        buffer += remaining;
        random_bytes(pool->buffer, sizeof(pool->buffer));
        memcpy(buffer, &pool->buffer[0], length);
        pool->pos = length;
    }
}

float random_pool_float(struct random_pool *pool)
{
    uint32_t value;
    random_pool_bytes(pool, (uint8_t *)&value, sizeof(value));
    return (float)value / (float)(1ULL << (8 * sizeof(value)));
}
