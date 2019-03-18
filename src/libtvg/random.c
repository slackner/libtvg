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
        int res = syscall(SYS_getrandom, buffer, length, 0);
        if (res < 0)
        {
            assert(errno == EINTR);
            continue;
        }

        assert(res <= length);
        buffer += res;
        length -= res;
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
        int res = read(dev_urandom, buffer, length);
        if (res < 0)
        {
            assert(errno == EINTR);
            continue;
        }

        assert(res <= length);
        buffer += res;
        length -= res;
    }
}

#endif  /* __linux__ */

float random_float(void)
{
    uint32_t value;
    random_bytes((uint8_t *)&value, sizeof(value));
    return (float)value / (float)(1ULL << (8 * sizeof(value)));
}
