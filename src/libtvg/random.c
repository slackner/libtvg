/*
 * Time-varying graph library
 * Random functions
 *
 * Copyright (c) 2018 Sebastian Lackner
 */

#define _GNU_SOURCE

#include <unistd.h>
#include <errno.h>
#include <sys/syscall.h>

#include "internal.h"

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

float random_float(void)
{
    uint32_t value;
    random_bytes((uint8_t *)&value, sizeof(value));
    return (float)value / (float)(1ULL << (8 * sizeof(value)));
}
