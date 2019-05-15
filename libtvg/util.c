/*
 * Time-varying graph library
 * Utility functions.
 *
 * Copyright (c) 2017-2018 Sebastian Lackner
 */

#include <time.h>
#include <stdarg.h>

#include "internal.h"

int init_libtvg(uint64_t api_version)
{
    init_futex();

    return (api_version == LIBTVG_API_VERSION);
}

void progress(const char *format, ...)
{
    static int last_len;
    int i, new_len;
    va_list args;

    va_start(args, format);
    new_len = vfprintf(stderr, format, args);
    for (i = new_len; i < last_len; i++) fputc(' ', stderr);
    fputc('\r', stderr);
    va_end(args);

    last_len = new_len;
}

uint64_t clock_monotonic(void)
{
    struct timespec monotime;
    clock_gettime(CLOCK_MONOTONIC, &monotime);
    return monotime.tv_sec * (uint64_t)1000 + monotime.tv_nsec / 1000000;
}

uint64_t count_lines(FILE *fp)
{
    char buf[0x4000];
    long int offset;
    uint64_t count = 0;
    size_t i, read;
    int newline = 1;
    int ret;

    offset = ftell(fp);
    assert(offset != -1L);

    while ((read = fread(buf, 1, sizeof(buf), fp)) > 0)
    {
        /* If we are just at the beginning of a newline (i.e., if newline == 1),
         * then nonzero input means we have at least one additional line. If the
         * block ends with a '\n', we don't know yet if more content will follow. */
        if (newline) count++;
        newline = (buf[--read] == '\n');

        for (i = 0; i < read; i++)
            if (buf[i] == '\n') count++;
    }

    ret = fseek(fp, offset, SEEK_SET);
    assert(!ret);
    return count;
}
