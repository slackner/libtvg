/*
 * Time-varying graph library
 * Synchronization functions.
 *
 * Copyright (c) 2018-2019 Sebastian Lackner
 */

#include <time.h>
#include <unistd.h>
#include <errno.h>
#include <limits.h>
#include <sys/syscall.h>

#include "internal.h"

#ifdef __linux__

static int wait_op = 128; /* FUTEX_WAIT | FUTEX_PRIVATE_FLAG */
static int wake_op = 129; /* FUTEX_WAKE | FUTEX_PRIVATE_FLAG */

static inline int futex_wait(int *addr, int val, struct timespec *timeout)
{
    return syscall(__NR_futex, addr, wait_op, val, timeout, 0, 0);
}

static inline int futex_wake(int *addr, int val)
{
    return syscall(__NR_futex, addr, wake_op, val, NULL, 0, 0);
}

void init_futex(void)
{
    int val = 0;

    futex_wait(&val, 10, NULL);

    if (errno == ENOSYS)
    {
        wait_op = 0; /* FUTEX_WAIT */
        wake_op = 1; /* FUTEX_WAKE */
        futex_wait(&val, 10, NULL);
    }

    if (errno == ENOSYS)
    {
        fprintf(stderr, "%s: Futexes not supported\n", __func__);
        assert(0);
    }
}

void event_init(struct event *event)
{
    event->value = 0;
}

void event_signal(struct event *event)
{
    __sync_fetch_and_or(&event->value, 1);
    futex_wake(&event->value, INT_MAX);
}

int event_wait(struct event *event, uint64_t timeout_ms)
{
    uint64_t timeout = clock_monotonic() + timeout_ms;
    struct timespec timespec;
    uint64_t now;

    while (!__sync_fetch_and_or(&event->value, 0))
    {
        timespec.tv_sec  = timeout_ms / 1000;
        timespec.tv_nsec = (timeout_ms % 1000) * 1000000;
        futex_wait(&event->value, 0, timeout_ms ? &timespec : NULL);
        if (!timeout_ms) continue;
        now = clock_monotonic();
        if ((int64_t)(now - timeout) >= 0) return 0;
        timeout_ms = timeout - now;
    }

    return 1;
}

void mutex_init(struct mutex *mutex)
{
    mutex->value = 0;
}

int mutex_trylock(struct mutex *mutex, uint64_t timeout_ms)
{
    uint64_t timeout = clock_monotonic() + timeout_ms;
    struct timespec timespec;
    uint64_t now;
    int val;

    while ((val = __sync_lock_test_and_set(&mutex->value, 1)))
    {
        timespec.tv_sec  = timeout_ms / 1000;
        timespec.tv_nsec = (timeout_ms % 1000) * 1000000;
        futex_wait(&mutex->value, val, timeout_ms ? &timespec : NULL);
        if (!timeout_ms) continue;
        now = clock_monotonic();
        if ((int64_t)(now - timeout) >= 0) return 0;
        timeout_ms = timeout - now;
    }

    return 1;
}

void mutex_lock(struct mutex *mutex)
{
    int res = mutex_trylock(mutex,0);
    assert(res);
}

void mutex_unlock(struct mutex *mutex)
{
    __sync_fetch_and_and(&mutex->value, 0);
    futex_wake(&mutex->value, 1);
}

#else   /* __linux__ */

void init_futex(void)
{
    /* Too bad, no futex support available :(. Instead of giving up,
     * let's try to get things working as best as possible. */
}

void event_init(struct event *event)
{
    event->value = 0;
}

void event_signal(struct event *event)
{
    __sync_fetch_and_or(&event->value, 1);
}

int event_wait(struct event *event, uint64_t timeout_ms)
{
    uint64_t timeout = clock_monotonic() + timeout_ms;
    struct timespec timespec;
    uint64_t now;

    timespec.tv_sec  = 0;
    timespec.tv_nsec = 1000000;

    while (!__sync_fetch_and_or(&event->value, 0))
    {
        nanosleep(&timespec, NULL);
        if (!timeout_ms) continue;
        now = clock_monotonic();
        if ((int64_t)(now - timeout) >= 0) return 0;
        timespec.tv_nsec = MIN(timespec.tv_nsec * 2, 50000000);
    }

    return 1;
}

void mutex_init(struct mutex *mutex)
{
    mutex->value = 0;
}

int mutex_trylock(struct mutex *mutex, uint64_t timeout_ms)
{
    uint64_t timeout = clock_monotonic() + timeout_ms;
    struct timespec timespec;
    uint64_t now;
    int val;

    timespec.tv_sec  = 0;
    timespec.tv_nsec = 1000000;

    while ((val = __sync_lock_test_and_set(&mutex->value, 1)))
    {
        nanosleep(&timespec, NULL);
        if (!timeout_ms) continue;
        now = clock_monotonic();
        if ((int64_t)(now - timeout) >= 0) return 0;
        timespec.tv_nsec = MIN(timespec.tv_nsec * 2, 50000000);
    }

    return 1;
}

void mutex_lock(struct mutex *mutex)
{
    int res = mutex_trylock(mutex,0);
    assert(res);
}

void mutex_unlock(struct mutex *mutex)
{
    __sync_fetch_and_and(&mutex->value, 0);
}

#endif  /* __linux__ */

void rwlock_init(struct rwlock *rwlock)
{
    mutex_init(&rwlock->control);
    mutex_init(&rwlock->write);
    rwlock->readers = 0;
}

void rwlock_lock_w(struct rwlock *rwlock)
{
    mutex_lock(&rwlock->control);
    mutex_lock(&rwlock->write);
    mutex_unlock(&rwlock->control);
}

void rwlock_unlock_w(struct rwlock *rwlock)
{
    assert(!rwlock->readers);
    mutex_unlock(&rwlock->write);
}

void rwlock_lock_r(struct rwlock *rwlock)
{
    mutex_lock(&rwlock->control);
    if (!__sync_fetch_and_add(&rwlock->readers, 1))
        mutex_lock(&rwlock->write);
    mutex_unlock(&rwlock->control);
}

void rwlock_unlock_r(struct rwlock *rwlock)
{
    if (!__sync_sub_and_fetch(&rwlock->readers, 1))
        mutex_unlock(&rwlock->write);
}
