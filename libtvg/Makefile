CC          = gcc
LIB_CFLAGS  = -O2 -g -Werror -Wall -fstack-protector-strong -fPIC
LIB_LDFLAGS = -shared -lm
APP_CFLAGS  = -O2 -g -Werror -Wall -fstack-protector-strong
APP_LDFLAGS = -L. -I. -ltvg -lm

UNAME_S = $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	LIBNAME      = libtvg.dylib
	LIB_LDFLAGS += -install_name @rpath/$(LIBNAME)
	APP_LDFLAGS += -Wl,-rpath,@executable_path
else
	LIBNAME      = libtvg.so
	LIB_LDFLAGS += -Wl,-soname,$(LIBNAME) -Wl,--no-undefined
	APP_LDFLAGS += -Wl,-rpath,'$$ORIGIN' -Wl,--no-undefined
ifndef NOMONGODB
	LIB_CFLAGS  += -I/usr/include/libbson-1.0 \
	               -I/usr/include/libmongoc-1.0 \
	               -DHAVE_LIBMONGOC
	LIB_LDFLAGS += -lmongoc-1.0 -lbson-1.0
endif
ifndef NOVALGRIND
	LIB_CFLAGS  += -DHAVE_VALGRIND
endif
endif

.PHONY: all
all: $(LIBNAME)

.PHONY: test
test: tests
	./tests

.PHONY: clean
clean:
	rm -f $(LIBNAME) tests *.gcda *.gcno default.profraw coverage.info

$(LIBNAME): $(wildcard *.c *.h)
	$(CC) $(LIB_CFLAGS) $(CFLAGS) -o $(LIBNAME) $(wildcard *.c) $(LIB_LDFLAGS) $(LDFLAGS)

tests: tests.c minheap.c ranges.c tvg.h internal.h $(LIBNAME)
	$(CC) $(APP_CFLAGS) $(CFLAGS) -o $@ tests.c minheap.c ranges.c $(APP_LDFLAGS) $(LDFLAGS)
