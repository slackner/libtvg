LIBNAME     = libtvg.so
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
	LIB_LDFLAGS += -Wl,-soname,$(LIBNAME) -Wl,--no-undefined
	APP_LDFLAGS += -Wl,-rpath,'$$ORIGIN' -Wl,--no-undefined

	# FIXME: For now, only compile MongoDB support on Linux.
	LIB_CFLAGS  += -I/usr/include/libbson-1.0 \
	               -I/usr/include/libmongoc-1.0 \
	               -DHAVE_LIBMONGOC
	LIB_LDFLAGS += -lmongoc-1.0 -lbson-1.0
endif

.PHONY: all
all: $(LIBNAME)

$(LIBNAME): $(wildcard *.c *.h)
	gcc $(LIB_CFLAGS) -o $(LIBNAME) $(wildcard *.c) $(LIB_LDFLAGS)

tests: tests.c tvg.h internal.h $(LIBNAME)
	gcc $(APP_CFLAGS) -o $@ $< $(APP_LDFLAGS)

.PHONY: test
test: tests
	./tests

.PHONY: clean
clean:
	rm -f $(LIBNAME) tests
