LIBNAME = libtvg.so
LDFLAGS = -L./lib -I./lib -ltvg

UNAME_S = $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	LIBNAME  = libtvg.dylib
	LDFLAGS += -Wl,-rpath,@executable_path/lib
else
	LDFLAGS += -Wl,-rpath,'$$ORIGIN/lib' -Wl,--no-undefined
endif

SOURCES  = $(wildcard *.c)
PROGRAMS = $(patsubst %.c,%,$(SOURCES))

.PHONY: all
all: libtvg/$(LIBNAME) libtvg/test $(PROGRAMS)

libtvg/$(LIBNAME): FORCE
	$(MAKE) -C libtvg $(LIBNAME)

libtvg/test: FORCE
	$(MAKE) -C libtvg test

%: %.c libtvg/tvg.h libtvg/$(LIBNAME)
	gcc -O2 -g -Werror -Wall -fstack-protector-strong -o $@ $< $(LDFLAGS)

FORCE:

.PHONY: test
test: libtvg/test

.PHONY: clean
clean:
	$(MAKE) -C libtvg clean
	rm -f $(PROGRAMS)
