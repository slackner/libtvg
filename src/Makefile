LDFLAGS  = -Wl,--no-undefined -Wl,-rpath,'$$ORIGIN/lib' -L./lib -I./lib -ltvg
SOURCES  = $(wildcard *.c)
PROGRAMS = $(patsubst %.c,%,$(SOURCES))

.PHONY: all
all: libtvg/libtvg.so libtvg/test $(PROGRAMS)

libtvg/libtvg.so: FORCE
	$(MAKE) -C libtvg libtvg.so

libtvg/test: FORCE
	$(MAKE) -C libtvg test

%: %.c libtvg/tvg.h libtvg/libtvg.so
	gcc -O2 -g -Werror -Wall -fstack-protector-strong -o $@ $< $(LDFLAGS)

FORCE:

.PHONY: test
test: libtvg/test

.PHONY: clean
clean:
	$(MAKE) -C libtvg clean
	rm -f $(PROGRAMS)
