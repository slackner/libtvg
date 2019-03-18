LIBNAME     = libtvg.so
LIB_LDFLAGS = -shared
APP_LDFLAGS = -L. -I. -ltvg

UNAME_S = $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	LIBNAME      = libtvg.dylib
	LIB_LDFLAGS += -install_name @rpath/$(LIBNAME)
	APP_LDFLAGS += -Wl,-rpath,@executable_path/
else
	LIB_LDFLAGS += -Wl,-soname,$(LIBNAME) -Wl,--no-undefined
	APP_LDFLAGS += -Wl,-rpath,'$$ORIGIN' -Wl,--no-undefined
endif

.PHONY: all
all: $(LIBNAME)

$(LIBNAME): $(wildcard *.c *.h)
	gcc -O2 -g -Werror -Wall -fstack-protector-strong -fPIC \
		$(LIB_LDFLAGS) -o $(LIBNAME) $(wildcard *.c) -lm

tests: tests.c tvg.h internal.h $(LIBNAME)
	gcc -O2 -g -Werror -Wall -fstack-protector-strong -o $@ $< $(APP_LDFLAGS) -lm

.PHONY: test
test: tests
	./tests

.PHONY: clean
clean:
	rm -f $(LIBNAME) tests
