UNAME_S = $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	LIBNAME = libtvg.dylib
else
	LIBNAME = libtvg.so
endif

.PHONY: all
all: libtvg/$(LIBNAME)

.PHONY: test
test: libtvg/test

.PHONY: clean
clean:
	$(MAKE) -C libtvg clean

libtvg/$(LIBNAME): FORCE
	$(MAKE) -C libtvg $(LIBNAME)

libtvg/test: FORCE
	$(MAKE) -C libtvg test

FORCE:
