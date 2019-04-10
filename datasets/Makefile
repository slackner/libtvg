SUBDIRS = $(dir $(wildcard */Makefile))

.PHONY: all
all: $(SUBDIRS)

.PHONY: clean
clean:

$(SUBDIRS): FORCE
	$(MAKE) -C "$@"

FORCE:
