SUBDIRS = $(dir $(wildcard */Makefile))

.PHONY: all
all: $(SUBDIRS)

$(SUBDIRS): FORCE
	$(MAKE) -C "$@"

FORCE:
