SUBDIRS = $(dir $(wildcard */Makefile))

.PHONY: all
all: $(SUBDIRS)

.PHONY: $(SUBDIRS)
$(SUBDIRS):
	$(MAKE) -C "$@"
