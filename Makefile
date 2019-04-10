SUBDIRS = $(dir $(wildcard */Makefile))

.PHONY: all
all: $(SUBDIRS)

.PHONY: test
test: FORCE
	$(MAKE) -C libtvg test

.PHONY: clean
clean:
	for dir in $(SUBDIRS); do \
		$(MAKE) -C "$$dir" clean; \
	done

datasets: libtvg explorer
explorer: libtvg

$(SUBDIRS): FORCE
	$(MAKE) -C "$@"

FORCE:
