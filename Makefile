SUBDIRS = $(dir $(wildcard */Makefile))


.PHONY: all
all: $(SUBDIRS)

.PHONY: test
test:
	$(MAKE) -C "src" test

.PHONY: clean
clean:
	for dir in $(SUBDIRS); do \
		$(MAKE) -C "$$dir" clean; \
	done

datasets: src

$(SUBDIRS): FORCE
	$(MAKE) -C "$@"

FORCE:
