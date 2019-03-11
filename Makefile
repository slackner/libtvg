SUBDIRS = $(dir $(wildcard */Makefile))


.PHONY: all
all: $(SUBDIRS)

datasets: src

.PHONY: $(SUBDIRS)
$(SUBDIRS):
	$(MAKE) -C "$@"

.PHONY: test
test:
	$(MAKE) -C "src" test

.PHONY: clean
clean:
	for dir in $(SUBDIRS); do \
		$(MAKE) -C "$$dir" clean; \
	done
