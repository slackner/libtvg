.PHONY: all
all: dtm/dtm/main

.PHONY: clean
clean:

dtm/dtm/main: dtm FORCE
	$(MAKE) -C dtm/dtm

dtm:
	git clone https://github.com/blei-lab/dtm.git

FORCE:
