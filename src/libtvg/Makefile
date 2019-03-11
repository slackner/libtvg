LDFLAGS  = -Wl,--no-undefined -Wl,-rpath,'$$ORIGIN' -L. -I. -ltvg

.PHONY: all
all: libtvg.so

libtvg.so: $(wildcard *.c *.h)
	gcc -O2 -g -Werror -Wall -fstack-protector-strong -fPIC \
		-shared -Wl,-soname,libtvg.so -Wl,--no-undefined \
		-o libtvg.so $^ -lm

tests: tests.c tvg.h internal.h libtvg.so
	gcc -O2 -g -Werror -Wall -fstack-protector-strong -o $@ $< $(LDFLAGS) -lm

.PHONY: test
test: tests
	./tests

.PHONY: clean
clean:
	rm -f libtvg.so tests
