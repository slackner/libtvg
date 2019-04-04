## Time Varying Graph Explorer

Installation
============

As a first step, please make sure that all required dependencies are installed.
The following commands can be used to install missing build, test and runtime
dependencies on Ubuntu or Debian operating systems:

```bash
# Build dependencies:
sudo apt-get install build-essential
sudo apt-get install wget
sudo apt-get install libmongoc-dev

# Test dependencies:
sudo apt-get install python3-mockupdb
sudo apt-get install clang llvm lcov

# Runtime dependencies:
sudo apt-get install python3
sudo apt-get install python3-numpy
pip3 install SimpleWebSocketServer
```

Afterwards, just clone this repository and run (in the root directory of this
repository):

```bash
make
```

Note that this might take quite some time, since this command also downloads
and preprocesses additional datasets. If everything goes well, the program
terminates with exitcode 0. All compiled programs will be located within the
src/ directory.

Self-Test
=========

This program comes with an extensive set of self-tests to ensure everything
works as expected and to detect regressions during the development process.
To run the tests, just execute:

```bash
make test
```

To run tests for the Python 3 bindings, execute:

```bash
./src/libtvg/pytvg.py
```

Time Varying Graph File Format
==============================

Most programs contained in this package require graph data with the following
input format:

```
# Lines starting with '#' are ignored
StartNode EndNode Weight Timestamp  \
StartNode EndNode Weight Timestamp   } Edges
StartNode EndNode Weight Timestamp  /
[...]
```

Lines starting with "#" are comments and have to be ignored. All remaining
lines describe edges. Each edge is defined by a start node ID, an end node ID,
a weight, and a timestamp. The node IDs can be arbitrary integers between 0 and
2^64 - 1. An example graph can be seen in src/data/example-tvg.graph.

Usage Example
=============

To start a server with the example dataset, run the following command:

```bash
src/explorer/server.py --labels=src/data/example-tvg.labels \
                       src/data/example-tvg.graph
```

To run the server with the full dataset, run:

```bash
src/explorer/server.py --labels=datasets/network-topics/result.labels \
                       datasets/network-topics/result-sum.graph
```

It is also possible to provide the URI of a MongoDB server, e.g.:

```bash
src/explorer/server.py mongodb://thabit:27021
```

In each case, open src/data/explorer/www/index.html in a web-browser of your
choice to interact with the server.
