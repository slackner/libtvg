Time Varying Graph Explorer
===========================

## Installation

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

On macOS, the following commands can be used to install required dependencies:

```bash
# Build dependencies:
xcode-select --install
brew install wget
brew install mongo-c-driver

# Test dependencies:
pip3 install mockupdb

# Runtime dependencies:
brew install python3
pip3 install numpy
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

## Self-Test

This program comes with an extensive set of self-tests to ensure everything
works as expected and to detect regressions during the development process.
To run the tests, just execute:

```bash
make test
```

To run tests for the Python 3 bindings, execute:

```bash
./libtvg/pytvg.py
```

## Documentation

Documentation can be found in the `docs/` directory.

* `docs/pytvg.md` describes how to use API functions provided by `pytvg`.
* `docs/load-tvg.md` describes different ways to load time varying graphs.

## Usage Example

To start a server with the example dataset, run the following command:

```bash
cd explorer
./server.py news_example.conf
```

To run the server with the full dataset, run:

```bash
cd explorer
./server.py news_network_topics.conf
```

It is also possible to stream datasets directly from a MongoDB server, e.g.:

```bash
cd explorer
./server.py news_ambiverse_thabit.conf
```

In each case, open explorer/html/index.html in a web-browser of your
choice to interact with the server.
