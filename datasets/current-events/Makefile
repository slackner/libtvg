.PHONY: all
all: events.json

current-events-20191011.zip:
	# FIXME: Verify that sha1sum is 83ee8f40f2bc904915e03c631d8e97673eeacce6.
	wget -O current-events-20191011.zip "https://heibox.uni-heidelberg.de/f/4f8f07a6d1674cfdada3/?raw=1"

events.json: current-events-20191011.zip
	unzip -o "current-events-20191011.zip"
	@touch events.json
