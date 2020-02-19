.PHONY: all
all: events.json

current-events-20191011.zip:
	wget -O current-events-20191011.zip "https://www.dropbox.com/s/ap2gfdg4ztqck5d/current-events-20191011.zip?dl=1"
	sha1sum -c current-events-20191011.zip.sha1

events.json: current-events-20191011.zip
	unzip -o "current-events-20191011.zip"
	@touch events.json
