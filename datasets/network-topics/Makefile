.PHONY: all
all: result-norm.graph

network-topics-20190625.zip:
	# FIXME: Verify that sha1sum is 2c34a24897200d2123d570444ce58181f1bef5d9.
	wget -O network-topics-20190625.zip "https://heibox.uni-heidelberg.de/f/11d550e95c/?raw=1"

result-norm.graph: network-topics-20190625.zip
	unzip -o "network-topics-20190625.zip"
	@touch result-norm.graph

networkTopics_data.zip:
	# FIXME: Verify that sha1sum is 6b4a4924155b30d0202ae9c2b6537edd806914bb.
	wget "https://dbs.ifi.uni-heidelberg.de/download/networkTopics_data.zip"

news_edgelist.tsv: networkTopics_data.zip
	unzip -o "networkTopics_data.zip"
	@touch news_edgelist.tsv

.PHONY: generate
generate: news_edgelist.tsv convert_nodes.py convert_graph.py
	./convert_nodes.py
	./convert_graph.py
