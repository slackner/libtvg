.PHONY: all
all: result-norm.graph

network-topics-20190625.zip:
	wget -O network-topics-20190625.zip "https://www.dropbox.com/s/6h1dq2msxv20msz/network-topics-20190625.zip?dl=1"
	sha1sum -c network-topics-20190625.zip.sha1

result-norm.graph: network-topics-20190625.zip
	unzip -o "network-topics-20190625.zip"
	@touch result-norm.graph

networkTopics_data.zip:
	wget -O networkTopics_data.zip "https://dbs.ifi.uni-heidelberg.de/download/networkTopics_data.zip"
	sha1sum -c networkTopics_data.zip.sha1

news_edgelist.tsv: networkTopics_data.zip
	unzip -o "networkTopics_data.zip"
	@touch news_edgelist.tsv

.PHONY: generate
generate: news_edgelist.tsv convert_nodes.py convert_graph.py
	./convert_nodes.py
	./convert_graph.py
