.PHONY: all
all: result-norm.graph result.nodes

networkTopics_data.zip:
	wget "https://dbs.ifi.uni-heidelberg.de/download/networkTopics_data.zip"

news_edgelist.tsv: networkTopics_data.zip
	unzip "networkTopics_data.zip"
	@touch news_edgelist.tsv

result-norm.graph: news_edgelist.tsv convert_graph.py
	./convert_graph.py

result.nodes: news_edgelist.tsv convert_nodes.py
	./convert_nodes.py
