.PHONY: all
all: result-norm.graph result.labels

networkTopics_data.zip:
	wget "https://dbs.ifi.uni-heidelberg.de/download/networkTopics_data.zip"

news_edgelist.tsv: networkTopics_data.zip
	unzip "networkTopics_data.zip"
	@touch news_edgelist.tsv

result-norm.graph: news_edgelist.tsv convert_graph.py
	./convert_graph.py

result.labels: news_edgelist.tsv convert_labels.py
	./convert_labels.py
