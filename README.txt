The presentations and final report source is in the docs folder
The source code is in the src folder
	loader.py contains the code for organizing the data and for the automatic constraint generator
	gpt2.py contains the scripts for training the language model
	sc_gpt2_model.py contains the implementation of the TMD transformer networks
Evaluation logs are available in data/gpt2/(full|nouns|none)_eval_logs.txt
	This file shows example outputs from the language model
The upper ontology words are in data/manual_words.txt
The tags for those words are in data/tags_m.txt
The top words of the word clusters are in data/top_words_[nvd].txt
The tags for the word clusters are in data/tags_[nvd].txt
The most frequent 1000 English words are in data/common_english.txt
The manually written tags for those are in data/tags_common_[nvd].txt
