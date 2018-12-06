# EGEL-Model

This project provides code of EGEL (Embedding Geographic Locations), a model for learning vector space embeddings of geographic locations, using Flickr tags and structured information extracted from various scientific resources. The structured information can be of two forms numerical and categorical information. 

The model is general and can be used to integrate any textual, numerical and categorical features into one low dimensional vector space embedding regardless of the source of the data and evaluation task. This implementation using Python-TensorFlow, a "library for numerical computation using data flow graphs" by Google. 

If you use this code in your research, please refer to this paper “Embedding Geographic Locations for Modelling the Natural Environment using Flickr Tags and Structured Data” by Shelan S. Jeawak, Christopher B. Jones, and Steven Schockaert (2018). 

# How to run this code?

import EGEL

model = EGEL.Model(embedding_size=50, learning_rate=0.5, batch_size=1024, scaling_factor=0.01, cat_weight=1)
#scaling_factor between 0-1 when scaling_factor=0 the model use the Numerical Features only and scaling_factor=1 the model use the Textual Features only

#input data files

Txt_file = 'textual_Features.txt' 
#the textual data file must be in the form of “region_id  term_id  value\n”

NF_file = 'Numerical_Features.txt'
#the numerical data file must be in the form of “region_id  feature_id  value\n”

Cat_file = 'Categorical_Features.txt'
#the categorical data file must be in the form of “region_id  feature_id \n”


#insert the number of features

region_len=200000 #number of regions or entities

NF_len=4 #number of numerical features

cat_len=180 #number of categories

vocab_len=100000 #number of terms in the corpus

model.fit_to_corpus(region_len,NF_len+cat_len+vocab_len) #context_len=NF_len+cat_len+vocab_len

num_epochs=30

model.train(num_epochs,Txt_file,NF_file,Cat_file)

#the embedding vectors will be saved in EGEL.txt file which contains vectors for all the regions
