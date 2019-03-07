# EGEL-Model

This project provides the code of EGEL (Embedding Geographic Locations), a model for learning vector space embeddings of geographic locations, using Flickr tags and structured information extracted from various scientific resources. The structured information can be of two forms: numerical and categorical information. 

This implementation uses Python and TensorFlow (a "library for numerical computation using data flow graphs" by Google). You can use the sample of the data to run the code, however, the results reported in the paper are based on larger datasets. 

If you use this code in your research, please refer to this paper “Embedding Geographic Locations for Modelling the Natural Environment using Flickr Tags and Structured Data” by Shelan S. Jeawak, Christopher B. Jones, and Steven Schockaert, ECIR 2019. 

# How to run this code?
import EGEL

model = EGEL.Model(embedding_size=50, learning_rate=0.5, batch_size=1024, scaling_factor=0.01, cat_weight=1)

model.fit(region_len,NF_len+cat_len+vocab_len)

model.train(num_epochs=30,Txt_file,NF_file,Cat_file)


