import EGEL
model = EGEL.Model(embedding_size=50, learning_rate=0.5, batch_size=1024, scaling_factor=0.01, cat_weight=1)

#scaling_factor between 0-1 when scaling_factor=0 the model use the Numerical Features only and scaling_factor=1 the model use the Textual Features only

#input data files
Txt_file = 'textual_Features.txt' #the textual data file must be in the form of  “region_id  term_id  value”
NF_file = 'Numerical_Features.txt'#the numerical data file must be in the form of  “region_id  feature_id  value”
Cat_file = 'Categorical_Features.txt'#the categorical data file must be in the form of “region_id  feature_id ”

    
region_len=200000 #number of regions or entities
NF_len=4 #number of numerical features
cat_len=180 #number of categories
vocab_len=100000 #number of terms in the corpus

model.fit_to_corpus(region_len,NF_len+cat_len+vocab_len) #context length = NF_len+cat_len+vocab_len
num_epochs=3
model.train(num_epochs,Txt_file,NF_file,Cat_file)


