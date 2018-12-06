from __future__ import division
from random import shuffle
import tensorflow as tf
import numpy as np
import sys
import psutil
import gc
import datetime



class NotTrainedError(Exception):
    pass

class NotFitToCorpusError(Exception):
    pass

class Model():
    def __init__(self, embedding_size, scaling_factor=0.01, batch_size=512, learning_rate=0.05, cat_weight=1):
        self.embedding_size = embedding_size
        self.scaling_factor = scaling_factor
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.cat_weight = cat_weight
        self.NFscaling_facor=None
        self.__embeddings = None

    def fit_to_corpus(self, region_len,context_len):
        self.__fit_to_corpus(region_len,context_len)
        self.__build_graph()
    def __fit_to_corpus(self, region_len,context_len):         
        self.context_size=context_len
        self.region_size=region_len
        
        
    def __build_graph(self):
        self.__graph = tf.Graph()
        with self.__graph.as_default(), self.__graph.device("/cpu:0"):
            
            scaling_factor = tf.constant([self.scaling_factor], dtype=tf.float32,
                                         name="scaling_factor")
          
            cat_weight = tf.constant([self.cat_weight], dtype=tf.float32,
                                         name="cat_weight")
            
            self.__focal_input = tf.placeholder(tf.int32, shape=[self.batch_size],
                                                name="focal_words")
            
            self.__context_input = tf.placeholder(tf.int32, shape=[self.batch_size],
                                                  name="context_words")
            
            self.__cooccurrence_count = tf.placeholder(tf.float16, shape=[self.batch_size],
                                                       name="cooccurrence_count")

            self.__NF_input = tf.placeholder(tf.float32, shape=[self.batch_size],
                                                  name="numerical_features")
            
            self.__tags_input = tf.placeholder(tf.float32, shape=[self.batch_size],
                                                       name="tags")
            
            self.__cat_input = tf.placeholder(tf.float32, shape=[self.batch_size],
                                                       name="categories") 
            

            focal_embeddings = tf.Variable(
                tf.random_uniform([self.region_size, self.embedding_size], 1.0, -1.0),name="focal_embeddings")#embedding_size=dimentions
            
            context_embeddings = tf.Variable(
                tf.random_uniform([self.context_size, self.embedding_size], 1.0, -1.0), name="context_embeddings")

            context_biases = tf.Variable(tf.random_uniform([self.context_size], 1.0, -1.0),
                                         name="context_biases")

            focal_embedding = tf.nn.embedding_lookup([focal_embeddings], self.__focal_input)
            context_embedding = tf.nn.embedding_lookup([context_embeddings], self.__context_input)
            context_bias = tf.nn.embedding_lookup([context_biases], self.__context_input)
            NFscaling_factor = 1-scaling_factor
            self.__nfsc=NFscaling_factor
            
            ###### Textual features part
            embedding_product = tf.reduce_sum(tf.multiply(focal_embedding, context_embedding),1)
            self.__emp=embedding_product
            print('embedding_product.shape',embedding_product.shape)      
            distance_expr = tf.square(tf.add_n([
                embedding_product,
                context_bias,
                tf.negative(tf.to_float(self.__cooccurrence_count))]))
            print('distance_expr.shape=',distance_expr.shape)
            self.__disexp=distance_expr
            tags_single_losses = tf.multiply(self.__tags_input, distance_expr)#mask
            self.__tsl=tags_single_losses
            
            ###### Numerical features part
            NF_single_losses = tf.multiply(self.__NF_input, distance_expr)#mask
            self.__nfsl=NF_single_losses
            print('NF_single_losses.shape=',NF_single_losses.shape)
            self.__tags_loss = tf.multiply(tf.reduce_sum(tags_single_losses),scaling_factor)
            print('tags loss shape',self.__tags_loss.shape)           
            self.__NF_loss = tf.multiply(tf.reduce_sum(NF_single_losses),NFscaling_factor)  
            print('NF loss shape',self.__NF_loss.shape)
            
            ###### Categorical features part          
            catembedding_product = tf.square(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(focal_embedding, context_embedding)),1))) 
            cat_single_losses = tf.multiply(self.__cat_input, catembedding_product)#mask
            self.__csl=cat_single_losses
            print('cat_single_losses.shape=',cat_single_losses.shape)
            self.__cat_loss = tf.multiply(tf.reduce_sum(cat_single_losses),cat_weight)
            print('cat loss shape',self.__cat_loss.shape)
            
            self.__total_loss = tf.add_n([self.__tags_loss, self.__NF_loss, self.__cat_loss])
            print('self.__total_loss shape',self.__total_loss.shape)
            
            self.__optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(
                self.__total_loss)##########NFoptimizer
            self.__combined_embeddings2 = (focal_embeddings)


    def train(self, num_epochs, corpus, NF, cat,log_dir=None, summary_batch_interval=1000,
              tsne_epoch_interval=None):
        should_write_summaries = log_dir is not None and summary_batch_interval
        should_generate_tsne = log_dir is not None and tsne_epoch_interval
        print('loading data......................')
        cooccurrences =[]
        f_NF=open(NF,'r')        
        cooccurrences = cooccurrences +[(int(l.split()[0]),int(l.split()[1]),float(l.split()[2]),0,1,0)#add mask
                           for l in f_NF]
        del(f_NF)
        print('length after the NF',len(cooccurrences))

        f_cat=open(cat,'r')
        cooccurrences = cooccurrences +[(int(l.split()[0]),int(l.split()[1])+10,int(l.split()[1]),0,0,1)#add mask
                   for l in f_cat]
        del(f_cat)
        print('length after the cat',len(cooccurrences))
        f_words=open(corpus,'r')
        cooccurrences = cooccurrences +[(int(l.split()[0]),int(l.split()[1]),float(l.split()[2]),1,0,0)#add mask
                         for l in f_words]

        del(f_words)
        print('length after the tags',len(cooccurrences))
      
	
        gc.collect()
        with tf.Session(graph=self.__graph) as session:#####start the main session
            print('start initializing global variables..........')
            tf.global_variables_initializer().run()###initializing the variables
            
            for epoch in range(num_epochs):#iterations
                print('\n memory information in the begining of itr '+str(epoch)+':\n',psutil.virtual_memory())
                #f1.write('\nitr'+str(epoch)+'\n')
                batches = self.__prepare_batches(cooccurrences)######divided data into batches
                shuffle(batches)
                for batch_index, batch in enumerate(batches):
                    i_s, j_s, counts,i1 ,i2, i3 = batch
                    if len(counts) != self.batch_size:
                        continue
                    feed_dict = {               #fill the holder with batch data
                        self.__focal_input: i_s,#location id
                        self.__context_input: j_s,#context id
                        self.__cooccurrence_count: counts,#cooccur
                        self.__tags_input:i1,#tag input mask
                        self.__NF_input:i2,#NF input mask 
                        self.__cat_input:i3}#cat input mask
                    __,tloss,tagsl,nfl,catl,csl,NFS,emp,dis,tsl,nfsl=session.run([self.__optimizer,self.__total_loss,self.__tags_loss,self.__NF_loss,self.__cat_loss,self.__csl,self.__nfsc,self.__emp,self.__disexp,self.__tsl,self.__nfsl], feed_dict=feed_dict)##########run optimization

                    print('NFscaling_factor',NFS)
                    print('tag total loss=',tagsl)
                    print('NF total loss=',nfl)
                    print('cat total loss=',catl)                    
                    print('total loss=',tloss)
                    
                
            self.__embeddings = self.__combined_embeddings2.eval()
            print('memory after the embedding:',psutil.virtual_memory())
            print('priniting on a file.........')
            np.savetxt('EGEL.txt',self.__embeddings,delimiter=' ',newline='\n',fmt='%.7f')
            current_time = datetime.datetime.now()
            print("Printed!!!!!!!!!!!!!!!! at {:%H:%M}".format(current_time)) 
        
                        
    def __prepare_batches(self,cooccurrences):
        shuffle(cooccurrences)
        i_indices, j_indices, counts, i1, i2, i3= zip(*cooccurrences)
        return list(_batchify(self.batch_size, i_indices, j_indices, counts,i1 ,i2,i3))
    
def _batchify(batch_size, *sequences):
    for i in range(0, len(sequences[0]), batch_size):
        yield tuple(sequence[i:i+batch_size] for sequence in sequences)
    del(sequences)
    gc.collect()
        

def _device_for_node(n):    
    return "/cpu:0"

        



