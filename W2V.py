# -*- coding: utf-8 -*-

#import multiprocessing
#from gensim.models import Word2Vec
#from sklearn.manifold import TSNE
#import matplotlib.pyplot as plt
#from ggplot import *
#import pandas as pd

import os.path
import tensorflow as tf
import numpy as np
import math
import time
import pickle

########################################################################################################################
# Métodos
########################################################################################################################

def getSkipGramData( window=2):

    def load_pickle(name):
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)

    filename = "datos/Extend/ARRAY_ARRAYS_TRAIN"

    if os.path.isfile(filename + ".pkl"):
        DATA = load_pickle(filename)
    else:
        print("No existe el pickle...")
        exit()

    DATA = list(map(int, DATA))
    LENGTH = max(DATA) + 1

    RES_LEN = (len(DATA) - (window * 2)) * (window * 2)

    batch_inputs = np.ndarray(shape=(RES_LEN), dtype=np.int32)
    batch_context = np.ndarray(shape=(RES_LEN, 1), dtype=np.int32)

    j = 0

    for i in range(window, len(DATA) - window):
        for w in range(1, window + 1):
            batch_inputs[j] = DATA[i]
            batch_context[j, 0] = DATA[i - w]
            j += 1

            batch_inputs[j] = DATA[i]
            batch_context[j, 0] = DATA[i + w]
            j += 1

    return (batch_inputs, batch_context, LENGTH)

########################################################################################################################
# Llamadas
########################################################################################################################

batch_size = 512
embedding_size = 64  # Tamaño del embedding (capa oculta)
window = 2 # Tamaño de la ventana del skipgram
num_sampled = 64 # Ejemplos negativos (Para el NSE)
seed = 100
learning_rate = 0.1
epochs = 1000

# Carpeta donde se almacena el modelo de la red
#model_path = 'models/w2v'
model_path = 'models/w2v/TITANXP'
model_name = 'model'
complete_path = model_path+"/"+model_name
emb_path = 'embeddings/w2v'

# Obtener todos lo datos
print("Obteniendo datos...")
ALL_X, ALL_Y, vocabulary_size = getSkipGramData(window=window)
print("Datos obtenidos!")

#Mezclar datos
TOTAL_LENGTH = len(ALL_X)
indx = np.arange(TOTAL_LENGTH)
np.random.seed(seed)
np.random.shuffle(indx)

ALL_X=ALL_X[indx]
ALL_Y=ALL_Y[indx]

#Separar datos en TRAIN, DEV y TEST
TRAIN_LENGTH = int(TOTAL_LENGTH*0.98)
DEV_LENGTH= int(TOTAL_LENGTH*0.01)
TEST_LENGTH = int(TOTAL_LENGTH*0.01)
#Train
X= ALL_X[:TRAIN_LENGTH]
Y= ALL_Y[:TRAIN_LENGTH]
#Dev
DEV_X  = ALL_X[TRAIN_LENGTH:TRAIN_LENGTH+DEV_LENGTH]
DEV_Y  = ALL_Y[TRAIN_LENGTH:TRAIN_LENGTH+DEV_LENGTH]
#Test
TEST_X = ALL_X[TRAIN_LENGTH+DEV_LENGTH:]
TEST_Y = ALL_Y[TRAIN_LENGTH+DEV_LENGTH:]

#Creación del grafo de TF.
graph = tf.Graph()

with graph.as_default():

    #Número global de iteraciones
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    #Datos de entrada -----------------------------------------------------------------------------------------------------------------------------
    #Array del tamaño del batch con las X
    train_inputs = tf.placeholder(tf.int32, shape=[None], name="train_inputs")
    #Array del tamaño del batch con las Y asociadas
    train_labels = tf.placeholder(tf.int32, shape=[None, 1], name="train_labels")

    #Conexión primera y segunda capa --------------------------------------------------------------------------------------------------------------
    # Matriz de pesos entre la capa de entrada y la de embeddings
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name='embeddings')

    #embeddings = tf.get_variable("embeddings", shape=[vocabulary_size, embedding_size],initializer=tf.contrib.layers.xavier_initializer(seed=seed))

    # Como la relacion entre ambas es lineal, no hay bias ni función de activación
    # Por tanto se puede obtener el embedding directamente de la matriz anterior ( Embedding de la palabra 10 == fila 10 de la matriz de pesos)
    embed = tf.nn.embedding_lookup(embeddings, train_inputs, name="embed")

    #Conexión segunda y última capa ---------------------------------------------------------------------------------------------------------------
    #Pesos y biases
    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],stddev=1.0 / math.sqrt(embedding_size)), name="weights")
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]), name="biases")


    #Cálculo de LOSS y optimizador- ---------------------------------------------------------------------------------------------------------------
    nce_loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=train_labels,
                       inputs=embed,
                       num_sampled=num_sampled,
                       num_classes=vocabulary_size),name="nce_loss")

    #optimizer = tf.train.AdamOptimizer(learning_rate).minimize(nce_loss,global_step=global_step)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(nce_loss,global_step=global_step)

    init = tf.global_variables_initializer()

    #Crear objeto encargado de almacenar la red
    saver = tf.train.Saver(max_to_keep=1)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

with tf.Session(graph=graph) as session:

    # We must initialize all variables before we use them.
    init.run()
    print('Initialized')

    #Obtener el checkpoint
    ckpt = tf.train.get_checkpoint_state(model_path)

    #Si existe el checkpoint restaurar
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(session, ckpt.model_checkpoint_path)
        print('Modelo cargado...')
    else:
        print('No existe modelo...')

    ################################################################################
    #Obtener la matriz de embeddings
    print("Obteniendo embeddings...")

    E = session.run(embeddings)
    E.dump(emb_path+"/EMB_MATRIX")

    print("Embedings obtenidos!")
    ################################################################################

    exit()

    #Obtener el número de iteraciones necesario y los datos restantes
    ITRS= int(len(X)/batch_size)
    RES_ITMS = len(X) - (ITRS*batch_size)

    #Si no es división exacta, se añade una iteración
    if(RES_ITMS>0):ITRS+=1

    print("Iteraciones necesarias:"+str(ITRS))

    for e in range(epochs):
        print("Epoch "+str(e))

        tss = time.time()

        for step in range(ITRS):

            #Si se añadió una iteración, introducir los datos restantes
            if(step==ITRS-1 and RES_ITMS>0):
                batch_inputs = X[(ITRS-1)*batch_size:]
                batch_context = Y[(ITRS-1)*batch_size:]
            #Si no, es un batch normal
            else:
                st = step*batch_size
                nd = (step+1)*batch_size

                batch_inputs = X[st:nd]
                batch_context = Y[st:nd]

            feed_dict = {train_inputs: batch_inputs, train_labels: batch_context}

            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            _, loss_val, gs = session.run([optimizer, nce_loss,global_step], feed_dict=feed_dict)

            if (gs % 100 == 0 and gs>0):

                # The average loss is an estimate of the loss over the last 100 batches.
                print('\tLast batch loss at global-step ', gs, ': ', loss_val)

                # Now, save the graph
                #saver.save(session, complete_path, global_step=global_step)

        print(time.time()-tss)
        saver.save(session, complete_path, global_step=global_step)

        # Probar con DEV
        feed_dict = {train_inputs: DEV_X, train_labels: DEV_Y}
        loss_val = session.run([nce_loss], feed_dict=feed_dict)
        print('DEV batch loss at global-step ' + str(gs) + ': ' + str(loss_val))
        with open('train_results.txt', 'a') as the_file: the_file.write('DEV\t' + str(gs) + '\t' + str(loss_val) + '\t' + str(time.time() - tss)+'\n')

        # Probar con TEST
        feed_dict = {train_inputs: TEST_X, train_labels: TEST_Y}
        loss_val = session.run([nce_loss], feed_dict=feed_dict)
        print('TEST batch loss at global-step ' + str(gs) + ': ' + str(loss_val))
        with open('train_results.txt', 'a') as the_file: the_file.write('TEST\t' + str(gs) + '\t' + str(loss_val) + '\t' + str(time.time() - tss)+'\n')


        #------------------------------------------------------------------------------------------------------

    #TODO:PARTE DE TEST


exit()

#
# #-----------------------------------------------
# # Pruebas con GENSIM
# #-----------------------------------------------
#
# LFM = LastFM()
# #LFM.createDict()
# #LFM.getPlaysByUser()
# #LFM.getCleanData(songMin=10,userMin=100,years=[2005,2006,2007,2008,2009], saveAs="datos/CLEAN_DATA_[10,100]")
# #LFM.generaFicherosUsuario(filename="datos/CLEAN_DATA_[10,100].pkl")
# #DATA = LFM.generaDatosW2V(filename="datos/CLEAN_DATA_[10,100].pkl")
#
# cores = multiprocessing.cpu_count()
#
# embedding_size = 200
# window  = 2
# iter = 20
# sg=1
# init_lr = 1 #is the initial learning rate (will linearly drop to min_alpha as training progresses).
# minm_lr = 0.0001
# seed = 100
#
# MODEL_NAME = "w2v_"+str(embedding_size)+"_"+str(window)+"_"+str(iter)+"_"+("SKG" if sg else "CBW")+"_["+str(init_lr)+","+str(minm_lr)+"]"
#
# # creating models will always take a few minutes, load pretrained whenever available
# if os.path.isfile('pretrained/'+MODEL_NAME):
#     print("Cargando modelo existente...")
#     model = Word2Vec.load('pretrained/'+MODEL_NAME)
#
# else:
#     # params are standard 300 dimensions and flag to use skip-gram instead of cbow
#     print("Generando modelo...")
#     DATA = LFM.generaDatosW2V(filename="datos/CLEAN_DATA_[10,100].pkl")
#     model = Word2Vec(DATA, size=embedding_size,seed = seed, sg=sg, window=window,alpha=init_lr, iter=iter,min_alpha=minm_lr ,workers=cores,compute_loss=True)
#     model.save('pretrained/'+MODEL_NAME)
#
# #Grabar resultados en fichero
#
# header = ("EMB_SIZE\tWINDOW\tITER\tSKG\tINIT_LR\tMIN_LR\tSEED\tLOSS")
# writeResultsInFile("TEST_RESULTS.tsv",header)
# loss = str(model.get_latest_training_loss()).replace(".",",")
# line = '\t'.join([str(embedding_size),str(window),str(iter),str(sg),str(init_lr),str(minm_lr),str(seed),loss])
# writeResultsInFile("TEST_RESULTS.tsv",line)
#
# print(loss)
#
#
# DATA = pd.read_pickle("datos/CLEAN_DATA_[10,100].pkl")
#
#
# #Dire Straits - Sultans Of Swing  344193
# #Dire Straits - Money For Nothing  344156
# #Director - Reconnect  344296
# print(model.wv.similarity('344193', '344156'))
# print(model.wv.similarity('344193', '344296'))
#
# RES = model.wv.most_similar(positive=['344193','344156'], negative=['344193'])
#
# for i in RES:
#     item = DATA.loc[(DATA.ID==int(i[0]))]
#     print(item.artname.unique(),item.traname.unique())
#
#
# #---------------------------------------------------------------------
# #TSNE con 2 variables
# #---------------------------------------------------------------------
#
# X = model[model.wv.vocab]
# X = X[:500]
#
#
# exit()
# tsne = TSNE(n_components=2)
# print("Inicio T-SNE")
# X_tsne = tsne.fit_transform(X)
# print("Fin T-SNE")
#
# DT = pd.DataFrame(columns=["x","y"])
# DT['x'] = X_tsne[:, 0]
# DT['y'] = X_tsne[:, 1]
#
# DT2 = DT.loc[(DT['x']<-17)]
#
# index = DT2.index.values
# for i in index:
#     print(X[i])
#
#
# plot = ggplot(data=DT,aesthetics=aes(x="x", y= "y"))
# plot += geom_point(data=DT,aesthetics=aes(x="x", y= "y"),color="b")
# #plot += geom_point(data=DT2,aesthetics=aes(x="x", y= "y"), color="r")
# plot.show()
#
