# -*- coding: utf-8 -*-

import os.path
import os
import tensorflow as tf
import numpy as np
import random
import time
import math
import pandas as pd
import pickle
from collections import Counter
from scipy.stats import linregress
from tensorflow.python.framework import ops as framework_ops

########################################################################################################################
# Métodos
########################################################################################################################

def getData(REAL_SONG_IDS=[],data ="datos/Append/ARRAY_ARRAYS_REC.pkl",shuffle=True, seed=100, train=.7):

    DATA = np.load(data)

    #DATA = DATA[:5];print("ELIMINAR FILTRO")

    #User, cancion 1, cancion 2, cancion 3
    TRAIN = np.empty([0, 4], dtype=np.int32)
    DEV = np.empty([0, 4], dtype=np.int32)
    TEST = np.empty([0, 4], dtype=np.int32)

    for u in range(len(DATA)):

        # Eliminar canciones escuchadas que no tienen embedding
        USER_PLAYS = DATA[u]
        USER_PLAYS = pd.DataFrame(list(map(int, USER_PLAYS)))
        USER_PLAYS = USER_PLAYS.loc[USER_PLAYS[0].isin(REAL_SONG_IDS)][0].values

        #Separar en train dev test
        TRAIN_ITEMS = int(len(USER_PLAYS)*train)
        DEV_ITEMS = int((len(USER_PLAYS)-TRAIN_ITEMS)/2)

        TRAIN_DATA = USER_PLAYS[:TRAIN_ITEMS]
        DEV_DATA = USER_PLAYS[TRAIN_ITEMS:TRAIN_ITEMS+DEV_ITEMS]
        TEST_DATA = USER_PLAYS[TRAIN_ITEMS+DEV_ITEMS:]

        # Crear ejemplos
        USR_TRAIN = np.empty([len(TRAIN_DATA)-2, 4], dtype=np.int32)
        USR_TRAIN[:,0] = u
        USR_TRAIN[:,1] = TRAIN_DATA[:len(TRAIN_DATA)-2]
        USR_TRAIN[:,2] = TRAIN_DATA[1:-1]
        USR_TRAIN[:,3] = TRAIN_DATA[2:]

        USR_DEV = np.empty([len(DEV_DATA) -2, 4], dtype=np.int32)
        USR_DEV[:, 0] = u
        USR_DEV[:, 1] = DEV_DATA[:len(DEV_DATA)-2]
        USR_DEV[:, 2] = DEV_DATA[1:-1]
        USR_DEV[:, 3] = DEV_DATA[2:]

        USR_TEST = np.empty([len(TEST_DATA)-2, 4], dtype=np.int32)
        USR_TEST[:, 0] = u
        USR_TEST[:, 1] = TEST_DATA[:len(TEST_DATA) -2]
        USR_TEST[:, 2] = TEST_DATA[1:-1]
        USR_TEST[:, 3] = TEST_DATA[2:]

        # Añadir al global
        TRAIN = np.concatenate([TRAIN,USR_TRAIN])
        DEV = np.concatenate([DEV,USR_DEV])
        TEST = np.concatenate([TEST,USR_TEST])

    #Mezclar
    if(shuffle):
        np.random.seed(seed)
        np.random.shuffle(TRAIN)
        np.random.shuffle(DEV)
        np.random.shuffle(TEST)

    return TRAIN[:,:3],np.matrix(TRAIN[:,-1]).T,DEV[:,:3],np.matrix(DEV[:,-1]).T,TEST[:,:3],np.matrix(TEST[:,-1]).T,len(DATA)

def getBatchData(X, Y, batch_size, step, ITRS, RES_ITMS):

    # Si se añadió una iteración, introducir los datos restantes
    if (step == ITRS - 1 and RES_ITMS > 0):
        batch_inputs = X[(ITRS - 1) * batch_size:]
        batch_context = Y[(ITRS - 1) * batch_size:]
    # Si no, es un batch normal
    else:
        st = step * batch_size
        nd = (step + 1) * batch_size

        batch_inputs = X[st:nd]
        batch_context = Y[st:nd]
    return(batch_inputs,batch_context)

def trainNet(X,Y, DX,DY,TX,TY,config, user_size, train=True, save=True):

    def getSlope(data):
        r = linregress(range(len(data)), data)
        return r.slope

    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    epochs = config['epochs']

    user_size = user_size
    song_size = W2V.shape[0]

    user_emb_size = 128
    song_emb_size = 64
    hidden_size = 32

    profile_size = user_emb_size + song_emb_size + song_emb_size

    num_sampled = 64  # Ejemplos negativos (Para el NSE)

    model_name = 'model'
    model_path = 'models/' + TEST_NAME + '/model'
    complete_path = model_path + "/" + model_name

    # Se almacena la loss de dev de las 5 últimas epochs
    slope_size = 100
    train_hist = []
    dev_hist = []

    # Creación del grafo de TF.
    graph = tf.Graph()

    with graph.as_default():

        if(config["seed"]!=1):
            tf.set_random_seed(config["seed"])

        # Número global de iteraciones
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        # Datos de entrada -----------------------------------------------------------------------------------------------------------------------------
        # Array del tamaño del batch con las X
        train_dataset = tf.placeholder(tf.int32, shape=[None, 3], name="train_inputs")

        # Array del tamaño del batch con las Y asociadas
        train_labels = tf.placeholder(tf.float32, shape=[None, 1], name="train_labels")

        # Embeddings ----------------------------------------------------------------------------------------------------------------------------------

        D2V_EMB_VRB = tf.Variable(D2V_EMB, trainable=False, name="doc_embeddings")

        W2V_EMB_VRB = tf.Variable(W2V, trainable=False, name="word_embeddings")

        # Operaciones -----------------------------------------------------------------------------------------------------------------

        # Embedding de documento
        ed = tf.nn.embedding_lookup(D2V_EMB_VRB, train_dataset[:, 0])

        # Embedding de cancion 1
        ec1 = tf.nn.embedding_lookup(W2V_EMB_VRB, train_dataset[:, 1])

        # Embedding de cancion 2
        ec2 = tf.nn.embedding_lookup(W2V_EMB_VRB, train_dataset[:, 2])

        # Cálculo de LOSS y optimizador- ---------------------------------------------------------------------------------------------------------------

        embed = []
        embed.append(ed)
        embed.append(ec1)
        embed.append(ec2)
        embed = tf.concat(embed, 1)

        # softmax weights, W and D vectors should be concatenated before applying softmax
        weights = tf.Variable(tf.truncated_normal([song_size, profile_size], stddev=1.0 / math.sqrt(song_size)), name="weights")
        # softmax biases
        biases = tf.Variable(tf.zeros([song_size]), name="biases")

        # Cálculo de LOSS y optimizador- ---------------------------------------------------------------------------------------------------------------
        nce_loss = tf.nn.nce_loss(weights=weights,
                                  biases=biases,
                                  labels=train_labels,
                                  inputs=embed,
                                  num_sampled=num_sampled,
                                  num_classes=song_size,
                                  name="nce_loss")

        batch_loss = nce_loss
        nce_loss = tf.reduce_mean(nce_loss)

        logits = tf.matmul(embed, tf.transpose(weights))
        logits = tf.nn.bias_add(logits, biases)

        top_v,top_i = tf.nn.top_k(logits,k=song_size)

        # logits = tf.nn.softmax(logits,axis=0)

        # optimizer = tf.train.AdamOptimizer(learning_rate).minimize(nce_loss,global_step=global_step)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(nce_loss, global_step=global_step)

        # Inicializar variables
        init = tf.global_variables_initializer()

        # Crear objeto encargado de almacenar la red
        saver = tf.train.Saver(max_to_keep=1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=config) as session:
    #with tf.Session(graph=graph) as session:

        # We must initialize all variables before we use them.
        init.run()


        if(save):
            # Obtener el checkpoint
            ckpt = tf.train.get_checkpoint_state(model_path)

            # Si existe el checkpoint restaurar
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(session, ckpt.model_checkpoint_path)

        #Si se desea entrenar
        if(train):
            #------------------------------------------------------------------------------------------------------
            # TRAIN
            #------------------------------------------------------------------------------------------------------

            ITRS= int(len(X)/batch_size)
            RES_ITMS = len(X) - (ITRS*batch_size)
            if(RES_ITMS>0):ITRS+=1 #Si no es división exacta, se añade una iteración

            for e in range(epochs):
                TRAIN_LOSS = 0.0

                print("Epoch "+str(e))

                for step in range(ITRS):

                    batch_inputs, batch_context = getBatchData(X,Y,batch_size,step,ITRS,RES_ITMS)

                    feed_dict = {train_dataset: batch_inputs, train_labels: batch_context}

                    _, loss_val, batch_lss, gs = session.run([train_step, nce_loss,batch_loss,global_step], feed_dict=feed_dict)

                    TRAIN_LOSS += np.sum(batch_lss)

                #print('Epoch '+str(e)+', last batch loss: '+str(loss_val))

                if (save): saver.save(session, complete_path, global_step=global_step)

                #------------------------------------------------------------------------------------------------------
                # DEV
                #------------------------------------------------------------------------------------------------------

                if(len(DX)>0):

                    ITRS_TEST = int(len(DX) / batch_size)
                    RES_ITMS_TEST = len(DX) - (ITRS_TEST * batch_size)
                    TEST_LOSS = 0.0
                    if (RES_ITMS_TEST > 0): ITRS_TEST += 1 # Si no es división exacta, se añade una iteración


                    for step in range(ITRS_TEST):

                        batch_inputs, batch_context = getBatchData(DX,DY,batch_size,step,ITRS_TEST,RES_ITMS_TEST)

                        feed_dict = {train_dataset: batch_inputs, train_labels: batch_context}
                        loss_val,batch_lss = session.run([nce_loss,batch_loss], feed_dict=feed_dict)

                        TEST_LOSS+=np.sum(batch_lss)

                    dev_hist.append(TEST_LOSS / (len(DX) * 1.0))
                    train_hist.append(TRAIN_LOSS / (len(X) *  1.0))

                if (len(dev_hist) >= slope_size):
                    # Calcular la pendiente
                    print(getSlope(dev_hist[-slope_size:]))
                    print(dev_hist[-slope_size:])

                    if (getSlope(dev_hist[-slope_size:]) > -1e-5 or e==epochs-1):

                        def printArray(data,data2):
                            for i in range(len(data)):
                                a=data[i]
                                b=data2[i]
                                print(str(a).replace(".",",")+"\t"+str(b).replace(".",","))

                        print("-" * 50)
                        print(e)
                        printArray(dev_hist,train_hist)
                        return
        else:
            # ------------------------------------------------------------------------------------------------------
            # TEST FINAL
            # ------------------------------------------------------------------------------------------------------
            batch_size = 300

            IN_TOP_5 = 0
            IN_TOP_10 = 0
            IN_TOP_20 = 0
            IN_TOP_50 = 0
            IN_TOP_100 = 0
            IN_TOP_1000 = 0
            real_positions = []

            items = 0

            ITRS_TEST = int(len(TX) / batch_size)
            RES_ITMS_TEST = len(TX) - (ITRS_TEST * batch_size)
            if (RES_ITMS_TEST > 0): ITRS_TEST += 1  # Si no es división exacta, se añade una iteración

            for step in range(ITRS_TEST):
                #tss = time.time()

                batch_inputs, batch_context = getBatchData(TX, TY, batch_size, step, ITRS_TEST, RES_ITMS_TEST)
                top_values,top_indx = session.run([top_v,top_i], feed_dict={train_dataset: batch_inputs, train_labels: batch_context})

                for j in range(len(batch_context)):
                    real_position = int(np.where(top_indx[j, :] == batch_context[j, 0])[0])
                    real_positions.append(real_position)

                    IN_TOP_5 += 1 if real_position <= 5 else 0
                    IN_TOP_10 += 1 if real_position <= 10 else 0
                    IN_TOP_20 += 1 if real_position <= 20 else 0
                    IN_TOP_50 += 1 if real_position <= 50 else 0
                    IN_TOP_100 += 1 if real_position <= 100 else 0
                    IN_TOP_1000 += 1 if real_position <= 1000 else 0

                items += len(batch_inputs)*1.0

                PTF = IN_TOP_5 / items
                PTT = IN_TOP_10 / items
                PTTW = IN_TOP_20 / items
                PTFT = IN_TOP_50 / items
                PTCIEN = IN_TOP_100 / items
                PTMIL = IN_TOP_1000 / items

                print(np.mean(real_positions), np.std(real_positions), np.median(real_positions))
                print(str(PTF).replace(".", ",") + "\t" + str(PTT).replace(".", ",") + "\t"
                      + str(PTTW).replace(".",",") + "\t" + str(PTFT).replace(".", ",") + "\t"
                      + str(PTCIEN).replace(".", ",") + "\t" + str(PTMIL).replace(".",","))

                #print(time.time()-tss)


            print(np.mean(real_positions), np.std(real_positions), np.median(real_positions))

        #------------------------------------------------------------------------------------------------------

########################################################################################################################
# Llamadas
########################################################################################################################

TEST_NAME="test_033"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#Se obtinenen cada uno de los embeddings aprendidos (incluidas las canciones)
D2V_DM_COMPLETE   = np.load('embeddings/d2v_dm/complete/EMB_MATRIX')
D2V_DBOW_COMPLETE = np.load('embeddings/d2v_dbow/complete/EMB_MATRIX')
W2V  = np.load('embeddings/w2v/EMB_MATRIX')

#Se concatenan los embeddings de los usuarios para generar el perfil
D2V_EMB = np.concatenate((D2V_DM_COMPLETE,D2V_DBOW_COMPLETE),axis=1) # El consolidado = Consolidado DM + Consolidado DBOW

#Se cargan los datos de train de las canciones para conocer los IDs reales
DATA = np.load("datos/Extend/ARRAY_ARRAYS_TRAIN.pkl")

#Obtener una lista con los ID de canciones que no están el la matriz de embeddings
ALL = set(list(range(W2V.shape[0])))
REAL = set(list(set(list(map(int, DATA)))))

########################################################################################################################
# TENSORFLOW
########################################################################################################################

#-----------------------------------------------------------------------------------------------------------------------
# Grid Search + Early Stopping
#-----------------------------------------------------------------------------------------------------------------------
'''
X, Y, DX, DY, TX, TY, user_size = getData(REAL_SONG_IDS=list(REAL))
rates = [0.000001,1, 0.01, 0.0001]

for r in rates:
    config = {"batch_size": 1024, "learning_rate": r, "epochs": 1000, "seed": 100}

    print("Learning Rate: "+str(r))
    print("-" * 50)
    trainNet(X, Y, DX, DY, TX, TY, config, user_size, train=True, save=False)
    print("-"*50)

exit()
'''
#-----------------------------------------------------------------------------------------------------------------------
# Train
#-----------------------------------------------------------------------------------------------------------------------
'''
X, Y, DX, DY, TX, TY, user_size = getData(REAL_SONG_IDS=list(REAL))

#Para esta fase, se une TRAIN y DEV
X = np.concatenate((X,DX),axis=0)
Y = np.concatenate((Y,DY),axis=0)

DX=[]
DY=[]

config = {"batch_size": 1024, "learning_rate": 0.01, "epochs":1000, "seed":100}
trainNet(X, Y, DX, DY, TX, TY, config, user_size, train=True, save=True)

exit()
'''
#-----------------------------------------------------------------------------------------------------------------------
# Test
#-----------------------------------------------------------------------------------------------------------------------

X, Y, DX, DY, TX, TY, user_size = getData(REAL_SONG_IDS=list(REAL))

config = {"batch_size": 1024, "learning_rate": 0.01, "epochs":1000, "seed":100}
trainNet(X, Y, DX, DY, TX, TY, config, user_size,train=False, save=True)