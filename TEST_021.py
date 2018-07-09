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
from scipy.stats import linregress
from collections import Counter

########################################################################################################################
# Métodos
########################################################################################################################

def getData(REAL_SONG_IDS=[],data ="datos/Append/ARRAY_ARRAYS_REC.pkl",pastData="datos/Append/ARRAY_ARRAYS_TRAIN_COMPLETE.pkl",shuffle=True, seed=100, train=.7):

    DATA = np.load(data)
    PAST_DATA = np.load(pastData)
    REAL_SET = set(REAL_SONG_IDS)

    random.seed(seed)

    #DATA = DATA[:20];print("ELIMINAR FILTRO")

    #User, cancion, +1 o 0
    TRAIN = np.empty([0, 3], dtype=np.int32)
    DEV = np.empty([0, 3], dtype=np.int32)
    TEST = np.empty([0, 3], dtype=np.int32)

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

        # Obtener IDS de canciones que no han sido escuchadas en TRAIN_COMPLETE y en TRAIN De ahora
        USED_IDS = set(list(map(int, PAST_DATA[u])))
        USED_IDS.update(TRAIN_DATA)
        TRAIN_NOT_USED = list(REAL_SET.difference(USED_IDS))

        # Obtener IDS de canciones que no han sido escuchadas en TRAIN_COMPLETE , en TRAIN y DEV
        USED_IDS.update(DEV_DATA)
        DEV_NOT_USED = list(REAL_SET.difference(USED_IDS))

        # Obtener IDS de canciones que no han sido escuchadas en TRAIN_COMPLETE , en TRAIN y DEV y TEST
        USED_IDS.update(TEST_DATA)
        TEST_NOT_USED = list(REAL_SET.difference(USED_IDS))

        # Crear casos de train
        TRAIN_RES = np.empty([len(TRAIN_DATA) * 2, 3], dtype=np.int32)
        TRAIN_RES[:, 0] = u
        TRAIN_RES[:len(TRAIN_DATA), 1] = TRAIN_DATA
        TRAIN_RES[:len(TRAIN_DATA), 2] = 1

        TRAIN_RES[len(TRAIN_DATA):, 1] = random.sample(TRAIN_NOT_USED, len(TRAIN_DATA))
        TRAIN_RES[len(TRAIN_DATA):, 2] = 0

        TRAIN = np.concatenate([TRAIN, TRAIN_RES])

        # Crear casos de dev
        DEV_RES = np.empty([len(DEV_DATA) * 2, 3], dtype=np.int32)
        DEV_RES[:, 0] = u
        DEV_RES[:len(DEV_DATA), 1] = DEV_DATA
        DEV_RES[:len(DEV_DATA), 2] = 1

        DEV_RES[len(DEV_DATA):, 1] = random.sample(DEV_NOT_USED, len(DEV_DATA))
        DEV_RES[len(DEV_DATA):, 2] = 0

        DEV = np.concatenate([DEV, DEV_RES])

        # Crear casos de test
        TEST_RES = np.empty([len(TEST_DATA) * 2, 3], dtype=np.int32)
        TEST_RES[:, 0] = u
        TEST_RES[:len(TEST_DATA), 1] = TEST_DATA
        TEST_RES[:len(TEST_DATA), 2] = 1

        TEST_RES[len(TEST_DATA):, 1] = random.sample(TEST_NOT_USED, len(TEST_DATA))
        TEST_RES[len(TEST_DATA):, 2] = 0

        TEST = np.concatenate([TEST, TEST_RES])

    #Mezclar
    if(shuffle):
        np.random.seed(seed)
        np.random.shuffle(TRAIN)
        np.random.shuffle(DEV)
        np.random.shuffle(TEST)


    return TRAIN[:,:2],np.matrix(TRAIN[:,-1]).T,DEV[:,:2],np.matrix(DEV[:,-1]).T,TEST[:,:2],np.matrix(TEST[:,-1]).T,len(DATA)

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

def trainNet(X,Y, DX,DY,TX,TY,config, user_size,train=True, save=True):

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

    num_sampled = 64  # Ejemplos negativos (Para el NSE)

    model_name = 'model'
    model_path = 'models/' + TEST_NAME + '/model'
    complete_path = model_path + "/" + model_name

    # Se almacena la loss de dev de las 5 últimas epochs
    slope_size = 100
    train_hist=[]
    dev_hist=[]

    # Creación del grafo de TF.
    graph = tf.Graph()

    with graph.as_default():

        if(config["seed"]!=1):
            tf.set_random_seed(config["seed"])

        # Número global de iteraciones
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        # Datos de entrada -----------------------------------------------------------------------------------------------------------
        # Array del tamaño del batch con las X
        train_dataset = tf.placeholder(tf.int32, shape=[None, 2], name="train_inputs")

        # Array del tamaño del batch con las Y asociadas
        train_labels = tf.placeholder(tf.float32, shape=[None, 1], name="train_labels")

        # Embeddings -----------------------------------------------------------------------------------------------------------------

        # Matriz W de embeddigs de las canciones obtenida en el W2V previo

        O1 = tf.Variable(tf.truncated_normal([user_size, user_emb_size], mean=0.0, stddev=1.0 / math.sqrt(user_emb_size)), name="O1")
        BO1 = tf.Variable(tf.zeros([user_emb_size]), name="BO1")

        W2V_EMB = tf.Variable(W2V, trainable=False, name="word_embeddings")  # FUNCIONA??? https://stackoverflow.com/questions/37326002/is-it-possible-to-make-a-trainable-variable-not-trainable/37327561

        T1 = tf.Variable(tf.truncated_normal([user_emb_size, hidden_size], mean=0.0, stddev=1.0 / math.sqrt(hidden_size)), name="T1")
        B1 = tf.Variable(tf.zeros([hidden_size]), name="B1")

        T2 = tf.Variable(tf.truncated_normal([song_emb_size, hidden_size], mean=0.0, stddev=1.0 / math.sqrt(hidden_size)), name="T2")
        B2 = tf.Variable(tf.zeros([hidden_size]), name="B2")

        # Operaciones -----------------------------------------------------------------------------------------------------------------

        # Embedding de documento
        ed = tf.nn.embedding_lookup(O1, train_dataset[:, 0])
        ed = ed + BO1

        # Embedding de cancion
        ec = tf.nn.embedding_lookup(W2V_EMB, train_dataset[:, 1])

        # Transformar a 32 documento
        hd = tf.matmul(ed,T1)+B1

        # Transformar a 32 cancion
        hw = tf.matmul(ec,T2)+B2

        # Obtener el producto escalar
        dot_prod = tf.reduce_sum(tf.multiply(hd, hw), 1, keep_dims=True)

        # Cálculo de LOSS y optimizador- ---------------------------------------------------------------------------------------------

        # Obtener la loss
        softplus_batch = tf.nn.softplus((1 - 2 * train_labels) * dot_prod)
        loss_softplus = tf.reduce_mean(softplus_batch)

        # Minimizar la loss
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss_softplus,global_step=global_step)

        # Obtener la probabilidad
        prob = tf.sigmoid(dot_prod)

        # Inicializar variables
        init = tf.global_variables_initializer()

        # Crear objeto encargado de almacenar la red
        saver = tf.train.Saver(max_to_keep=1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=config) as session:

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

                    _, loss_val,softplus, gs = session.run([train_step, loss_softplus,softplus_batch,global_step], feed_dict=feed_dict)

                    TRAIN_LOSS += np.sum(softplus[:, 0])

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
                        loss_val, softplus = session.run([loss_softplus,softplus_batch], feed_dict=feed_dict)

                        TEST_LOSS+=np.sum(softplus[:,0])

                    dev_hist.append(TEST_LOSS/(len(DX)*1.0))
                    train_hist.append(TRAIN_LOSS/(len(X)*1.0))

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

            TRUE_POSITIVE = 0
            FALSE_NEGATIVE = 0
            FALSE_POSITIVE = 0
            TRUE_NEGATIVE = 0

            ITRS_TEST = int(len(TX) / batch_size)
            RES_ITMS_TEST = len(TX) - (ITRS_TEST * batch_size)
            TEST_LOSS = 0.0
            # Si no es división exacta, se añade una iteración
            if (RES_ITMS_TEST > 0): ITRS_TEST += 1

            for step in range(ITRS_TEST):
                # Si se añadió una iteración, introducir los datos restantes
                if (step == ITRS_TEST - 1 and RES_ITMS_TEST > 0):
                    batch_inputs = TX[(ITRS_TEST - 1) * batch_size:]
                    batch_context = TY[(ITRS_TEST - 1) * batch_size:]
                # Si no, es un batch normal
                else:
                    st = step * batch_size
                    nd = (step + 1) * batch_size

                    batch_inputs = TX[st:nd]
                    batch_context = TY[st:nd]

                feed_dict = {train_dataset: batch_inputs, train_labels: batch_context}
                loss_val,loss_batch,  p = session.run([loss_softplus,softplus_batch, prob], feed_dict=feed_dict)

                TEST_LOSS += np.sum(loss_batch)

                # Cálculo de TP/TN/FN/FP para cada batch

                for i in range(len(p)):
                    real = batch_context[i, 0]
                    pred = p[i, 0]

                    if (real == 1) and (pred > 0.5): TRUE_POSITIVE += 1
                    if (real == 1) and (pred <= 0.5): FALSE_NEGATIVE += 1
                    if (real == 0) and (pred > 0.5): FALSE_POSITIVE += 1
                    if (real == 0) and (pred <= 0.5): TRUE_NEGATIVE += 1

            PRECISION = TRUE_POSITIVE / (TRUE_POSITIVE + FALSE_POSITIVE * 1.0)
            RECALL = TRUE_POSITIVE / (TRUE_POSITIVE + FALSE_NEGATIVE * 1.0)

            F1 = 2.0 * ((PRECISION * RECALL) / (PRECISION + RECALL))

            print("-" * 50)
            print("LOSS:\t" + str(TEST_LOSS / (len(TX) * 1.0)))
            print("PRECISION:\t" + str(PRECISION))
            print("RECALL:\t" + str(RECALL))
            print("F1:\t" + str(F1))
            print("-" * 50)
            print("")
        #------------------------------------------------------------------------------------------------------

########################################################################################################################
# Llamadas
########################################################################################################################

TEST_NAME="test_021"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#Para conocer los embeddings de las canciones
W2V  = np.load('embeddings/w2v/EMB_MATRIX')

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

config = {"batch_size": 1024, "learning_rate": 0.0001, "epochs": 102, "seed": 100}
trainNet(X, Y, DX, DY, TX, TY, config, user_size,train=True,save=True)

exit()
'''
#-----------------------------------------------------------------------------------------------------------------------
# Test
#-----------------------------------------------------------------------------------------------------------------------


X, Y, DX, DY, TX, TY, user_size = getData(REAL_SONG_IDS=list(REAL))

config = {"batch_size": 1024, "learning_rate": 0.0001, "epochs": 102, "seed": 100}
trainNet(X, Y, DX, DY, TX, TY, config, user_size,train=False, save=True)
