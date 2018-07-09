# -*- coding: utf-8 -*-

import os.path
import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
import math
import time
import pickle

########################################################################################################################
# Métodos
########################################################################################################################

def getData(window=2,seed = 100,shuffle=True, file=None):

    #Cargar datos del tipo [[canciones user1],[canciones user2],...]
    if(file==None):DATA = np.load("datos/Append/ARRAY_ARRAYS_TRAIN.pkl")
    else:DATA = np.load(file)
    #print("ELIMINAR FILTRO--->");DATA = DATA[:5]

    #Obtener el número total de usuarios
    NUM_USERS = len(DATA)

    #Matriz resultado
    RES_COLS = window + 1 + 1 # USER ID, [C1,C2], C3
    RES = np.empty([0, RES_COLS], dtype=np.int32)

    #Para cada usuario, crear ejemplos
    for u in range(NUM_USERS):
        tss = time.time()

        USR_SONGS = DATA[u]
        USR_SONGS_SIZE = len(USR_SONGS)

        if(USR_SONGS_SIZE<window+1):print(str(u)+": "+str(USR_SONGS_SIZE));continue

        USR_RES_ROWS = USR_SONGS_SIZE - window

        #Crear matriz de retorno
        USR_RES = np.empty([USR_RES_ROWS, RES_COLS], dtype=np.int32)

        #Cambiar id de usuario
        USR_RES[:,0] = u

        #Obtener los datos (uno más para la clase (canción siguiente))
        for i in range(window+1):
            TO = USR_RES_ROWS+i
            USR_RES[:,1+i]= USR_SONGS[i:TO]

        #Finalmente añadir estos datos al conjunto de todos los usuarios
        RES = np.concatenate([RES,USR_RES])

    #Mezclar
    if(shuffle):
        np.random.seed(seed)
        np.random.shuffle(RES)

    #Dividir en TRAIN, DEV y TEST
    #98/1/1 TODO: VALE ASÍ ???????????

    TOTAL_LENGTH = len(RES)
    DEV_LENGTH = int(TOTAL_LENGTH * 0.01)
    TEST_LENGTH = int(TOTAL_LENGTH * 0.01)
    TRAIN_LENGTH = TOTAL_LENGTH - DEV_LENGTH - TEST_LENGTH

    # Train
    TRAIN = RES[:TRAIN_LENGTH,:]
    X = TRAIN[:, :window + 1]
    Y = np.matrix(TRAIN[:, -1]).T  # Hay que pasarla a matriz y transponer
    # Dev
    DEV = RES[TRAIN_LENGTH:TRAIN_LENGTH+DEV_LENGTH,:]
    DEV_X = DEV[:, :window + 1]
    DEV_Y = np.matrix(DEV[:, -1]).T  # Hay que pasarla a matriz y transponer
    # Test
    TEST = RES[TRAIN_LENGTH+DEV_LENGTH:,:]
    TEST_X = TEST[:, :window + 1]
    TEST_Y = np.matrix(TEST[:, -1]).T  # Hay que pasarla a matriz y transponer

    return(X,Y,DEV_X,DEV_Y,TEST_X,TEST_Y, NUM_USERS)

########################################################################################################################
# Llamadas
########################################################################################################################

#Cargar matriz de pesos W2V
W2V_emb_path = 'embeddings/W2V/EMB_MATRIX'
W2V_EMB = np.load(W2V_emb_path)

batch_size = 1024
vocabulary_size = W2V_EMB.shape[0]
embedding_size_w = W2V_EMB.shape[1]  # Tamaño del embedding (capa oculta)
embedding_size_d = embedding_size_w
window=2
seed = 100
num_sampled = 64 # Ejemplos negativos (Para el NSE)
learning_rate = 0.1
epochs = 1500

# Obtener todos lo datos
print("Obteniendo datos...")

model_name = 'complete'
model_path = 'models/d2v_dm/complete'
emb_path = 'embeddings/d2v_dm/complete'
X,Y,DEV_X,DEV_Y,TEST_X,TEST_Y, user_size = getData(window=window,seed=seed, file="datos/Append/ARRAY_ARRAYS_TRAIN_COMPLETE.pkl")

#model_name = 'present'
#model_path = 'models/d2v_dm/present'
#emb_path = 'embeddings/d2v_dm/present'
#X,Y,DEV_X,DEV_Y,TEST_X,TEST_Y, user_size = getData(window=window,seed=seed, file="datos/Append/ARRAY_ARRAYS_TRAIN_PRESENT.pkl")

print("Datos obtenidos!")

# Carpeta donde se almacena el modelo de la red
complete_path = model_path+"/"+model_name

#Creación del grafo de TF.
graph = tf.Graph()

with graph.as_default():

    #Número global de iteraciones
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    #Datos de entrada -----------------------------------------------------------------------------------------------------------------------------
    #Array del tamaño del batch con las X
    train_dataset = tf.placeholder(tf.int32, shape=[None, window + 1], name="train_inputs")

    #Array del tamaño del batch con las Y asociadas
    train_labels = tf.placeholder(tf.int32, shape=[None, 1], name="train_labels")

    #Conexión primera y segunda capa --------------------------------------------------------------------------------------------------------------
    # Matriz W de embeddigs de las canciones obtenida en el W2V previo
    word_embeddings = tf.Variable(W2V_EMB,trainable=False, name="word_embeddings") # FUNCIONA??? https://stackoverflow.com/questions/37326002/is-it-possible-to-make-a-trainable-variable-not-trainable/37327561

    # Matriz D de embeddigs para los usuarios (lo que se aprende ahora)
    doc_embeddings = tf.Variable(tf.random_uniform([user_size, embedding_size_d], -1.0, 1.0), name="doc_embeddings")

    #Conexión segunda y última capa ---------------------------------------------------------------------------------------------------------------

    combined_embed_vector_length = embedding_size_w + embedding_size_d
    # softmax weights, W and D vectors should be concatenated before applying softmax
    weights = tf.Variable(tf.truncated_normal([vocabulary_size, combined_embed_vector_length], stddev=1.0 / math.sqrt(combined_embed_vector_length)), name="weights")
    # softmax biases
    biases = tf.Variable(tf.zeros([vocabulary_size]), name="biases")

    #Cálculo de LOSS y optimizador- ---------------------------------------------------------------------------------------------------------------

    embed = []
    #Obtener el embedding del documento (Columna 0 del batch)
    embed_d = tf.nn.embedding_lookup(doc_embeddings, train_dataset[:, 0])
    embed.append(embed_d)

    #Obtener la SUMA de los embeddings de las canciones (Columnas restantes del batch)
    #El primero se hace fuera. (No se puede crear un vector de 0 inicial e ir añadiendo dado que no se conoce el tamaño del batch)
    embed_w = tf.nn.embedding_lookup(word_embeddings, train_dataset[:, 1])

    for j in range(2,window+1):
        embed_w += tf.nn.embedding_lookup(word_embeddings, train_dataset[:, j])
    embed.append(embed_w)

    #Se concatena el embedding del usuario con la SUMA de los de las canciones (No se suma el embedding del doc) (hasta ahora era un array de 2 elementos)
    embed = tf.concat(embed, 1)


    nce_loss = tf.nn.nce_loss(weights=weights,
                          biases=biases,
                          labels=train_labels,
                          inputs=embed,
                          num_sampled=num_sampled,
                          num_classes=vocabulary_size,
                          name="nce_loss")

    nce_loss = tf.reduce_mean(nce_loss)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(nce_loss,global_step=global_step)

    init = tf.global_variables_initializer()

    #Crear objeto encargado de almacenar la red
    saver = tf.train.Saver(max_to_keep=1)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

with tf.Session(graph=graph,config=tf.ConfigProto(gpu_options=gpu_options)) as session:

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

    E = session.run(doc_embeddings)
    E.dump(emb_path+"/EMB_MATRIX")

    print("Embedings obtenidos!")
    ################################################################################

    exit()



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

            feed_dict = {train_dataset: batch_inputs, train_labels: batch_context}
            _, loss_val, gs = session.run([optimizer, nce_loss,global_step], feed_dict=feed_dict)


            if (gs % 1000 == 0 and gs>0):

                # The average loss is an estimate of the loss over the last 100 batches.
                print('\tLast batch loss at global-step ', gs, ': ', loss_val)

                # Now, save the graph
                #saver.save(session, complete_path, global_step=global_step)

        print(time.time()-tss)
        saver.save(session, complete_path, global_step=global_step)

        # Probar con DEV
        feed_dict = {train_dataset: DEV_X, train_labels: DEV_Y}
        loss_val = session.run([nce_loss], feed_dict=feed_dict)
        print('DEV batch loss at global-step ' + str(gs) + ': ' + str(loss_val))
        with open('train_results_D2V_DM_'+model_name+'.txt', 'a') as the_file: the_file.write('DEV\t' + str(gs) + '\t' + str(loss_val) + '\t' + str(time.time() - tss)+'\n')

        # Probar con TEST
        feed_dict = {train_dataset: TEST_X, train_labels: TEST_Y}
        loss_val = session.run([nce_loss], feed_dict=feed_dict)
        print('TEST batch loss at global-step ' + str(gs) + ': ' + str(loss_val))
        with open('train_results_D2V_DM_'+model_name+'.txt', 'a') as the_file: the_file.write('TEST\t' + str(gs) + '\t' + str(loss_val) + '\t' + str(time.time() - tss)+'\n')


        #------------------------------------------------------------------------------------------------------

    #TODO:PARTE DE TEST


