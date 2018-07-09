# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import scipy.sparse as sp


import sys, time, os
import sqlite3 as lite
import fileinput, csv
from ggplot import *
import collections
from difflib import SequenceMatcher
import pickle
from datetime import datetime, timedelta
import re

import tensorflow as tf

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class LastFM:

    DATA_PATH = "/Users/pablo/Desktop/W2VEC/Datasets/lastfm"
    DATABASE_NAME = "LastFM.sqlite"

    ########################################################################
    # Métodos
    ########################################################################

    def __init__(self):
        self.DATA_PATH = "datos/"

        #reload(sys)
        #sys.setdefaultencoding('utf8')

        pd.set_option('display.max_rows', 10)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)

    def doQuery(self,SQL):

        con = lite.connect(self.DATA_PATH+"/"+self.DATABASE_NAME)
        con.text_factory = lambda x: str(x, 'utf-8', 'ignore')
        dft = pd.read_sql_query(SQL, con)
        return dft

    def getUserDataReport(self):

        USRS = self.doQuery("Select * from user")

        print("User;Género;Edad;País;Registro;Nº canciones escuchadas;Artista más escuchado;Artista más escuchado(Veces);Canción más escuchada;Canción más escuchada(Veces)")

        for u, r in USRS.iterrows():
            USER_ID = str(r.id)

            LINE = USER_ID+";"+str(r.gender)+";"+str(r.age)+";"+str(r.country)+";"+str(r.registered)


            RET = doQuery("Select userid,timestamp,artname,traname from user_track where userid ='"+USER_ID+"' ")

            MC_ARTIST = collections.Counter(RET.artname).most_common(1)[0]
            MC_SONG = collections.Counter(RET.traname).most_common(1)[0]

            LINE+=";"+str(len(RET))+";"+str(MC_ARTIST[0])+";"+str(MC_ARTIST[1])+";"+str(MC_SONG[0])+";"+str(MC_SONG[1])

            print(LINE)

    def getAllDataReport(self):

        def getMC(Name, COL):

            MC_COUNTRIES = collections.Counter(COL)
            MC_COUNTRIES = MC_COUNTRIES.most_common(len(MC_COUNTRIES.values()))
            MC_COUNTRY = MC_COUNTRIES[0]

            printW(Name+": " + str(len(MC_COUNTRIES)))

            for i in range(len(MC_COUNTRIES)):
                p = MC_COUNTRIES[i]
                #print("\t· " + str(p[0]) + " [" + str(p[1]) + "]")
                print(str(p[0]) + "\t" + str(p[1]))

            print("")

        SNG = self.doQuery("Select * from user_track")
        USR = self.doQuery("Select * from user")

        print("Datos obtenidos....\n\n")

        SNG["arsongs"] = SNG.artname+" - "+SNG.traname

        MC_ARTISTS = collections.Counter(SNG.artname)
        MC_ARTIST = MC_ARTISTS.most_common(1)[0]

        MC_SONGS = collections.Counter(SNG.arsongs)
        MC_SONG = MC_SONGS.most_common(1)[0]

        self.printW("Usuarios: "+str(len(USR))+"\n")

        getMC("Géneros",USR.gender)
        getMC("Edades",USR.age.astype(str))
        getMC("Países",USR.country)

        self.printW("Artistas: "+str(len(MC_ARTISTS.values())))
        self.printW(MC_ARTIST)
        self.printW("Canciones: "+str(len(MC_SONGS.values())))
        self.printW(MC_SONG)

        MC_SONGS = collections.Counter(SNG.traname)
        MC_SONG = MC_SONGS.most_common(1)[0]

        self.printW("Canciones: "+str(len(MC_SONGS.values())))
        self.printW(MC_SONG)

        return False

    def createDict(self, pickleName="datos/DUP_ID_DICT", similarity=0.95):

        '''
        Este método genera un diccionario donde, para cada nombre de canción diferente, se le asigna un ID.
        Si existen repetidos , les asigna a todos el mismo id, de forma que (a,b,b',b',b',c) tendrían ids (0,1,1,1,1,2)

        Funcionamiento básico.
         · Para cada artista, se obtiene su lista de canciones
         · Se compara cada cancion con el resto y, si existen similares se les asigna a todas (las similares y la que se compara) el mismo id.

        Posee alguna optimización para reducir el orden de complejidad a logaritmico.

        Los resultados se almacenan en dos pickles:
            · DUP_ID_DICT => Diccionario con clave = Nombre de cancion y valor= ID
            · DUP_ID_DICT_INV => Diccionario anterior inevertido (clave=valor,valor=clave)

        Genera tambien los pickles:
            · ALL_SONGS => Todas las canciones (incluidas repeticiones)
            · ALL_SONGS_UNIQUE => Las canciones sin repeticion
            · ALL_SONGS_ID_COUNT => Todas las canciones con los id generados y el número de veces que aparece cada canción


        :param: Grado de similitud deseado.
        :return: Diccionario con, nombre de cancion => Id asignado. Además se guarda en un pickle
        '''

        ####################################################################################################################

        def similar(a, b):
            # Metodo para ver si 2 strings son parecidos. Retorna porcentaje en tanto por uno
            return SequenceMatcher(None, str(a), str(b)).ratio()

        def checkDuplicates(CURR_ID, SONG_LIST, ARTIST):
            # A partir de la lista de canciones, se encarga de detectar duplicados e incremetar el id actual
            # el cual retorna para continuar en la función principal.

            AVOID_LIST = []
            RES = {}

            TEMP_LIST = SONG_LIST
            i = 0

            regex = r"(\d)"

            for s in SONG_LIST:
                TEMP_LIST = TEMP_LIST[1:]

                SIMILARS = []

                # Si ya se visitó en una iteración anterior nada
                if s in AVOID_LIST:
                    continue

                # Buscar items similares del actual
                for s2 in TEMP_LIST:
                    SML = similar(s, s2)

                    # Si ya se visitó en una iteración anterior nada
                    if s2 in AVOID_LIST:
                        continue

                    # Si son muy similares
                    if (SML > similarity):

                        m1 = re.finditer(regex, s)
                        m2 = re.finditer(regex, s2)
                        m1_items = []
                        m2_items = []
                        arequal = True

                        for matchNum, match in enumerate(m1):
                            m1_items.append(match)
                        for matchNum, match in enumerate(m2):
                            m2_items.append(match)

                        if (len(m1_items) == len(m2_items)):
                            for i in range(len(m2_items)):
                                arequal = m1_items[i].group() == m2_items[i].group()

                                if not arequal: break
                        else:
                            arequal = False

                        # Si son iguales
                        if (arequal):
                            SIMILARS.append(s2)
                            AVOID_LIST.append(s2)

                i += 1

                # Si el item posee repetidos, asignar el mismo id a todos
                if (len(SIMILARS) > 0):
                    SIMILARS.append(s)
                    for i in range(len(SIMILARS)):
                        RES.update({str(ARTIST) + " - " + str(SIMILARS[i]): CURR_ID});
                    CURR_ID += 1;

                else:
                    SIMILARS.append(s)
                    RES.update({str(ARTIST) + " - " + str(s): CURR_ID})
                    CURR_ID += 1;

            if (len(RES) != len(SONG_LIST)):
                print("Algo va mal...")

            return RES, CURR_ID

        def getID(s, DICT):
            id = DICT.get(s.longname, None)
            if (id is None):
                return (-1)
            else:
                return (id)

        ####################################################################################################################

        # Si no existe el diccionario, generar
        if os.path.isfile(pickleName + ".pkl"):
            print("Ya existe un diccionario previo...")
            return()

        if not os.path.isfile("datos/ALL_SONGS.pkl") and not os.path.isfile("datos/ALL_SONGS_UNIQUE.pkl"):
            print("No existen los pickles:\n\t · ALL_SONGS.pkl\n\t · ALL_SONGS_UNIQUE.pkl")

            ALL_SONGS = self.doQuery("select userid,timestamp,artname,traname from user_track")
            ALL_SONGS.artname = ALL_SONGS.artname.astype(str)
            ALL_SONGS.traname = ALL_SONGS.traname.astype(str)
            ALL_SONGS["longname"] = ALL_SONGS.artname + " - " + ALL_SONGS.traname
            ALL_SONGS.to_pickle("datos/ALL_SONGS.pkl");

            ALL_SONGS_UNIQUE = ALL_SONGS.drop_duplicates("longname")
            ALL_SONGS_UNIQUE = ALL_SONGS_UNIQUE.sort_values("longname")
            ALL_SONGS_UNIQUE.to_pickle("datos/ALL_SONGS_UNIQUE.pkl");

        else:
            ALL_SONGS_UNIQUE = pd.read_pickle("datos/ALL_SONGS_UNIQUE.pkl")
            ALL_SONGS = pd.read_pickle("datos/ALL_SONGS.pkl")

        ####################################################################################################################

        CURR_ID = 0

        RES = {}

        ARTISTS = ALL_SONGS_UNIQUE.groupby("artname")

        for i, g in ARTISTS:
            ARTIST = i
            # Si hay más de una canción para el artista, hay que ver si se repiten.
            if (len(g) > 1):
                # Comparar los nombres de las canciones buscando repetidos
                PARTIAL_RES, CURR_ID = checkDuplicates(CURR_ID, g.traname.values, ARTIST)
                RES.update(PARTIAL_RES)

            # Si solo hay una canción para el artista, se le asigna el número.
            else:
                RES.update({str(g.longname.values[0]): CURR_ID})
                CURR_ID += 1
            print(len(RES))

        #----------------------------------------------------------------------------------------
        # Crear diccionario con "Para cada string, el id al que se refiere" y el inverso
        #----------------------------------------------------------------------------------------

        self.save_pickle(RES, pickleName)

        # Crear el diccionario inverso
        '''ESTE DICCIONARIO TENDRÁ MENOS ITEMS DADO QUE SE ELIMINAN LAS CLAVES REPETIDAS'''

        INV_DICT = dict(zip(RES.values(), RES.keys()))
        self.save_pickle(INV_DICT, pickleName + "_INV")

        #----------------------------------------------------------------------------------------
        # Añadir al dataframe el ID y el número de reproducciones
        #----------------------------------------------------------------------------------------
        if not os.path.isfile("datos/ALL_SONGS_ID_COUNT.pkl"):

            # Añadir a cada cancion su id
            ID_COL = ALL_SONGS.apply(lambda row: getID(row, RES), axis=1)
            ALL_SONGS_ID = pd.concat([ALL_SONGS, ID_COL], axis=1)
            ALL_SONGS_ID.columns = ["user", "timestamp",'artname', 'traname', 'longname', 'ID']

            ID_COUNT = ALL_SONGS_ID.groupby(['ID']).size().reset_index(name='counts')
            RES = pd.merge(ALL_SONGS_ID, ID_COUNT, how='left', on=['ID'])
            RES.timestamp = pd.to_datetime(RES.timestamp, format="%Y-%m-%dT%H:%M:%SZ")
            RES.to_pickle("datos/ALL_SONGS_ID_COUNT.pkl");
        else:
            print("Fichero ya existente...")

        #----------------------------------------------------------------------------------------

    def generaFicherosUsuario(self, filename):
        '''
        A partir de los datos (YA FILTRADOS) , genera para cada usuario un fichero con las canciones escuchadas,
        en orden y representadas por su id
        :param: Diccionario con "Titulo"=>ID
        '''

        DATA = pd.read_pickle(filename)

        # Obtener los id de usuarios
        USER_LIST = DATA.groupby("user")

        #Para cada usuario
        for u,r in USER_LIST:

            #Imprimir ID
            self.printB(u)

            # Obtener todas las canciones del usuario
            USER_SONG_LIST = r

            # Ordenar por fecha
            USER_SONG_LIST.sort_values("timestamp", ascending=True, inplace=True)
            USER_SONG_LIST = USER_SONG_LIST.reset_index()

            # Lista con los id
            ID_LIST = USER_SONG_LIST.ID.values

            # Crear fichero con canciones para cada uno de los usuarios (Separado por ;)
            file = open("datos/usuarios/" + str(u) + ".txt", "w")
            file.write(";".join(str(x) for x in ID_LIST))
            file.close()

    def getCleanData(self,songMin=10,userMin=100,years=[2005,2006,2007,2008,2009],saveAs="MAIN_DATA"):

        if os.path.isfile("datos/ALL_SONGS_ID_COUNT.pkl"):
            ALL_SONGS_ID_COUNT = pd.read_pickle("datos/ALL_SONGS_ID_COUNT.pkl")
        else:
            print("No existen el pickle:\n\t · ALL_SONGS_ID_COUNT.pkl [generado en createDict()]")
            return ()

        INIT_LEN = len(ALL_SONGS_ID_COUNT)

        ALL_SONGS_ID_COUNT["year"] = ALL_SONGS_ID_COUNT.timestamp.dt.year.astype(int)

        print("-"*70)
        #--------------------------------------------------------------------------------------------------------------
        print(" · Eliminar datos que NO sean de los años " +(','.join(str(x) for x in years))+".")
        ALL_SONGS_ID_COUNT = ALL_SONGS_ID_COUNT.loc[ALL_SONGS_ID_COUNT.year.isin(years)]

        #--------------------------------------------------------------------------------------------------------------
        print(" · Eliminar canciones con menos de " + str(songMin) +" rerpoducciones.")

        #print(len(ALL_SONGS_ID_COUNT.loc[ALL_SONGS_ID_COUNT.counts < songMin]))
        ALL_SONGS_ID_COUNT = ALL_SONGS_ID_COUNT.loc[ALL_SONGS_ID_COUNT.counts >=songMin]

        #--------------------------------------------------------------------------------------------------------------
        print(" · Eliminar usuarios con menos de " + str(userMin) +" rerpoducciones.")

        USERS = ALL_SONGS_ID_COUNT.groupby(["user"])
        BAD_USERS=[]

        for u,d in USERS:
            if len(d)<userMin: BAD_USERS.append(u)


        ALL_SONGS_ID_COUNT = ALL_SONGS_ID_COUNT.loc[~ALL_SONGS_ID_COUNT.user.isin(BAD_USERS)]


        #--------------------------------------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------------------------------

        '''
        Es necesario añadir unos nuevos ID para facilitar el OneHot y eliminar los no utilizados
        '''

        #Se crea un diccionario con los IDs actuales (con saltos) y el correspondiente ID nuevo
        key = ALL_SONGS_ID_COUNT.ID.unique().tolist()
        value = range(0, len(key))
        dict = dict(zip(key, value))

        #Eliminar vieja columna y añadir la nueva
        ID_NEW = ALL_SONGS_ID_COUNT.ID.apply(lambda row: dict.get(row))
        ALL_SONGS_ID_COUNT = ALL_SONGS_ID_COUNT.drop(['ID'], axis=1)
        ALL_SONGS_ID_COUNT = pd.concat([ALL_SONGS_ID_COUNT, ID_NEW], axis=1)

        #--------------------------------------------------------------------------------------------------------------


        print("-"*70)

        print("Reproducciones iniciales: "+str(INIT_LEN))
        print("Reproducciones eliminadas: "+str(INIT_LEN-len(ALL_SONGS_ID_COUNT)))
        print("Usuarios eliminados: "+str(len(BAD_USERS))+"\n\t" + ','.join(BAD_USERS))

        ALL_SONGS_ID_COUNT.to_pickle(saveAs+".pkl");




        return ALL_SONGS_ID_COUNT

    ########################################################################
    # Métodos W2V
    ########################################################################

    def getSongDataArrays(self, train=75):
        '''
        Retorna un array de arrays [[cancion1,cancion2... ],[cancion1, cancion2...]...]
        :param: Array de arrays
        '''

        filename = "datos/CLEAN_DATA_[10,100].pkl"

        DATA = pd.read_pickle(filename)

        # Obtener los id de usuarios
        USER_LIST = DATA.groupby("user")

        RES_TRAIN=[]
        RES_REC=[]

        RES_USERS=[]

        #Para cada usuario
        for u,r in USER_LIST:

            # Obtener todas las canciones del usuario
            USER_SONG_LIST = r

            # Ordenar por fecha
            USER_SONG_LIST.sort_values("timestamp", ascending=True, inplace=True)
            USER_SONG_LIST = USER_SONG_LIST.reset_index()

            # Lista con los id
            ID_LIST = USER_SONG_LIST.ID.values.tolist()
            # Lista con las fechas

            # Separar para embeddigns y para recomendacion
            TRAIN_INDX = int(len(ID_LIST)*(train/100))

            TRAIN_LIST = ID_LIST[:TRAIN_INDX]
            REC_LIST = ID_LIST[TRAIN_INDX:]

            #Pasar a string las listas
            TRAIN_LIST = list(map(str, TRAIN_LIST))
            REC_LIST = list(map(str, REC_LIST))

            RES_TRAIN.extend(TRAIN_LIST)
            RES_REC.extend(REC_LIST)

            RES_USERS.append(u)

        self.save_pickle(RES_TRAIN,"datos/Extend/ARRAY_ARRAYS_TRAIN")
        self.save_pickle(RES_REC,"datos/Extend/ARRAY_ARRAYS_REC")

        return(RES_TRAIN,RES_REC,RES_USERS)

    def getUserDataArrays(self, train=75 , days=30, minSongs=5):
        '''
        Genera los datos para el método D2V.

        :param train: Porcentaje de datos para TRAIN
        :param days: Número de días para el perfil reciente
        :param minSongs: Número mínimo de canciones que ha de tener un usuario en su perfil reciente para no ser eliminado
        :return: Retorna datos de TRAIN COMPLETOS y RECIENTES y datos de DEV y TEST
        '''

        filename = "datos/CLEAN_DATA_[10,100].pkl"

        DATA = pd.read_pickle(filename)

        # Obtener los id de usuarios
        USER_LIST = DATA.groupby("user")

        RES_TRAIN_PRESENT=[]
        RES_TRAIN_COMPLETE=[]
        RES_REC=[]

        RES_USERS=[]

        #Para cada usuario
        for u,r in USER_LIST:

            # Obtener todas las canciones del usuario
            USER_SONG_LIST = r

            # Ordenar por fecha
            USER_SONG_LIST.sort_values("timestamp", ascending=True, inplace=True)
            USER_SONG_LIST = USER_SONG_LIST.reset_index()

            # Lista con los id y las fechas
            ID_LIST = USER_SONG_LIST.ID.values.tolist()
            DATES_LIST = USER_SONG_LIST.timestamp.values.tolist()
            DATES_LIST = np.array(list(map(lambda x: datetime.utcfromtimestamp(x / 1000000000), DATES_LIST)))

            # Separar TRAIN DEV y TEST
            TRAIN_INDX = int(len(ID_LIST)*(train/100))

            TRAIN_LIST = ID_LIST[:TRAIN_INDX]
            REC_LIST = ID_LIST[TRAIN_INDX:]
            DATES_TRAIN = DATES_LIST[:TRAIN_INDX]

            # Obtener X días antes
            PRESENT_START = DATES_TRAIN[-1] - timedelta(days=days)
            PAST_LENGTH = sum((DATES_TRAIN >= PRESENT_START) == False)
            PRESENT_ITEMS = TRAIN_LIST[PAST_LENGTH:]

            #Si el usuario no posee suficientes canciones en el perfil presente, no se añade
            if(len(PRESENT_ITEMS)<=minSongs):
                print("Skiping "+str(u))
                continue

            #Pasar a string las listas
            PRESENT_LIST = list(map(str,PRESENT_ITEMS))
            TRAIN_LIST = list(map(str, TRAIN_LIST))
            REC_LIST = list(map(str, REC_LIST))

            RES_TRAIN_COMPLETE.append(TRAIN_LIST)
            RES_TRAIN_PRESENT.append(PRESENT_LIST)
            RES_REC.append(REC_LIST)

            RES_USERS.append(u)

        print(len(RES_TRAIN_COMPLETE),len(RES_USERS))

        self.save_pickle(RES_TRAIN_COMPLETE,"datos/Append/ARRAY_ARRAYS_TRAIN_COMPLETE")
        self.save_pickle(RES_TRAIN_PRESENT,"datos/Append/ARRAY_ARRAYS_TRAIN_PRESENT")
        self.save_pickle(RES_REC,"datos/Append/ARRAY_ARRAYS_REC")


        return(RES_TRAIN_COMPLETE,RES_TRAIN_PRESENT,RES_REC,RES_USERS)

    def getSkipGramData(self, window=2):

        filename = "datos/ARRAY_ARRAYS_EXTEND"

        if os.path.isfile(filename+".pkl"):
            DATA = self.load_pickle(filename)
            DATA = self.save_pickle(filename)

        else:
            print("No existe el pickle, creando...")
            DATA = self.getDataArrays();

        DATA = list(map(int, DATA))
        LENGTH = max(DATA)+1

        RES_LEN = (len(DATA)-(window*2))*(window*2)

        batch_inputs = np.ndarray(shape=(RES_LEN), dtype=np.int32)
        batch_context = np.ndarray(shape=(RES_LEN, 1), dtype=np.int32)

        j=0

        for i in range(window,len(DATA)-window):
            for w in range(1,window+1):

                batch_inputs[j] = DATA[i]
                batch_context[j,0] = DATA[i-w]
                j+=1

                batch_inputs[j] = DATA[i]
                batch_context[j, 0] = DATA[i+w]
                j+=1

        return(batch_inputs,batch_context,LENGTH)

    ########################################################################
    # Estadísticas e info acerca de los datos
    ########################################################################

    def getPlaysByUser(self, times = list(range(10,110,10)) ):

        if os.path.isfile("datos/ALL_SONGS_ID_COUNT.pkl"):
            ALL_SONGS_ID_COUNT = pd.read_pickle("datos/ALL_SONGS_ID_COUNT.pkl")

        else:
            print("No existen el pickle:\n\t · ALL_SONGS_ID_COUNT.pkl [generado en createDict()]")
            return()

        #Obtener los id de usuarios
        USER_LIST = ALL_SONGS_ID_COUNT.groupby("user")


        print('\t'.join(str(x) for x in ["user",1]+times))

        for u, s in USER_LIST:
            COUNTS = [len(s)]
            for i in times: COUNTS.append(len(s.loc[s.counts>=i]))

            COUNTS = [u]+COUNTS
            print('\t'.join(str(x) for x in COUNTS))

    def getPlaysByYear(self):

        if os.path.isfile("datos/ALL_SONGS_ID_COUNT.pkl"):
            ALL_SONGS_ID_COUNT = pd.read_pickle("datos/ALL_SONGS_ID_COUNT.pkl")

        else:
            print("No existen el pickle:\n\t · ALL_SONGS_ID_COUNT.pkl [generado en createDict()]")
            return ()


        MIN_YEAR = min(ALL_SONGS_ID_COUNT.year)
        MAX_YEAR = max(ALL_SONGS_ID_COUNT.year)
        YEARS = (list(range(MIN_YEAR, MAX_YEAR + 1)))

        USER_LIST = ALL_SONGS_ID_COUNT.groupby("user")

        print('\t'.join(str(x) for x in ["user"] + YEARS))

        for u, r in USER_LIST:
            COUNTS = []
            for y in YEARS:
                COUNTS.append(len(r.loc[(r.year == y)]))

            COUNTS = [u] + COUNTS
            print('\t'.join(str(x) for x in COUNTS))

    def getUserDays(self):

        if os.path.isfile("datos/ALL_SONGS_ID_COUNT.pkl"):
            ALL_SONGS_ID_COUNT = pd.read_pickle("datos/ALL_SONGS_ID_COUNT.pkl")

        else:
            print("No existen el pickle:\n\t · ALL_SONGS_ID_COUNT.pkl [generado en createDict()]")
            return ()

        ALL_SONGS_ID_COUNT["date"] = ALL_SONGS_ID_COUNT.timestamp.dt.date

        USER_LIST = ALL_SONGS_ID_COUNT.groupby("user")
        print("user\tdays")
        for u, r in USER_LIST:
            days = r.groupby("date")
            print(str(u) + "\t" + str(len(days)))

    ########################################################################
    # Métodos auxiliares
    ########################################################################

    def printW(self,x):
        print(bcolors.WARNING + str(x) + bcolors.ENDC)

    def printB(self,x):
        print(bcolors.BOLD + str(x) + bcolors.ENDC)

    def printG(self,x):
        print(bcolors.OKGREEN + str(x) + bcolors.ENDC)

    def save_pickle(self,obj, name):
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, 2)

    def load_pickle(self,name):
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)
