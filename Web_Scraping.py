import subprocess
import os
import re
from tokenize import Ignore
from unidecode import unidecode
import numpy as np

''' Stemming '''
import nltk  # Importamos la bilbioteca nltk
# Importamos la funcion word_tokenize desde el modulo tokenize
from nltk.tokenize import word_tokenize
# Se importa la funcion stopwords desde el modulo corpus
from nltk.corpus import stopwords
# Algoritmos de stemming
from nltk.stem import SnowballStemmer# Importamos la funcion word_tokenize desde el modulo tokenize
from nltk.tokenize import word_tokenize
# Se importa la funcion stopwords desde el modulo corpus
from nltk.corpus import stopwords
# Algoritmos de stemming
from nltk.stem import SnowballStemmer
import string  # Importar el módulo de caracteres de puntuación
import pandas as pd

''' Tkinter '''
import tkinter as tk # Importar libreria tk
from tkinter import * # libreria para hacer interfaces
from tkinter import filedialog # Sirve para leer archivos del administrador
import shutil #libreria para manipular archivos
import os #libreria para manipular directorios del S.O
from tkinter import messagebox
from tkinter import ttk


# Carpeta de Dcoumentos txt
carpeta_txt = 'gatos\\gatos\\spiders\\'
# Nombre de archivos
archivos_txt = []
# Texto de los PDFs
texto_txt = ''
# Texto sin modificaciones
texto_original = ''
# Texto tokenizado
texto_tokenizado = []
# Diccionario de terminos tokenizados
diccionario_terminos = {}
# Diccionario sin tokenizacion
diccionario_terminos_originales = {}
# Matriz TD_IDF
matriz_tf_idf = []

# Ventana ------------------------------------------------------------------------
''' Configuracion inicial de botones y ventana '''
raiz = Tk() #se crea el objeto de la ventana

# Funciones ----------------------------------------------------------------------
def on_cerrar_ventana():
    # Cerrar Tkinter
    raiz.destroy()


def abrir_txt_con_aplicacion_predeterminada(nombre_txt):
    ''' Abrir txt con app predeterminada'''
    try:
        subprocess.Popen([nombre_txt], shell=True)
    except Exception as e:
        print(f"No se pudo abrir el archivo text: {e}")


def ejecutarScript():
    # Ruta del script a ejecutar
    ruta_script = 'gatos\\gatos\\spiders\\spider1.py'

    # Obtiene la ruta completa al directorio del script actual
    directorio_actual = os.path.dirname(os.path.abspath(__file__))

    # Obtén el directorio actual antes de cambiarlo
    directorio_actual = os.getcwd()

    # Concatena la ruta del script con la ruta completa
    ruta_completa = os.path.join(directorio_actual, ruta_script)

    # Cambia el directorio actual al directorio del script
    os.chdir(os.path.dirname(ruta_completa))

    # Ejecuta el comando "scrapy crawl ejemplo_spider"
    subprocess.run(["scrapy", "crawl", "spider1"])
    print("Comando ejecutado")

    # Regresa al directorio anterior
    os.chdir(directorio_actual)


def biblioteca_nltk():
    ''' Eliminacion de StopWords, Signos de puntuacion, Tokenizacion y Stemming '''
    # Descargar recursos de NLTK
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('snowball_data')
    print("Se descargaron los recursos de NLTK")


def quitar_acentos(palabra):
    # Utiliza unidecode para quitar acentos y caracteres diacríticos
    palabra_sin_acentos = unidecode(palabra)
    return palabra_sin_acentos


def quitar_acentos_de_texto(texto):
    palabras = texto.split()  # Divide el texto en palabras
    palabras_sin_acentos = [quitar_acentos(palabra) for palabra in palabras]
    texto_sin_acentos = ' '.join(palabras_sin_acentos)
    return texto_sin_acentos


def leer_txt(ruta_txt):
    """ Función para leer un archivo PDF y retornar su contenido de texto """
    print(ruta_txt)
    texto = ''
    with open(ruta_txt, 'r', encoding = 'utf-8', errors='ignore') as archivo:
        texto = archivo.read()
        texto = texto.lower()  # Convertir todo el texto a minúsculas
        texto = ''.join(c for c in texto if c not in string.punctuation or c in '()')
        texto = re.sub(r'[^\w\s]', '', texto)  # Eliminar símbolos de puntuación y otros caracteres no alfanuméricos
        texto = quitar_acentos_de_texto(texto)  # Eliminar acentos
        print(texto)     # Log en consola
        return texto


def tokenizar_stopwords(texto):
    """ Eliminar Signos de Puntuacion """
    texto = texto.translate(str.maketrans('', '', string.punctuation))  # Eliminar signos de puntuación
    
    ''' Texto Tokenizado y sin StopWords '''
    # Tokenización
    tokens = word_tokenize(texto)
    
    ''' Texto sin StopWords '''
    # Eliminación de stopwords en español
    sw = set(stopwords.words('spanish'))
    # Guardamos palabra por palabra de los tokens anteriores verifica la version en minusculas
    stop_words = [word for word in tokens if word.lower() not in sw]
    
    ''' Stemming '''
    # Stemming en español
    stemmer = SnowballStemmer('spanish')
    # Realiza stemming palabra por palabra de
    stemming_stopWords_tokens = [stemmer.stem(
        word) for word in stop_words]
    return stemming_stopWords_tokens


def tokenizado_sinsteamming(texto):
    """ Eliminar Signos de Puntuacion """
    texto = texto.translate(str.maketrans(
        '', '', string.punctuation))  # Eliminar signos de puntuación
    ''' Texto Tokenizado y sin StopWords '''
    # Tokenización
    tokens = word_tokenize(texto)
    # Eliminación de stopwords en español
    sw = set(stopwords.words('spanish'))
    # Guardamos palabra por palabra de los tokens anteriores verifica la version en minusculas
    tokenizado = [word for word in tokens if word.lower() not in sw]
    return tokenizado


def obtener_texto_tokenizado():
    # Ruta de pdf's
    global carpeta_txt
    # Texto de los PDFs
    global texto_txt
    # Recabar nombres de txt
    global archivos_txt
    # Texto sin modificaciones
    global texto_original
    # Texto tokenizado
    global texto_tokenizado
    # Diccionario de terminos tokenizados
    global diccionario_terminos
    # Diccionario sin tokenizacion
    global diccionario_terminos_originales
    # Contador de terminos
    contador = 0
    # 2do contador
    contador2 = 0
    # 3er contador
    contador3 = 0
    # 4to contador
    contador4 = 0

    # Ciclo de lectura por archivo
    for archivo in archivos_txt:
        ruta_txt = os.path.join(carpeta_txt, archivo)
        texto_txt = leer_txt(ruta_txt)  # Extraer el texto del txt actual
        # Texto con terminos tokenizados
        terminos = tokenizar_stopwords(texto_txt)
        # Guardar texto sin modificaciones
        texto_original += texto_txt + '\n'
        # Guardar texto tokenizado, stemming, etc.
        texto_tokenizado += terminos
    
        ''' Guardar en diccionario tokenizado '''
        for i in range(len(terminos)):
            if terminos[i] in diccionario_terminos:
                # Si el término ya está en el diccionario, agrega el nombre del archivo
                diccionario_terminos[terminos[i]].append(archivo)
            else:
                # Si el término no está en el diccionario, crea una nueva entrada con el archivo
                diccionario_terminos[terminos[i]] = [archivo]
                contador+=1
            print(contador)     # Log en consola

        terminos_sin_stem = tokenizado_sinsteamming(texto_txt)

        ''' Guardar en diccionario sin stem '''            
        for j in range(len(terminos)):
            if terminos_sin_stem[j] in diccionario_terminos_originales:
                diccionario_terminos_originales[terminos_sin_stem[j]].append(archivo)
            else:
                diccionario_terminos_originales[terminos_sin_stem[j]] = [archivo]
                contador2+=1
            print(contador2)     # Log en consola
    print("Se guardo todo")     # Log en consola
    
    ''' Guardar texto tokenizado y sin StopWords '''
    texto_procesado = (f"'{word}'\n" if word ==
                        'endDoc$' else f"'{word}', " for word in texto_tokenizado)
    with open('swandSStep3-4.txt', 'w', encoding='utf-8') as archivo_tokenizado:
        archivo_tokenizado.writelines(texto_procesado)
    print("Se guardo swandSStep3-4.txt")


    ''' Diccionarios '''
    ''' txt de Diccionario(Con stem) '''
    diccionario1 = 'DiccionarioStep2.txt'   # Guardar texto extraido
    with open(diccionario1, 'w') as archivo:
        for palabra, documentos in diccionario_terminos.items():
            documentos_str = ', '.join(documentos)
            archivo.write(f'{palabra}: {documentos_str}\n')
            contador3+=1
            print(contador3)
    print("Se guardo DiccionarioStep2.txt")

    ''' txt de Diccionario (sin stem) '''
    diccionario2 = 'Diccionario_sinstem_Step2.txt'   # Guardar texto extraido
    with open(diccionario2, 'w') as archivo:
        for palabra, documentos in diccionario_terminos_originales.items():
            documentos_str = ', '.join(documentos)
            archivo.write(f'{palabra}: {documentos_str}\n')
            contador4+=1
            print(contador4)
    print("finalizo todo")     # Log en consola
    return texto_tokenizado


def generar_matriz_tf_idf():
    ''' Matriz tf-idf '''
    global diccionario_terminos
    global matriz_tf_idf
    global archivos_txt

    matriz_tf_idf = np.zeros((len(archivos_txt) + 1, len(diccionario_terminos)))
    matriz_tf= np.zeros((len(archivos_txt), len(diccionario_terminos)))
    
    contador = 0

    # Llenado de la matriz con tf-idf
    for i, archivo in enumerate(archivos_txt):
        for j, termino_n in enumerate(diccionario_terminos.keys()):  
            # Calcula tf              
            tf = diccionario_terminos[termino_n].count(archivo)
            # Calcular tf-idf
            lista_archivos = diccionario_terminos[termino_n]
            # print(lista_archivos)
            nk = len(set(lista_archivos))
            # print(nk)
            n = len(archivos_txt)
            # print(n)
            idf = np.log10(n / nk)
            # print(idf)
            matriz_tf[i, j] = tf
            matriz_tf_idf[i, j] = tf * idf
            
            contador+=1
            print(contador)

                
    # Guardar matriz tf-idf en un archivo csv
    tf = pd.DataFrame(matriz_tf)
    tf.index = archivos_txt
    tf.to_csv('matriz_tf.csv', sep='\t', header= list(diccionario_terminos.keys()))


def agregar_consulta_matriz_td_idf(terminos_consulta):
    global matriz_tf_idf
    global diccionario_terminos
    global archivos_txt

    # Vecto de consulta
    vector_consulta = np.zeros((1,len(diccionario_terminos)))
    # Llenado del vector con tf-idf
    for i, termino_diccionario in enumerate(diccionario_terminos.keys()):  
        for termino_consulta in terminos_consulta:
            # Si el termino esta en el diccionario   
            if termino_consulta == termino_diccionario:
                # Contar el número de veces que esta el termino en la consulta
                tf = terminos_consulta.count(termino_consulta)
                # Calcular tf-idf
                lista_archivos = diccionario_terminos[termino_consulta]
                # Número de archivos en los que aparece el término
                nk = len(set(lista_archivos))
                # Número total de archivos
                n = len(archivos_txt)
                # Calcular idf
                idf = np.log10(n / nk)
                # Colocamos el valor en el vector de consulta
                vector_consulta[0, i] = tf * idf

                print(tf)
                print(n)
                print(nk)
                print(idf)
                # input("Press Enter to continue...")

    # Agregar vector de consulta a la matriz tf-idf
    matriz_tf_idf[-1] = vector_consulta
    # matriz_tf_idf = np.vstack([matriz_tf_idf, vector_consulta])

    # Guardar matriz tf-idf en un archivo csv
    idf = pd.DataFrame(matriz_tf_idf)
    fila_nombre = archivos_txt.copy()
    fila_nombre.append('consulta')
    idf.index = fila_nombre
    idf.to_csv('matriz_tf_idf.csv', sep='\t', header= list(diccionario_terminos.keys()))
    print("Vector_Consulta:")
    print(vector_consulta)


def caluclar_distancia_euclidiana(punto1, punto2):
    # Calcular la distancia euclidiana entre dos vectores
    return np.linalg.norm(punto1 - punto2)


def calcular_centroides(grupos):
    # Calcular los nuevos centroides de cada grupo
    centroides = []

    # Recorrer cada grupo
    for grupo in grupos:

        # Si el grupo no está vacío
        if len(grupo) > 0:

            # Calcular el centroide como el promedio
            # de los atributos de cada objeto en el grupo
            centroide = np.mean(grupo, axis=0)
            # Agregar el centroide a la lista de centroides
            centroides.append(centroide)
        else:
            # Si el grupo está vacío, el centroide es un punto aleatorio
            # Se vuelve a calcular el centroide
            centroide = np.random.rand(grupo.shape[1])
            
            # Agregar el centroide a la lista de centroides como un arreglo nuevo aleatorio
            centroides.append(centroide)
    return centroides


def k_means(matriz, k, iteraciones):
    # Número de objetos(Filas)
    n = matriz.shape[0]
    print("número de objetos:")
    print(n)

    # Número de atributos(Columnas)
    atributos = matriz.shape[1]
    # print("no. atributos:")
    # print(atributos)

    # Inicializar centroides de manera aleatoria
    # Generar k números aleatorios entre 0 y 1
    # Columnas = atributos
    # Filas = k (No. de clusters)
    centroides = np.random.uniform(0, 3, size=(k, atributos))
    # print("centroides:")
    # print(centroides)
    # input()

    iteracionesNo = 0

    # Número de iteraciones a realizar K-Means
    for _ in range(iteraciones):

        # Contador
        print("iteracionesNo:")
        print(iteracionesNo)
        iteracionesNo += 1

        # Inicializar k grupos vacíos (Clusters)
        grupos = [[] for _ in range(k)]

        # Asignar cada objeto al centroide más cercano
        # Recorriendo cada objeto(n) de la matriz
        for i in range(n):

            print("Número objeto:")
            print(i)          

            # Calcular la distancia euclidiana entre el objeto y cada centroide
            distancias = [caluclar_distancia_euclidiana(matriz[i], centroide) for centroide in centroides]

            # print("Matriz[i]:")
            # print(matriz[i])

            # print("distancias:")
            # print(distancias)

            # Obtener el índice del centroide más cercano al objeto(n)
            grupo_asignado = distancias.index(min(distancias))

            # print("grupo_asignado:")
            # print(grupo_asignado)

            # Agregar el objeto al grupo correspondiente
            grupos[grupo_asignado].append(i)  # Almacenar el índice del documento

        # Calcular nuevos centroides
        nuevos_centroides = calcular_centroides([matriz[grupo] for grupo in grupos])

        # print("centroides:")
        # print(centroides)

        # Asignamos los nuevos centroides obtenidos(re-calculo de los centroides)
        centroides = nuevos_centroides

        # print("nuevos_centroides:")
        # print(nuevos_centroides)
    # Retornar grupos
    return grupos


# Boton 1
# Función para cambiar el color del botón al pasar el mouse sobre él
def cambiar_color1(event):
    boton_DicMos.configure(bg="#202123", fg="white")
# Función para restaurar el color original del botón al salir del mouse
def restaurar_color1(event):
    boton_DicMos.configure(bg="#444654", fg="white")


# Boton 2
# Función para cambiar el color del botón al pasar el mouse sobre él
def cambiar_color2(event):
    boton_StemStop.configure(bg="#202123", fg="white")
# Función para restaurar el color original del botón al salir del mouse
def restaurar_color2(event):
    boton_StemStop.configure(bg="#444654", fg="white")


# Boton 3
# Función para cambiar el color del botón al pasar el mouse sobre él
def cambiar_color3(event):
    boton_MatrizTxt.configure(bg="#202123", fg="white")
# Función para restaurar el color original del botón al salir del mouse
def restaurar_color3(event):
    boton_MatrizTxt.configure(bg="#444654", fg="white")

# Boton 4
# Función para cambiar el color del botón al pasar el mouse sobre él
def cambiar_color4(event):
    boton_guardar.configure(bg="#202123", fg="white")
# Función para restaurar el color original del botón al salir del mouse
def restaurar_color4(event):
    boton_guardar.configure(bg="#444654", fg="white")


# Crear una fuente personalizada con un tamaño grande
fuente_grande = ("Arial", 17)


# Botón 1 Mostar Diccionario -----------------------------------------------------------------------------
def mostar_diccionario():
    print("Mostró el diccionario")
    nombre_txt = "DiccionarioStep2.txt"
    # Abre txt
    abrir_txt_con_aplicacion_predeterminada(nombre_txt)

# Crear un botón y asociarle la función mostrar diccionario
boton_DicMos = Button(raiz, text="Mostrar Diccionario", command = mostar_diccionario, bg="#444654", fg="white", relief="flat", padx=10, pady=5)

# se enpaqueta el botón y se colóca en la ventana. 
boton_DicMos.pack()
# posiciona el bóton en los pixeles de la pantallas
boton_DicMos.place(x=100, y=60)


# Botón 2 Mostar Texto Stem y sin Stop -----------------------------------------------------------------------------
def mostar_Stem_Stop():
    print("Mostró el Texto steam y sin StopWords")
    nombre_txt = "swandSStep3-4.txt"
    # Abre txt
    abrir_txt_con_aplicacion_predeterminada(nombre_txt)


# Crear un botón y asociarle la función mostar_Stem_Stop)
boton_StemStop = Button(raiz, text="Mostrar Texto Procesado", command = mostar_Stem_Stop, bg="#444654", fg="white", relief="flat", padx=10, pady=5)
# se enpaqueta el botón y se colóca en la ventana. 
boton_StemStop.pack()
# posiciona el bóton en los pixeles de la pantallas
boton_StemStop.place(x=250, y=60) 


# Botón 3 Mostar Matriz -----------------------------------------------------------------------------
def btnMatrizTxt():
    print("Mostró el Texto steam y sin StopWords")
    nombre_txt = "matriz_tf_idf.csv"
    # Abre txt
    abrir_txt_con_aplicacion_predeterminada(nombre_txt)


# Crear un botón y asociarle la función btnMatrizTxt
boton_MatrizTxt = Button(raiz, text="Mostrar Matriz TF_IDF", command = btnMatrizTxt, bg="#444654", fg="white", relief="flat", padx=10, pady=5)
# se enpaqueta el botón y se colóca en la ventana. 
boton_MatrizTxt.pack()
# posiciona el bóton en los pixeles de la pantallas
boton_MatrizTxt.place(x=428, y=60)   

# Botón Consulta -----------------------------------------------------------------------------
# Función para obtener el texto del Entry al presionar el botón
def obtener_texto():
    ''' Obtener consulta '''
    consulta = entry.get()
    k = int(entry2.get())
    iteraciones = int(entry3.get())

    print("Obtener consulta")
    print(consulta)
    print("Obtener k")
    print(k)

    ''' Aplicar Stopwods y stemming '''
    consulta = tokenizar_stopwords(consulta)
    print("Aplicar Stopwods y stemming con exito")
    
    ''' Agregar consulta en Matriz tf-idf '''
    agregar_consulta_matriz_td_idf(consulta)
    
    ''' Mostrar resultado '''
    # Ejecutar algoritmo k-means al conjunto de documentos
    resultados = k_means(matriz_tf_idf, k, iteraciones)

    print("Resultados:")
    print(resultados)

    # Índice del grupo que contiene la consulta
    indice_grupo_consulta = None

    # Indice de la consulta
    indice_consulta = matriz_tf_idf.shape[0] - 1
    print("indice_consulta:")
    print(indice_consulta)

    for i, grupo in enumerate(resultados):
        if indice_consulta in grupo:
            indice_grupo_consulta = i
            break

    # Filtrar los documentos del clúster de la consulta
    documentos_en_cluster_consulta = [i + 1 for i in resultados[indice_grupo_consulta]]

    # Eliminamos el ultimo elemento de la lista
    documentos_en_cluster_consulta.pop()

    print("documentos_en_cluster_consulta:")
    print(documentos_en_cluster_consulta)

    # Mostramos los documentos en terminal y los guardamos en una lista
    print("Documentos en el clúster de la consulta:")

    with open('resultado.txt', 'w') as archivo:
        archivo.write(f"Consulta: {consulta}" + "\n")
        archivo.write(f"K: {k}" + "\n")
        archivo.write(f"Iteraciones: {iteraciones}" + "\n")
        archivo.write(f"Resultados: " + "\n")
        for indice in documentos_en_cluster_consulta:
            print(f"Doc{indice}.txt")
            archivo.write(f"Doc{indice}.txt" + "\n")

    # Nombre del txt
    nombre_txt = "resultado.txt"
    # Abre txt
    abrir_txt_con_aplicacion_predeterminada(nombre_txt)

    # Mostrar los documentos del clúster de la consulta
    label_resultado.configure(text=(f"Mostrando resultados en txt..."))

# Crear un Entry para ingresar texto
entry = Entry(raiz,  width=27, bg="#444654", fg="white", relief="flat", font=fuente_grande)
entry.pack(pady=100, ipady=100)
entry.place(x=50, y=245)

# Crear un Entry para ingresar texto
entry2 = Entry(raiz,  width=15, bg="#444654", fg="white", relief="flat", font=fuente_grande)
entry2.pack(pady=100, ipady=100)
entry2.place(x=420, y=245)

# Crear un Entry para ingresar texto
entry3 = Entry(raiz,  width=15, bg="#444654", fg="white", relief="flat", font=fuente_grande)
entry3.pack(pady=100, ipady=100)
entry3.place(x=420, y=200)

# Crear un botón para guardar el texto
boton_guardar = Button(raiz, text= "Enviar consulta " , command=obtener_texto, bg="#444654", fg="white", relief="flat", padx=10, pady=5)
boton_guardar.pack(pady=20)
boton_guardar.place(x=640, y=243) 

# Crear un Label para mostrar el resultado
fuente_resultado = ("Arial", 13)
resultado = ""
label_resultado = tk.Label(raiz, text=resultado, bg="#343641", fg="white", font=fuente_resultado)
label_resultado.pack()
label_resultado.place(x=50, y=170)

# Info
fuente_pequeña = ("Arial", 8)
label_facts = tk.Label(raiz, text="Free Research. RI Sistem must produce accurate information, powered by team 9. RI Sistem October 20 Version", bg="#343641", fg="#C5C5D2", font=fuente_pequeña)
label_facts.pack()
label_facts.place(x=59, y=278) 

# Ventana principal ------------------------------------------------------------------

# Titulo de la ventana
raiz.title("RI Sistem") 
 #Tamaño de la ventana
raiz.geometry("800x300")
# Desactivar el redimensionamiento de la ventana
raiz.resizable(False, False)
#Color de la ventana
raiz.config(bg="#343641")  # Establecer el color de fondo en el formato hexadecimal
# Configurar el manejador de eventos para el cierre de la ventana
raiz.protocol("WM_DELETE_WINDOW", on_cerrar_ventana)


# Eventos boton 1
boton_DicMos.bind("<Enter>", cambiar_color1)
boton_DicMos.bind("<Leave>", restaurar_color1)

# Eventos boton 2
boton_StemStop.bind("<Enter>", cambiar_color2)
boton_StemStop.bind("<Leave>", restaurar_color2)

# Eventos boton 3
boton_MatrizTxt.bind("<Enter>", cambiar_color3)
boton_MatrizTxt.bind("<Leave>", restaurar_color3)

# Evento Boton consulta
boton_guardar.bind("<Enter>", cambiar_color4)
boton_guardar.bind("<Leave>", restaurar_color4)


# Ejecucion de Script Web Crawling
# ejecutarScript()
# print("Se ejecuto el script")
# input()

# Recabar nombres de txt obtenidos
archivos_txt = [archivo for archivo in os.listdir(
    carpeta_txt) if archivo.endswith('.txt')]

# Cargar bibliotecas de NLTK
biblioteca_nltk()
print("Bibliotecas cargadas")

# Obtener texto tokenizado
texto_tokenizado = obtener_texto_tokenizado()
print("Texto Tokenizado con exito...")


# Crear matriz tf-idf
generar_matriz_tf_idf()
print("Puedes escribir tú consulta...")

''' Metodo del codo '''
# Metodo del codo
# Rango de valores de k que deseas evaluar
k_range = range(1, 100)

# Lista para almacenar los valores de WCSS
wcss = []

# Realiza K-Means para cada valor de k
for k in k_range:
    grupos = k_means(matriz_tf_idf, k, 100)
    # Calcula WCSS para este valor de k
    wcss_sum = 0
    for i in range(k):
        centroid = np.mean(matriz_tf_idf[grupos[i]], axis=0)
        distances = [caluclar_distancia_euclidiana(p, centroid) for p in matriz_tf_idf[grupos[i]]]
        wcss_sum += sum([d ** 2 for d in distances])
    wcss.append(wcss_sum)

# Mostrar gráfica de codo
import matplotlib.pyplot as plt
plt.plot(k_range, wcss, marker='o', label='Tachas', color='red')
plt.xlabel('Cluster')
plt.ylabel('WCSS')
plt.title('Método del codo')
plt.show()

''' Mostrar Ventana '''

#Se visualiza la ventena (Siempre debe de ir al final)
raiz.mainloop()