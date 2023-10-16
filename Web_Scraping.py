import subprocess
import os
import numpy as np
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
import re

# Carpeta de Dcoumentos txt
current_dir = os.path.dirname(__file__)
folder_path = "prueba_final\\prueba_final\\spiders\\"
carpeta_txt = os.path.join(current_dir, folder_path)

# Texto de los PDFs
texto_txt = ''
# Recabar nombres de txt
archivos_txt = [archivo for archivo in os.listdir(
    carpeta_txt) if archivo.endswith('.txt')]
# Texto sin modificaciones
texto_original = ''
# Texto tokenizado
texto_tokenizado = []
# Diccionario de terminos tokenizados
diccionario_terminos = {}
# Diccionario sin tokenizacion
diccionario_terminos_originales = {}


def ejecutarScript():
    # Ruta del script a ejecutar
    ruta_script = '.\\prueba_final\\prueba_final\\spiders\\quotes.py'

    # Obtiene la ruta completa al directorio del script actual
    directorio_actual = os.path.dirname(os.path.abspath(__file__))

    # Concatena la ruta del script con la ruta completa
    ruta_completa = os.path.join(directorio_actual, ruta_script)

    # Cambia el directorio actual al directorio del script
    os.chdir(os.path.dirname(ruta_completa))

    # Ejecuta el comando "scrapy crawl ejemplo_spider"
    subprocess.run(["scrapy", "crawl", "quotes"])
    print("Comando ejecutado")


def biblioteca_nltk():
    ''' Eliminacion de StopWords, Signos de puntuacion, Tokenizacion y Stemming '''
    # Descargar recursos de NLTK
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('snowball_data')
    print("Se descargaron los recursos de NLTK")

def eliminar_caracteres_especiales(cadena):
    # Utilizamos una expresión regular para encontrar todos los caracteres no alfanuméricos
    # y reemplazarlos con una cadena vacía.
    cadena_limpia = re.sub(r'[^a-zA-Z0-9\s]', '', cadena)
    return cadena_limpia

def leer_pdf(ruta_txt):
    """ Función para leer un archivo PDF y retornar su contenido de texto """
    print(ruta_txt)
    texto = ''
    with open(ruta_txt, 'r', encoding = 'utf-8', errors='ignore') as archivo:
        texto = archivo.read()
        texto = texto.lower()  # Convertir todo el texto a minúsculas
        texto = eliminar_caracteres_especiales(texto)
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
        texto_txt = leer_pdf(ruta_txt)  # Extraer el texto del PDF actual
        # Texto con terminos tokenizados
        terminos = tokenizar_stopwords(texto_txt)
        # Guardar texto sin modificaciones
        texto_original += texto_txt + '\n'
        # Guardar texto tokenizado, stemming, etc.
        texto_tokenizado += terminos
        
        print(terminos)     # Log en consola
        print(texto_tokenizado)     # Log en consola
    
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


def matriz_tf_idf():
    ''' Matriz tf-idf '''
    matriz_tf_idf = np.zeros((len(archivos_txt), len(diccionario_terminos_originales)))
    matriz_tf= np.zeros((len(archivos_txt), len(diccionario_terminos_originales)))
    
    # Llenado de la matriz con tf-idf
    for i, archivo in enumerate(archivos_txt):
        for j, termino_n in enumerate(diccionario_terminos_originales.keys()):  
            # Calcula tf              
            tf = diccionario_terminos_originales[termino_n].count(archivo)
            # Calcular tf-idf
            lista_archivos = diccionario_terminos_originales[termino_n]
            print(lista_archivos)
            nk = len(set(lista_archivos))
            print(nk)
            n = len(archivos_txt)
            print(n)
            idf = np.log10(n / nk)
            print(idf)
            matriz_tf[i, j] = tf
            matriz_tf_idf[i, j] = tf * idf
                
    # Guardar matriz tf-idf en un archivo csv
    idf = pd.DataFrame(matriz_tf_idf)
    idf.index = archivos_txt
    idf.to_csv('matriz_tf_idf.csv', sep='\t', header= list(diccionario_terminos_originales.keys()))
    
    # Guardar matriz tf-idf en un archivo csv
    tf = pd.DataFrame(matriz_tf)
    tf.index = archivos_txt
    tf.to_csv('matriz_tf.csv', sep='\t', header= list(diccionario_terminos_originales.keys()))
    
    # Guardar matriz tf-idf en un archivo de texto  
    # np.savetxt('matriz_tf_idf.txt', matriz_tf_idf, fmt='%d')
    
    # Guardar matriz tf en un archivo de texto
    # np.savetxt('matriz_tf.txt', matriz_tf, fmt='%d')
    print(matriz_tf)


# Ejecucion de Script Web Crawling
ejecutarScript()
print("Se ejecuto el script")

# cargar chingaderas
biblioteca_nltk()

# Obtener texto tokenizado
texto_tokenizado = obtener_texto_tokenizado()
print(texto_tokenizado)
print("Diccionario Creado")

input()

# Crear matriz tf-idf
matriz_tf_idf()
print("Finalizo todo")