import cv2
import face_recognition as fr
import os
import numpy as np
from datetime import datetime
import time

# Ruta para acceder a la carpeta
path = os.path.join(os.path.abspath(os.getcwd()), 'Photos')
images = []
clases = []
lista = os.listdir(path)

# Carga de imágenes y nombres
for lis in lista:
    try:
        #Leer las imagenes de los rostros
        curImg = cv2.imread(f'{path}/{lis}')
        #Almacenamiento de las imagenes
        images.append(curImg)
        #Almacenamiento del nombre
        clases.append(os.path.splitext(lis)[0])
    except Exception as e:
        print(f"Error al leer la imagen {lis}[0]")
    
    print(clases)

# Codificación de los rostros
def findEncodings(images):
    encodeList = []
    
    #Iterar
    for img in images:
        #Correccion de color
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #codificamos la imagen del rostro
        encode = fr.face_encodings(img)
        #Verificar si reconoce rostro y Almacenar
        if encode:
            encodeList.append(encode[0])
    return encodeList

# Obtiene las codificaciones de todas las imágenes conocidas
encodeListKnown = findEncodings(images)

# Registro en CSV del nombre, fecha y hora
def horario(nombre):
    if not os.path.exists('Horarios.csv'):
        with open('Horarios.csv','w') as h:
            h.write('Nombre,Fecha,Hora\n')
    
    #Abrir el archivo en modo lectura y escritura
    with open('Horarios.csv', 'r+') as f:
        #Lee todas las lineas
        data = f.readlines()
        #Extrae los nombres registrados
        nombres = [line.split(',')[0] for line in data if line.strip()]
        #Condicion en caso de que no se encuentre el registro en nombres
        if nombre not in nombres:
            #Extraer la fecha
            fecha = datetime.now().strftime('%Y:%m:%d')
            #Obtiene la hora actual
            hora = datetime.now().strftime('%H:%M:%S')
            #Escribe el nombre y hora en el archivo csv
            f.write(f'{nombre},{fecha},{hora}\n')

# Iniciar webcam
cap = cv2.VideoCapture(0)

# Conjunto para almacenar los nombres ya reconocidos en esta sesión
reconocidos = set()
# Contador de fotogramas
frame_count = 0

while True:
    #Lee un frame de la camara
    success, frame = cap.read()
    #Condicion para terminar si no se realiza captura
    if not success:
        break
    
    #Incrementa el contador
    frame_count += 1

    # Procesar solo 1 de cada 3 frames
    if frame_count % 3 != 0:
        #Muestra el frame actual
        cv2.imshow('Reconocimiento Facial', frame)
        #Si se presiona ESC finaliza el programa
        if cv2.waitKey(1) & 0xFF == 27:
            break
        #Salta a la siguiente iteracion sin procesar el reconocimiento
        continue
    
    # Leer el tiempo inicial para medir la duración del procesamiento entre fotogramas
    start = time.time()  
    
    # Reducción de la imagen para mejorar el procedimiento
    imgS = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    #Conversion a RGB
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
    #Detectar los rostros
    facesloc = fr.face_locations(imgS)
    #Codifica los rostros encontrados
    facescod = fr.face_encodings(imgS, facesloc)
    
    #Condicion si no se detecta, muestra el frame y continúa
    if not facescod:
        cv2.imshow("Reconocimiento Facial", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue
    
    #Iterar cada
    for encodeFace, faceLoc in zip(facescod, facesloc):
        #Comparación de rostro en db con la cámara
        matches = fr.compare_faces(encodeListKnown, encodeFace)
        #Claculo de las similitudes
        faceDis = fr.face_distance(encodeListKnown, encodeFace)
        
        #Buscar el valor más bajo
        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            #Obtiene el nombre correspondiente
            nombre = clases[matchIndex].upper()

            # Si no ha sido reconocido antes, registrar
            if nombre not in reconocidos:
                #Registra la hora de entrada
                horario(nombre)
                reconocidos.add(nombre)
                
            #Extracción de las coordenadas
            y1, x2, y2, x1 = faceLoc
            # Escalar a frame original
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            
            #Dibujar  
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, nombre, (x1+6, y2-6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    #Finalización de la lectura del tiempo        
    end = time.time()
    print(f"⏱ Tiempo por frame: {end - start:.2f} segundos")
    
    #Mostrar frames
    cv2.imshow("Reconocimiento Facial", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
        break

#Libera la cámara
cap.release()
#Cierra todas las ventanas de openCV
cv2.destroyAllWindows()
