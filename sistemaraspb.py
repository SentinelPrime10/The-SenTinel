import urllib
import numpy as np
import cv2

url='http://192.168.173.144:8080/shot.jpg'

# Llamada al metodo
fgbg = cv2.BackgroundSubtractorMOG2(history=800, varThreshold=1000)
 
while(1):
	# Leemos el siguiente frame
	#ret, frame = cap.read()
 
	# Si hemos llegado al final del video salimos
	#if not ret:
		#break
	imgResp=urllib.urlopen(url)
	imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
	frame=cv2.imdecode(imgNp,-1)
	
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	fgmask = fgbg.apply(frame)
 
	contornosimg = fgmask.copy()
 
	contornos, hierarchy = cv2.findContours(contornosimg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
	for c in contornos:
		if cv2.contourArea(c) < 500:
			continue
		(x, y, w, h) = cv2.boundingRect(c)
		#cv2.drawContours(frame, contornos, -1, (0, 0, 255), 2)
		cv2.rectangle(frame, (x, y),(x + w, y + h) , (0, 255, 0), 6)
	
	# Mostramos las capturas
	frame1=cv2.resize(frame,(640,370))
	fgmask1=cv2.resize(fgmask,(640,370))
	contornosimg1=cv2.resize(contornosimg,(640,370))
	cv2.imshow('Camara',frame1)
	cv2.imshow('Umbral',fgmask1)
	cv2.imshow('Contornos',contornosimg1)


#	cv2.imshow('Camara',frame)
#	cv2.imshow('Umbral',fgmask)
#	cv2.imshow('Contornos',contornosimg)
 
	# Sentencias para salir, pulsa 's' y sale
	k = cv2.waitKey(30) & 0xff
	if k == ord("s"):
		break
 
cap.release()
cv2.destroyAllWindows()
