import math as ma
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import RPi.GPIO as GPIO 
import time

def tiempos(alpha,r,n):
    #alpha = angulo transitorio
    # r = factor de reducción del vector
    # n = cantidad vectores intermedios 

    #voltajes fase neutro
    Van = ma.sin(ma.radians(alpha))
    Vbn = ma.sin(ma.radians(alpha -120))
    Vcn = ma.sin(ma.radians(alpha -240))
    #vector voltajes fase
    Vf = np.array([Van,Vbn,Vcn])
    #Proyección ortogonal sobre plano de 2 dimensiones (d,q)
    Vd,Vq = np.matmul(((2/3)*(np.array([[1,-0.5,-0.5],[0,(ma.sqrt(3)/2),-(ma.sqrt(3)/2)]]))),Vf)

    # Magnitud y ángulo de la proyección
    Vref = ma.sqrt(Vd**2 + Vq**2)*(ma.sqrt(2)/2)*0.5*r
    #print(f"Vref = {Vref}")
    betha = round(ma.atan(Vq/Vd),9) 

    if (abs(betha)>ma.radians(60)):
        betha = round(abs(betha)-ma.radians(30),9)
    #print(f"betha = {np.degrees(betha) }")

    #Tiempos
    Tz = (1/(60*n))*1000
    a = Vref/((ma.sqrt(2)/2))
    T1 = abs(Tz*round(a*(ma.sin(ma.radians(60) - betha) / ma.sin(ma.radians(60))),6)) 
    T2 = abs (Tz*round(a*(ma.sin(betha)/ma.sin(ma.radians(60))),6))
    T0 = Tz-T1-T2
    #T = np.transpose(np.array([T0/2,T1,T2,T0/2]))
    T =([T0/2,T1,T2,T0/2])
    return T

def secuencia_s1 (n):
    #Secuencia de bits para canal a,b y c en el sector 1 (0-60)
    sa ="0111"*n
    sb ="0011"*n
    sc ="0001"*n
    return sa,sb,sc
def secuencia_s2 (n):
    #Secuencia de bits para canal a,b y c en el sector 1 (60-120)
    sa ="1100"*n
    sb ="1110"*n
    sc ="1000"*n
    return sa,sb,sc
def secuencia_s3 (n):
    #Secuencia de bits para canal a,b y c en el sector 1 (120-180)
    sa ="0001"*n
    sb ="0111"*n
    sc ="0011"*n
    return sa,sb,sc
def secuencia_s4 (n):
    #Secuencia de bits para canal a,b y c en el sector 1 (180-240)
    sa ="1000"*n
    sb ="1100"*n
    sc ="1110"*n
    return sa,sb,sc
def secuencia_s5 (n):
    #Secuencia de bits para canal a,b y c en el sector 1 (240-300)
    sa ="0011"*n
    sb ="0001"*n
    sc ="0111"*n
    return sa,sb,sc
def secuencia_s6 (n):
    #Secuencia de bits para canal a,b y c en el sector 1 (300-360)
    sa ="1110"*n
    sb ="1000"*n
    sc ="1100"*n
    return sa,sb,sc

def secuencia_total(n):
    sa = "a"
    sb = "b"
    sc = "c"
    
    #Agregar los bits del sector 1
    a_aux,b_aux,c_aux = secuencia_s1(n)
    sa = sa+a_aux
    sb = sb+b_aux
    sc = sc+c_aux
    #Agregar los bits del sector 2
    a_aux,b_aux,c_aux = secuencia_s2(n)
    sa = sa+a_aux
    sb = sb+b_aux
    sc = sc+c_aux
    #Agregar los bits del sector 3
    a_aux,b_aux,c_aux = secuencia_s3(n)
    sa = sa+a_aux
    sb = sb+b_aux
    sc = sc+c_aux
    #Agregar los bits del sector 4
    a_aux,b_aux,c_aux = secuencia_s4(n)
    sa = sa+a_aux
    sb = sb+b_aux
    sc = sc+c_aux
    #Agregar los bits del sector 5
    a_aux,b_aux,c_aux = secuencia_s5(n)
    sa = sa+a_aux
    sb = sb+b_aux
    sc = sc+c_aux
    #Agregar los bits del sector 6
    a_aux,b_aux,c_aux = secuencia_s6(n)
    sa = sa+a_aux
    sb = sb+b_aux
    sc = sc+c_aux
    return sa,sb,sc


def datos (n):
    angulos = np.arange(0, 360, 360/(6*n))
    t = np.array(["t(ms)"]) 
    r=1
    for i in angulos:
        row =tiempos(i,r,n)
        t = np.vstack([t,row[0]])
        t = np.vstack([t,row[1]])
        t = np.vstack([t,row[2]])
        t = np.vstack([t,row[3]])
    sa,sb,sc = secuencia_total(n)
    dat = np.transpose(np.array([list(sa),list(sb),list(sc)]))
    dat = np.append(dat, t, axis = 1)
    return dat


def SalidaRasperry(data):
    d = np.shape(data)
    Nfilas = d[0]
    for i in range(1,Nfilas):
        DatoA = data[i][0]
        DatoB = data[i][1]
        DatoC = data[i][2]
        TimeOn = data[i][3]
        if  TimeOn >0:
            GPIO.output(PinA,DatoA)
            GPIO.output(PinB,DatoB)
            GPIO.output(PinC,DatoC)
            time.sleep(TimeOn)
  


############################################################################################### 

n=1
#Matriz de datos con:
#Estado transistor a, Estado transistor b,Estado transistor c, tiempo que dura ese estado (ms)
data = datos(n)
data.shape
#df_dir1 = pd.DataFrame(data)
#display (df_dir1)
PinA = 3
PinB = 5
PinC = 7
GPI0.setup(pinA,GPIO.OUT)
GPI0.setup(pinB,GPIO.OUT)
GPI0.setup(pinC,GPIO.OUT)
print(SalidaRasperry(data))  