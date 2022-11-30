import math as ma
import numpy as np
#import matplotlib.pyplot as plt
#import pandas as pd
import RPi.GPIO as GPIO 

import time

def tiempos(alpha,r,n,k):
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
    Tz = (1/(60*n*k))*1000
    a = Vref/((ma.sqrt(2)/2))
    T1 = abs(Tz*round(a*(ma.sin(ma.radians(60) - betha) / ma.sin(ma.radians(60))),6)) 
    T2 = abs (Tz*round(a*(ma.sin(betha)/ma.sin(ma.radians(60))),6))
    T0 = Tz-T1-T2
    #T = np.transpose(np.array([T0/2,T1,T2,T0/2]))
    T =([T0/2,T1,T2,T0/2])
    return T

def secuencia_s1 (n,k):
    #Secuencia de bits para canal a,b y c en el sector 1 (0-60)
    sa ="0111"*n*k
    sb ="0011"*n*k
    sc ="0001"*n*k
    return sa,sb,sc
def secuencia_s2 (n,k):
    #Secuencia de bits para canal a,b y c en el sector 1 (60-120)
    sa ="1100"*n*k
    sb ="1110"*n*k
    sc ="1000"*n*k
    return sa,sb,sc
def secuencia_s3 (n,k):
    #Secuencia de bits para canal a,b y c en el sector 1 (120-180)
    sa ="0001"*n*k
    sb ="0111"*n*k
    sc ="0011"*n*k
    return sa,sb,sc
def secuencia_s4 (n,k):
    #Secuencia de bits para canal a,b y c en el sector 1 (180-240)
    sa ="1000"*n*k
    sb ="1100"*n*k
    sc ="1110"*n*k
    return sa,sb,sc
def secuencia_s5 (n,k):
    #Secuencia de bits para canal a,b y c en el sector 1 (240-300)
    sa ="0011"*n*k
    sb ="0001"*n*k
    sc ="0111"*n*k
    return sa,sb,sc
def secuencia_s6 (n,k):
    #Secuencia de bits para canal a,b y c en el sector 1 (300-360)
    sa ="1110"*n*k
    sb ="1000"*n*k
    sc ="1100"*n*k
    return sa,sb,sc

def secuencia_total(n,k):
    sa = "0"
    sb = "0"
    sc = "0"
    
    #Agregar los bits del sector 1
    a_aux,b_aux,c_aux = secuencia_s1(n,k)
    sa = sa+a_aux
    sb = sb+b_aux
    sc = sc+c_aux
    #Agregar los bits del sector 2
    a_aux,b_aux,c_aux = secuencia_s2(n,k)
    sa = sa+a_aux
    sb = sb+b_aux
    sc = sc+c_aux
    #Agregar los bits del sector 3
    a_aux,b_aux,c_aux = secuencia_s3(n,k)
    sa = sa+a_aux
    sb = sb+b_aux
    sc = sc+c_aux
    #Agregar los bits del sector 4
    a_aux,b_aux,c_aux = secuencia_s4(n,k)
    sa = sa+a_aux
    sb = sb+b_aux
    sc = sc+c_aux
    #Agregar los bits del sector 5
    a_aux,b_aux,c_aux = secuencia_s5(n,k)
    sa = sa+a_aux
    sb = sb+b_aux
    sc = sc+c_aux
    #Agregar los bits del sector 6
    a_aux,b_aux,c_aux = secuencia_s6(n,k)
    sa = sa+a_aux
    sb = sb+b_aux
    sc = sc+c_aux
    return sa,sb,sc


def datos (n,k):
    angulos = np.arange(0, 360, 360/(6*n))
    t = np.array([0]) 
    r=1
    if k == 2:

        for i in angulos:
            row =tiempos(i,r*0.5,n)
            t = np.vstack([t,row[0]])
            t = np.vstack([t,row[1]])
            t = np.vstack([t,row[2]])
            t = np.vstack([t,row[3]])
            row =tiempos(i,r ,n)
            t = np.vstack([t,row[4]])
            t = np.vstack([t,row[5]])
            t = np.vstack([t,row[6]])
            t = np.vstack([t,row[7]])
    elif k == 3:
            row =tiempos(i,r*0.3,n)
            t = np.vstack([t,row[0]])
            t = np.vstack([t,row[1]])
            t = np.vstack([t,row[2]])
            t = np.vstack([t,row[3]])
            row =tiempos(i,r*0.6,n)
            t = np.vstack([t,row[4]])
            t = np.vstack([t,row[5]])
            t = np.vstack([t,row[6]])
            t = np.vstack([t,row[7]])
            row =tiempos(i,r ,n)
            t = np.vstack([t,row[8]])
            t = np.vstack([t,row[9]])
            t = np.vstack([t,row[10]])
            t = np.vstack([t,row[11]])
        
    sa,sb,sc = secuencia_total(n,k)
    dat = np.transpose(np.array([list(sa),list(sb),list(sc)]))
    dat = np.append(dat, t, axis = 1)
    return dat

def typhoon(data,n):
    sh = data.shape
    s = 0
    A = []
    B = []
    C = []
    for i in range(0,sh[0]-1):
        f = data[1+i]
        t = f[3]
        p = round(float(t)*(1000000)/(16.666666666666666))
        for i in range(p):
            A.append(f[0])
            B.append(f[1])
            C.append(f[2])
    return A,B,C

def SalidaRasperry(data):
    d = np.shape(data)
    Nfilas = d[0]
    k = 0
 
    while k  < 5000:
        for i in range(1,Nfilas-1):
           
            DatoA = int(data[i][0])
            DatoB = int(data[i][1])
            DatoC = int(data[i][2])
            TimeOn = float (data[i][3])
            
            if  TimeOn >0:
                print (DatoA)
                GPIO.output(PinA,DatoA)
                GPIO.output(PinB,DatoB)
                GPIO.output(PinC,DatoC)
                time.sleep(TimeOn/1000000)
        k += 1
     
    


############################################################################################### 

n=7
k = 2
#Matriz de datos con:
#Estado transistor a, Estado transistor b,Estado transistor c, tiempo que dura ese estado (ms)
data = datos(n,k)
data.shape
#df_dir1 = pd.DataFrame(data)
#display (df_dir1)
PinA = 3
PinB = 5
PinC = 7
GPIO.setmode(GPIO.BOARD)
GPIO.setup(PinA,GPIO.OUT)
GPIO.setup(PinB,GPIO.OUT)
GPIO.setup(PinC,GPIO.OUT)

data = datos(n,k)
#A,B,C = typhoon(data,n)
SalidaRasperry(data)



 
