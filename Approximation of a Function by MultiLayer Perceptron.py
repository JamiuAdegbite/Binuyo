"""

DESCRIPTION:
    
    This project describe a simple implementation of Multilayer Perceptron (MLP) Neural Network
    
    through manual coding (without the use of TensorFlow). A special kind of neurons was used 
    
    for layer 2 and 3 (binary neurons). The goal is to define and train an MLP dedicated to the
    
    approximation of a function F(x)
    
DATASET    

    The dataset was generated manually using Fourier Series Transformation
    
    y= F(x) = a0 + a1 cos(x) + b1 sin(x) + a2 cos(2x) + b2 sin(2x)
    
    After generating the input and output data, the function F(x) was assumed to be unknown
    
    
STEPS
    
    1) Define the function F(x) and simulate input and output data
    
    2) Simulate the training and test sets and assume the function F(x) to be unknown
        
    3) Define precisely an MLP with 4 layers and 30 neurons
    
        L1 = input layer 1 neuron
        
        L2 = layer of size 18 = 2 groups of 9 neurons Z1 Z2 ...Z9 and U1 U2 ...U9
        
        L3 = layer of size 10 = 10 neurons S1...S10
        
        L4 = output layer of size 1 = 1 neuron denoted R
        
    4) Implement a program to compute the states of L2 and the states of L3
    
        for any given input x
        
    5) Apply the function G(x) versus x for x= 0, 0.01,0.02, ...0.99, 1
        
    5) Construct the MLP to predict the results of the training and test sets and 
    
       estimate the mean square error by comparing it to the true output.
    

"""



""" Import the libraries """

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import random
from statistics import mean



""" Select Four (4) random numbers a0,a1,a2,b1,b2"""

np.random.seed(100)
a0= (np.random.uniform(low=-1, high=1, size=1)) 
np.random.seed(67)
a1= (np.random.uniform(low=-1, high=1, size=1))
np.random.seed(706)
b1= (np.random.uniform(low=-1, high=1, size=1))
np.random.seed(9)
a2= (np.random.uniform(low=-0.5, high=0.5, size=1))
np.random.seed(7)
b2= (np.random.uniform(low=-0.5, high=0.5, size=1))



""" Define the fourier function to generate the input and output dataset"""

def func1 (a):
    
      result_func = a0 + (a1 * np.cos((a))) + (b1 * np.sin((a))) + (a2*np.cos(2*(a)) + (b2 * np.sin(2*(a))))
      
      return result_func            


"""Generate random numbers as input to the function""" 

np.random.seed(56877)

xplot= np.sort(np.random.uniform(low=0, high=1, size=50))


"""Get the output data""" 

yplot=func1(xplot)


"""Plot the Function F(x)"""

plt.figure(figsize=(8, 6))
plt.plot(xplot, yplot, color='r', marker='o')
plt.ylabel('F(x)')
plt.xlabel('x')
plt.title('Plot of F(x) vs x')



"""Generate random numbers as input training set to the function""" 

np.random.seed(80008)
train_set= np.sort(np.random.uniform(low=0, high=1, size=1000))
np.around(train_set[:10],decimals=3)


"""Get the output data for the training set"""

train_outputs= func1(train_set)
np.around(train_outputs[:10],decimals=3)


"""Generate random numbers as input for the test set to the function""" 

random.seed(600)
test_set= np.sort(np.random.uniform(low=0, high=1, size=500))
np.around(test_set[:10],decimals=3)


"""Get the output data for the test set"""

test_outputs= func1(test_set)
np.around(test_outputs[:10],decimals=3)

x  = np.sort(train_set)

"""
    Define the MLP Architecture with 4 layers

"""

""" Compute the binary states of layer2 and layer 3"""


def states(x):
    def g(v): #Binary activation function
        if v>0:
            return 1
        if v<=0:
            return 0

    #layer2 defined as l2() with input x
    def l2(x):
        z=list()
        u=list()
         
        for i in range(9): 
            za= x - ((i+1)/10)
            zi = g(za)
            z.append(zi)        
        for i in range(9):
            ua = -x + ((i+1)/10)
            ui = g(ua)
            u.append(ui)        
        pl = (sum(z) + (9-sum (u)))/2 #define the position
    
        return z,u, pl
    
    z1,u1,pl=l2(x)
    
    def l3(x):
        z,u,pl  = l2(x)
        J = []
        for i  in range (10):
            if pl==i:
                a=1
            else:
                    a=0
            J.append(a)
        return J, pl

    j1,pl=l3(x)
    
    return z1,u1,j1,pl



"""
    Apply the program to the first 20 values x1 ... x20 in the training set and
    
    verify if the values taken by L2 and L3 correspond to the theoretically expected values

"""

b = []
for i in range(20):  
    d=states(x[i])
    b.append(d)       
b = pd.DataFrame(b, columns=['z(L2)','u(L2)','L3','pl'])
a=pd.DataFrame(x[:20])
c = pd.concat([b,a], axis=1)
c.columns=['z(Layer 2)','u(Layer 2)','Layer 3','Location', 'input values of x']
c


"""

 Estimate the weights (Learning the weights) for the last layer
 
"""


x1= pd.DataFrame(train_set)
y1= pd.DataFrame(train_outputs)     

d = pd.concat([x1,y1], axis=1)
d.columns=['x','y']
        
"""Learning of the weights for the last layer"""

w1=mean(d[(d.x > 0) & (d.x < 0.1)].y)
w2= mean(d[(d.x > 0.1) & (d.x < 0.2)].y)
w3= mean(d[(d.x > 0.2) & (d.x < 0.3)].y)
w4= mean(d[(d.x > 0.3) & (d.x < 0.4)].y)
w5= mean(d[(d.x > 0.4) & (d.x < 0.5)].y)
w6 = mean(d[(d.x > 0.5) & (d.x < 0.6)].y)
w7 = mean(d[(d.x > 0.6) & (d.x < 0.7)].y)
w8 = mean(d[(d.x > 0.7) & (d.x < 0.8)].y)
w9 = mean(d[(d.x > 0.8) & (d.x < 0.9)].y)
w10 = mean(d[(d.x > 0.9) & (d.x < 1.0)].y)


"""Weights"""

w1,w2,w3,w4,w5,w6,w7,w8,w9,w10


"""Define the last layer architecture"""

def l4(x):
        p,b = l3(x)             
        k = b/10
        k1 = (b+1)/10
        
        yv = w1*p[0] +w2*p[1] +w3*p[2] +w4*p[3] +w5*p[4] +w6*p[5] +w7*p[6] + w8*p[7] +w9*p[8] +w10*p[9]     
        return yv


"""
    Combine the layers to give a full description of the MLP Architecture
    
"""    

def y1(x):
    def g(v): #Binary activation function
        if v>0:
            return 1
        if v<=0:
            return 0

    #layer2 defined as l2() with input x
    def l2(x):
        z=list()
        u=list()
         
        for i in range(9): 
            za= x - ((i+1)/10)
            zi = g(za)
            z.append(zi)        
        for i in range(9):
            ua = -x + ((i+1)/10)
            ui = g(ua)
            u.append(ui)        
        pl = (sum(z) + (9-sum (u)))/2
    
        return z,u, pl
    
    z1,u1,pl=l2(x)
    
    def l3(x):
        z,u,pl  = l2(x)
        J = []
        for i  in range (10):
            if pl==i:
                a=1
            else:
                    a=0
            J.append(a)
        return J, pl

    j1,pl=l3(x)        

    x1= pd.DataFrame(train_set)
    y1= pd.DataFrame(train_outputs)     

    d = pd.concat([x1,y1], axis=1)
    d.columns=['x','y']
        
    w1=mean(d[(d.x > 0) & (d.x < 0.1)].y)
    w2= mean(d[(d.x > 0.1) & (d.x < 0.2)].y)
    w3= mean(d[(d.x > 0.2) & (d.x < 0.3)].y)
    w4= mean(d[(d.x > 0.3) & (d.x < 0.4)].y)
    w5= mean(d[(d.x > 0.4) & (d.x < 0.5)].y)
    w6 = mean(d[(d.x > 0.5) & (d.x < 0.6)].y)
    w7 = mean(d[(d.x > 0.6) & (d.x < 0.7)].y)
    w8 = mean(d[(d.x > 0.7) & (d.x < 0.8)].y)
    w9 = mean(d[(d.x > 0.8) & (d.x < 0.9)].y)
    w10 = mean(d[(d.x > 0.9) & (d.x < 1.0)].y)
    
    def l4(x):
        p,b = l3(x)             
        k = b/10
        k1 = (b+1)/10
        
        yv = w1*p[0] +w2*p[1] +w3*p[2] +w4*p[3] +w5*p[4] +w6*p[5] +w7*p[6]                + w8*p[7] +w9*p[8] +w10*p[9]     
        return yv
    yv = l4(x)
    
    return yv
  

"""
     Implement the program and compute G(x)

"""

bb= pd.DataFrame(np.arange(0.0, 1.0, 0.01).tolist()) #Define the input x
bb2= np.arange(0.0, 1.0, 0.01).tolist()
h1h=func1(bb) #Apply the MLP function to generate the output
gg=[]
for i in range(100): 
    bbmp= y1(bb2[i]) 
    gg.append(bbmp)    


 
"""
    Plot the function G(x) versus x for x= 0, 0.01,0.02, ...0.99, 1
    
    On the same graph plot the (assumed unknown) function F(x)
    
"""

plt.figure(figsize=(9, 7))
plt.plot(bb,gg,color='b', marker='o', label='G(x)')
plt.plot(bb,h1h,color='r', marker='o', label='F(x)')
plt.ylabel('G(x)/F(x)')
plt.xlabel('x')
plt.title('Plot of G(x)and F(x) vs x')
plt.legend(bbox_to_anchor=(0.85, 0.98), loc='upper left', borderaxespad=0.)


"""
     
            Compute the training set prediction
    
            Compute the test set prediction
            
            Compare the mean square error for both
    
"""




""" Training Set Prediction """


predtr=[]
for i in range(1000): 
    pp= y1(x[i]) 
    predtr.append(pp)
    
prederr=abs(predtr-train_outputs)



""" Mean Square Error for Training Set """

msetr= (sum((prederr*prederr)))/1000 
'{0:.6f}'.format(msetr)




"""Plot of Predicted Result and True Output (Training Set)"""

plt.figure(figsize=(8, 6))
plt.plot(train_set,train_outputs,color='b', marker='o', label='True Output')
plt.plot(train_set,predtr,color='r', marker='o', label='Predicted Output')
plt.ylabel('Result (y)')
plt.xlabel('Input value (x)')
plt.title('Plot of Predicted Result and True Output (Training Set)')
plt.legend(bbox_to_anchor=(0.6, 0.8), loc='upper left', borderaxespad=0.)



""" Test Set Prediction """

predtest1=[]
for i in range(500): 
    ppt= y1(test_set[i]) 
    predtest1.append(ppt)

predtest=abs(predtest1-test_outputs)




""" Mean Square Error for Test Set """

msetest= (sum((predtest*predtest)))/500

'{0:.6f}'.format(msetest)




"""Plot of Predicted Result and True Output (Test Set)"""

plt.figure(figsize=(8, 6))
plt.plot(test_set,test_outputs,color='b', marker='o', label='True Output')
plt.plot(test_set,predtest1,color='r', marker='o', label='Predicted Output')
plt.ylabel('Result (y)')
plt.xlabel('Input value (x)')
plt.title('Plot of Predicted Result and True Output (Test Set)')
plt.legend(bbox_to_anchor=(0.6, 0.8), loc='upper left', borderaxespad=0.)








