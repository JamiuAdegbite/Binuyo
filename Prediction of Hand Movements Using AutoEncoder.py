
"""
DESCRIPTION
This project involves the implementation of autoencoder for automatic classification tasks.

DATASET
A Vicon motion capture camera system was used to record users performing 3 
hand postures with markers attached to a left-handed gloves. Markers with a rigid pattern 
on the glove was used to establish a local co-ordinate system. 
The goal of the classification task is to correctly predict the hand movements.

The dataset for this project can be found here "https://archive.ics.uci.edu/ml/datasets/MoCap+Hand+Postures#"

STEPS
1) Define an MLP Architecture involving 3 layers  

   a. PCA Analysis was used for investigating the number of hidden layers.

   b.  Monitoring of average cross entropy, updating of the weights and thresholds through gradient descent
       and monitoring the loss functon 


2) Performance analysis was done via Confusion matrix and ploting of performance for both the training and testing sets
   More than 80% accuracy for the classification method was obtained.

3) Investigation of the effectts of performance on several parameters like 
   a. Change of Batch Size, 
   b. Initialization of weights and biases
   c. Gradient Descent Step Size, 
   d. dimension of number of neurons in the hidden layers

4) Analysis of the global states of the hidden layer  

   a. Fixing of the number of hidden layer based on best performance
   b. Performing PCA analysis
   c. Investigation and display of profile activities for the hidden neuron to determing which neuron has
      the best differentiation between the classes.

"""




""" This code import the necessary tools for the work """

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits import mplot3d
from sklearn.utils import shuffle
import random



"Set the seed for the random number generator"
random.seed(500)

"Import and shuffle the data to achieve randomness"
data1 = shuffle(pd.read_csv("Summarized_Data.csv").iloc[:,1:17]).reset_index(drop=True)

"Display first 5 rows of the data"
data1.head(5)         


"Get the output Class from the new dataset"
y_output = data1['Class']     

"Get the input (explanatory data"               
x_input = data1.iloc[:,0:15]


"""
Divide the input data into classes
and display number of classes in the dataset used
"""

D1= data1[data1['Class']==1].iloc[:,0:15]
D2= data1[data1['Class']==2].iloc[:,0:15]
D3= data1[data1['Class']==3].iloc[:,0:15]
print ("Class 1 : "+ str(D1.shape[0]),"Class 2: "+ str(D2.shape[0]), "Class 3: "+str(D3.shape[0]) )


"""Convert the output data into binary code"""

yy={}
for i in range(10002):
    
    if y_output[i]==1:   
      yy[i]=[0,0,1]
      
    elif y_output[i]==2:
        yy[i]=[0,1,0] 
    else: 
        yy[i]=[1,0,0] 
yyy=pd.DataFrame.from_dict(data=yy,orient='index')


""" Selecting 2 tentative sizes (h) for the hidden layer by PCA Analysis """


"Scaling and Spliting the data into training and test sets (80%:20%)"
scaler=preprocessing.StandardScaler()
scaled_data=pd.DataFrame(scaler.fit_transform(x_input))
data_train0,data_test0,y_train0,y_test0= train_test_split(scaled_data,y_output,test_size=0.2, random_state=100)
data_train,data_test,y_train,y_test= train_test_split(scaled_data,yyy,test_size=0.2, random_state=100)


"Retrieving the classes in the training and test sets"

"Training set: Class 1"
D1tr=y_train0[y_train0[:,]==1]

"Training set: Class 2"
D2tr=y_train0[y_train0[:,]==2] 

"Training set: Class 3"
D3tr=y_train0[y_train0[:,]==3] 

"Test set: Class 1"
D1te=y_test0[y_test0[:,]==1]  

"Test set: Class 2"
D2te=y_test0[y_test0[:,]==2]   

"Test set: Class 3"
D3te=y_test0[y_test0[:,]==3]    


"Displaying the number of classes in the training and test sets"

print ("Class 1 (training set): "+ str(D1tr.shape[0]),"Class 2 (training set): "+ str(D2tr.shape[0]), "Class 3 (training set): "+str(D3tr.shape[0]) )
print ("Class 1 (test set): "+ str(D1te.shape[0]),"Class 2 (test set): "+ str(D2te.shape[0]), "Class 3 (test set): "+str(D3te.shape[0]) )

"Display first 5 rows of the training data" 
data_train.head(5)


"Scaling the training data"
data_train_std=StandardScaler().fit_transform(data_train) 


"Finding the Covariance Matrix for the training set and displaying the first 4 rows of data"
cov_mat1=np.corrcoef(np.transpose(data_train_std))
pd.DataFrame(cov_mat1).head(4) 


"Finding the eigen values and eigen vectors and displaying the results"
eig_vals1,eig_vecs1=np.linalg.eig(cov_mat1) #Get the Eigen values and Eigen vectors
print((eig_vals1)) #print the eigen values
plt.plot(eig_vals1)


"Plots the Eigen values"
plt.figure(figsize=(7, 5))
plt.figure(1)
L1=sorted(eig_vals1,reverse=True, )
plt.plot(L1,  marker='o', label='Eigen Values', color='r')
plt.ylabel('Eigen Values')
plt.xlabel('Cases')
plt.title('Plot of Eigen Values')
plt.legend()


"Plot the cummulative sum of the explained variance"
dgvv1=eig_vals1
da1vv1=[]
for i in range(15):    
    if i ==0:
        Ri = dgvv1[i]
    else:
        Ri+=dgvv1[i]        
    da1vv1.append(Ri)
    
plt.plot(da1vv1/(sum(dgvv1)),  marker='o', label='Lj', color='r')
plt.ylabel('Lj')
plt.xlabel('j')
plt.title('Plot of Lj vs j')
plt.legend()


L1=sorted(eig_vals1,reverse=True)

"""
The smallest number h95 that explaine about 95% variance in our data
This number is estaimated from the plot above and code below
h95 is thus set as the minimum number of neurons in the hidden layer for learning
"""

compare1=sum(L1)*0.95
s1=0
count1=0
for i in L1:
    count1=count1+1
    s1=s1+i
   # print(i)
    if s1>compare1:
       # print(count1)
        break
    
h95=count1
print("The smallest number h95 is: " + str(h95)) #Estimated to be 11


"""
This code perform a PCA analysis on the input training set and plot it in 3 dimension
to see if the projection of the classes are separated or not

"""
#PCA Analysis
pca0 = PCA(n_components=3)
pca0.fit(data_train)
print(pca0.explained_variance_) 
# Store results of PCA in a data frame
result1=pd.DataFrame(pca0.transform(D1), columns=['PCA%i' % i for i in range(3)], index=D1.index)
result2=pd.DataFrame(pca0.transform(D2), columns=['PCA%i' % i for i in range(3)], index=D2.index)
result3=pd.DataFrame(pca0.transform(D3), columns=['PCA%i' % i for i in range(3)], index=D3.index)
print(result2)


"Plots of the Principal components in 3 Dimensions"
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1,1,1, projection='3d') 
ax.scatter(result1['PCA0'], result1['PCA1'], result1['PCA2'],s=8,marker='*', color='g', label='Class 1')
ax.scatter(result2['PCA0'], result2['PCA1'], result2['PCA2'],s=8,marker='o', color = 'b', label='Class 2')
ax.scatter(result3['PCA0'], result3['PCA1'], result3['PCA2'],s=8, marker='+', color='r', label='Class 3')
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.legend (loc='best')
ax.set_title("PCA on the input vector training set")


"""
To estimate a larger plausible value of h95,
PCA Analysis was conducted for each of the classes

"""

"Class 1"


"Standardized and print the first 4 cases of class 1"
D1_std=StandardScaler().fit_transform(D1)
pd.DataFrame(D1_std).head(4) #standardized data for class 1


"Get the covariance matrix of the data"
cov_D1=np.corrcoef(np.transpose(D1_std))
pd.DataFrame(cov_D1)#Covariance matrix for class 1


"Get the Eigen Values and Eigen Vectors for Class 1"
eig_valsD1,eig_vecsD1=np.linalg.eig(cov_D1)
print(eig_valsD1)

pd.DataFrame(eig_vecsD1)

"Plot the Eigen Values"
plt.figure(figsize=(7, 5))
plt.figure(1)
L1=sorted(eig_valsD1,reverse=True, )
plt.plot(L1,  marker='o', label='Eigen Values', color='r')
plt.ylabel('Eigen Values')
plt.xlabel('')
plt.title('Plot of Eigen Values for Class 1')
plt.legend()

"Plot the cummulative sum of the explained variance considering Class 1"
dgvv=eig_valsD1
da1vv=[]
for i in range(15):    
    if i ==0:
        Ri = dgvv[i]
    else:
        Ri+=dgvv[i]        
    da1vv.append(Ri)
    
plt.plot(da1vv/(sum(dgvv)),  marker='o', label='Lj', color='r')
plt.ylabel('Lj')
plt.xlabel('j')
plt.title('Plot of Lj vs j for Class 1')
plt.legend()


"""
The smallest number h95 that explaine about 95% variance in Class 1
This number is estaimated from the plot above and code below
h95 is thus set as U1 for Class 1
"""
LD1=sorted(eig_valsD1,reverse=True)
compareD1=sum(LD1)*0.95
sD1=0
countD1=0
LD1=sorted(eig_valsD1,reverse=True, )
for i in LD1:
    countD1=countD1+1
    sD1=sD1+i
    #print(i)
    if sD1>compareD1:
        #print(countD1)
        break
    
U1=countD1
print("The smallest number is: " + str(U1))


"""
This code perform a PCA analysis on Class 1 input training set and plot in 3 dimension
"""

pcac1 = PCA(n_components=3)
pcac1.fit(D1)
# Store results of PCA in a data frame
resultD1=pd.DataFrame(pcac1.transform(D1), columns=['PCA%i' % i for i in range(3)], index=D1.index)
print(resultD1)


"Projection of  Class 1 in 3 dimension"
fig2 = plt.figure(figsize=(8, 6))
ax2 = fig2.add_subplot(1,1,1, projection='3d') 
ax2.scatter(resultD1['PCA0'], resultD1['PCA1'], resultD1['PCA2'],s=8,marker='o', color='g', label='Class 1')
ax2.set_xlabel("PC1")
ax2.set_ylabel("PC2")
ax2.set_zlabel("PC3")
ax2.legend (loc='center left')
ax2.set_title("PCA on the input vector training set for Class 1")


"Class 2"

"Standardized and print the first 4 cases of class 2"
D2_std=StandardScaler().fit_transform(D2) 
pd.DataFrame(D2_std).head(4)  #standardized data for class 2

"Get the covariance matrix of the data"
cov_D2=np.corrcoef(np.transpose(D2_std))
pd.DataFrame(cov_D2).head(4)#Covariance matrix for class 2


"Get the Eigen Values and Eigen Vectors for Class 2"
eig_valsD2,eig_vecsD2=np.linalg.eig(cov_D2)
print(eig_valsD2)


"Plot the Eigen Values"
plt.figure(figsize=(7, 5))
plt.figure(1)
L1=sorted(eig_valsD2,reverse=True, )
plt.plot(L1,  marker='o', label='Eigen Values', color='r')
plt.ylabel('Eigen Values')
plt.xlabel('Cases')
plt.title('Plot of Eigen Values for Class 2')
plt.legend()


"Plot the cummulative sum of the explained variance considering Class 2"
dgvv2=eig_valsD2
da1vv2=[]
for i in range(15):    
    if i ==0:
        Ri = dgvv2[i]
    else:
        Ri+=dgvv2[i]        
    da1vv2.append(Ri)
    
plt.plot(da1vv2/(sum(dgvv2)),  marker='o', label='Lj', color='r')
plt.ylabel('Lj')
plt.xlabel('j')
plt.title('Plot of Lj vs j for Class 2')
plt.legend()

LD2=sorted(eig_valsD2,reverse=True)


"""
The smallest number h95 that explaine about 95% variance in Class 2
This number is estimated from the plot above and code below
h95 is thus set as U2 for Class 2
"""
compareD2=sum(LD2)*0.95
sD2=0
countD2=0
for i in LD2:
    countD2=countD2+1
    sD2=sD2+i
    #print(i)
    if sD2>compareD2:
     #   print(countD2)
        break
    
U2=countD2
U2
print("The smallest number is: " + str(U2))


"""
This code perform a PCA analysis on Class 2 input training set and plot in 3 dimension
"""

pcac3 = PCA(n_components=3)
pcac3.fit(D2)
# Store results of PCA in a data frame
resultD2=pd.DataFrame(pcac3.transform(D2), columns=['PCA%i' % i for i in range(3)], index=D2.index)
print(resultD2)


"Projection of  Class 2 in 3 dimension"

fig3 = plt.figure(figsize=(8, 6))
ax3 = fig3.add_subplot(1,1,1, projection='3d') 
ax3.scatter(resultD2['PCA0'], resultD2['PCA1'], resultD2['PCA2'],s=8,marker='*', color='b', label='Class 2')
ax3.set_xlabel("PC1")
ax3.set_ylabel("PC2")
ax3.set_zlabel("PC3")
ax3.legend (loc='center left')
ax3.set_title("PCA on the input vector training set for Class 2")


"Class 3"

"Standardized and print the first 4 cases of class 3"

D3_std=StandardScaler().fit_transform(D3)
pd.DataFrame(D3_std).head(4)  


"Get the covariance matrix of the data"
cov_D3=np.corrcoef(np.transpose(D3_std))
pd.DataFrame(cov_D3).head(4)#Covariance matrix for class 3


"Get the Eigen Values and Eigen Vectors for Class 3"
eig_valsD3,eig_vecsD3=np.linalg.eig(cov_D3)
print(eig_valsD3)


"Plot the Eigen Values"
plt.figure(figsize=(7, 5))
plt.figure(1)
L1=sorted(eig_valsD3,reverse=True, )
plt.plot(L1,  marker='o', label='Eigen Values', color='r')
plt.ylabel('Eigen Values')
plt.xlabel('Cases')
plt.title('Plot of Eigen Values for Class 3')
plt.legend()


"Plot the cummulative sum of the explained variance considering Class 2"

dgvv3=eig_valsD3
da1vv3=[]
for i in range(15):    
    if i ==0:
        Ri = dgvv3[i]
    else:
        Ri+=dgvv3[i]        
    da1vv3.append(Ri)

plt.plot(da1vv3/(sum(dgvv3)),  marker='o', label='Lj', color='r')
plt.ylabel('Lj')
plt.xlabel('j')
plt.title('Plot of Lj vs j for Class 3')
plt.legend()

LD3=sorted(eig_valsD3,reverse=True)

"""
The smallest number h95 that explaine about 95% variance in Class 3
This number is estimated from the plot above and code below
h95 is thus set as U3 for Class 3
"""

compareD3=sum(LD3)*0.95
sD3=0
countD3=0
for i in LD3:
    countD3=countD3+1
    sD3=sD3+i
   # print(i)
    if sD3>compareD3:
       # print(countD3)
        break
    
U3=countD3
U3
print("The smallest number is: " + str(U3))


"""
This code perform a PCA analysis on Class 3 input training set and plot in 3 dimension
"""
pcac4 = PCA(n_components=3)
pcac4.fit(D3)
# Store results of PCA in a data frame
resultD3=pd.DataFrame(pcac4.transform(D3), columns=['PCA%i' % i for i in range(3)], index=D3.index)
print(resultD3)


"Projection of  Class 3 in 3 dimension"

fig4 = plt.figure(figsize=(8, 6))
ax4 = fig4.add_subplot(1,1,1, projection='3d') 
ax4.scatter(resultD3['PCA0'], resultD3['PCA1'], resultD3['PCA2'],s=8,marker='+', color='r', label='Class 3')
ax4.set_xlabel("PC1")
ax4.set_ylabel("PC2")
ax4.set_zlabel("PC3")
ax4.legend (loc='center left')
ax4.set_title("PCA on the input vector training set for Class 3")



""" The larger plausible value of h95 is thus calculated as follows"""

hL= U1+U2+U3 #Estimated to be 34

"""Print out the value of hL"""
print("Plausible value of hL is: " + str(hL)) 


"""PART 2, 3 & 4 : 
    Defining MLP Architecture,
    Automatic Training and 
    Performance Analysis
"""


def fun (b,w,l, h ):
    
    """
    This function define the MLP Architecture, the learning, performance, and analysis
    
    It takes as an input, the batch-size, the seed number for definiting different initialization of weights and biases,
    the initial learning rate, and the number of neurons in the hidden layer
    
    the output produces training accuracy, test set accuracy,loss function for both the training and the test set
    weights and biases for the hidden layer,loss function, average batch loss , || W(n+1)- W(n)|| / ||Wn||, average gradient
    """
       
    import math
    training_epochs = 1000 #training epoch
    batch_size= b #batch size
    display_step=1 #step_size

    n_hidden = h #number of hidden layer =h95 first
    n_input = 15  #number of inputs equal number of features
    n_classes = 3  #number of classes number of output
    d=math.sqrt(h*15 + h + h *3 + 3) #square root of dimension

    #learning
    global1_step=tf.Variable(0,trainable=False)
    initial_learning_rate=l
    learning_rate=tf.compat.v1.train.exponential_decay(initial_learning_rate,    global_step=global1_step, decay_steps=training_epochs, decay_rate=0.9)

    add_global=global1_step.assign_add(1)

    X=tf.compat.v1.placeholder("float",[None,n_input])
    Y=tf.compat.v1.placeholder("float",[None,n_classes])
    
    random.seed(w)
    
    """initialization of the weights and bisases"""
    weights={
            'h': tf.Variable(tf.random_normal([n_input,n_hidden])),
            'out':tf.Variable(tf.random_normal([n_hidden,n_classes]))
            }

    biases={
            'b':tf.Variable(tf.random_normal([n_hidden])),
            'out':tf.Variable(tf.random_normal([n_classes]))
            }

    """Define the MLP architecture with one hidden layer"""
    
    def MLP(x):
        layer_1=tf.add(tf.matmul(x,weights['h']), biases['b'])
        layer_1=tf.nn.relu(layer_1)
        out_layer=tf.matmul(layer_1, weights['out'])+biases['out']

        return out_layer
    
    """ Construct the MLP model"""
    logits=MLP(X)

    #define loss and optimizer

    """Define the Loss function and optimizer"""
    
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=logits))
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
    
    correct_prediction=tf.equal(tf.argmax(logits,1),tf.argmax(Y,1))
    
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    
    """ Define the Confusion Matrix"""
    confusion_matrix=tf.math.confusion_matrix(tf.argmax(logits,1),tf.argmax(Y,1))
    
   
    """Initializing the variables"""
    
    init=tf.global_variables_initializer()
   
    with tf.Session() as sess:
        sess.run(init)
        ini_acu=sess.run(accuracy,feed_dict={X:data_train,Y:y_train})
        train_accu1=[]
        test_accu1=[]
        train_loss=[]
        test_loss=[]
        L_R1=[]
        LOSS1=[]
        W_n1=[]
        relW=[]
        G_n1=[]
        G_ave1=[]
        BACRE1=[]
        ACRE1=[]
        #Training cycle
        for epoch in range(training_epochs):
            
            avg_cost=0
            total_batch=int(data_train.shape[0]/batch_size)
            store=np.append(np.reshape(sess.run(weights['h']),(1,n_hidden*n_input)),np.reshape(sess.run(weights['out']),(1,n_classes*n_hidden)))
            store=np.append(store,np.reshape(sess.run(biases['b']),(1,n_hidden)))
            store=np.append(store,np.reshape(sess.run(biases['out']),(1,n_classes)))
            for i in range(total_batch):
                       
                step,rate=sess.run([add_global,learning_rate])
                L_R1.append(rate)
                # print(rate)
                random.seed(1000)
                randidx=np.random.randint(8001,size=batch_size)
                batch_xs=data_train.iloc[randidx,:]
                batch_ys=y_train.iloc[randidx,:]
           
                sess.run(optimizer,feed_dict={X:batch_xs,Y:batch_ys})
                c = sess.run(loss,feed_dict={X:batch_xs,Y:batch_ys})
                BACRE1.append(sess.run(accuracy,feed_dict={X:batch_xs,Y:batch_ys}))
                #print(c)
                LOSS1.append(c)
                W1=np.reshape(sess.run(weights['h']),(1,n_hidden*n_input))
                W2=np.reshape(sess.run(weights['out']),(1,n_classes*n_hidden))
                W3=np.reshape(sess.run(biases['b']),(1,n_hidden))
                W4=np.reshape(sess.run(biases['out']),(1,n_classes))
                W=np.concatenate((W1,W2,W3,W4),axis=1)
                #W=np.append(W1,W2)
                #W=np.append(W,W3)
                #W=np.append(W,W4)
                WW=LA.norm(W-store)
                relW.append(WW/(LA.norm(store)))
                W_n1.append(WW)
                # print(WW)
                G_n1.append(WW/rate)
                G_ave1.append(WW/(rate*d))
                store=W
                avg_cost+=c/total_batch
                if step%80==0:
                    train=sess.run(accuracy,feed_dict={X:data_train,Y:y_train})
                    train_accu1.append(train)
                    test=sess.run(accuracy,feed_dict={X:data_test,Y:y_test})
                    test_accu1.append(test)
                    train_loss.append(sess.run(loss,feed_dict={X:data_train,Y:y_train}))
                    test_loss.append(sess.run(loss,feed_dict={X:data_test,Y:y_test}))
        ax=sess.run(confusion_matrix,feed_dict={X:data_test,Y:y_test})
        ay=sess.run(confusion_matrix,feed_dict={X:data_train,Y:y_train})      

    return train_accu1, test_accu1,train_loss, test_loss,ax,ay,LOSS1,W_n1,G_ave1,W1,W3,BACRE1


""" Case 1: 
    Assuming the number of hidden neuron is (h95) = 11 as estimated before
    run the MLP for number  of hidden neurons in the layer as 11
"""

train_accu1, test_accu1,train_loss, test_loss,axcom,aycom,LOSS1,W_n1,G_ave1,W11,W31,BACRE1 = fun (1000, 1000, 0.01, 11) 


""" Get the Confusion Matrix for Performance Evaluation (Training Set) """

vv = np.array([sum(aycom[0,:]),sum(aycom[1,:]),sum(aycom[2,:])])
np.around((aycom.T/vv).T, decimals=3)


""" Get the Confusion Matrix for Performance Evaluation (Test Set) """
vvt = np.array([sum(axcom[0,:]),sum(axcom[1,:]),sum(axcom[2,:])])
np.around((axcom.T/vvt).T, decimals=3)


""" Plot the Loss Function for the Batch Size """
plt.figure(figsize=(7, 5))
plt.plot(range(8000),BACRE1,  color='b', label='Loss Function')
plt.ylabel('Batchsize Loss Function')
plt.xlabel('Cases')
plt.title('Plot of Batch-Size Loss  Function')
plt.legend()


""" Plot the Loss Function """
plt.figure(figsize=(7, 5))
plt.plot(range(8000),LOSS1,  color='b', label='Loss Function')
plt.ylabel('Loss Function')
plt.xlabel('Cases')
plt.title('Plot of Loss  Function')
plt.legend() 


""" Plot the change of norm of weights """
plt.figure(figsize=(7, 5))
plt.plot(range(8000),W_n1,  color='b',  label='|| W (n+1) - W(n) || / || W(n) || ')
plt.ylabel('|| W (n+1) - W (n) || / || W (n) || ')
plt.xlabel('Cases')
plt.title('The change of the norm of weights ||W (n+1) - W(n) || / || W(n) || ')
plt.legend() 


""" Plot the average gradients """
plt.figure(figsize=(7, 5))
plt.plot(range(8000),G_ave1,  color='b',  label=' || G_n || / d ')
plt.ylabel('|| G_n || / d ')
plt.xlabel('Cases')
plt.title(' || G_n || / d ')
plt.legend()


""" Plot the Performance of both training and test sets together"""
fig = plt.figure(figsize=(7,5))
ax = plt.subplot(111)
ax.plot( train_accu1, color='b',label='performance on training set')
ax.plot( test_accu1, color='r',label='performance on test set')
plt.ylabel('performance')
plt.title('performance on test set and training set')
ax.legend(loc='center')
plt.show()


""" Plot the loss function for both the training and test sets together"""
fig = plt.figure(figsize=(7,5))
ax = plt.subplot(111)
ax.plot( train_loss, color='b',label='Loss function on training set')
ax.plot( test_loss, color='r',label='Loss function on test set')
plt.ylabel('Loss Function')
plt.title('Loss Function on test set and training set ')
ax.legend(loc='center')
plt.show()


""" Case 2: 
    Assuming the number of hidden neuron is (h95) = 34 as estimated before
    run the MLP for number  of hidden neurons in the layer as 34
"""

train_accu12, test_accu12,train_loss2, test_loss2,ax2com,ay2com,LOSS12,W_n12,G_ave12,W34,W35,BACRE2 = fun (1000, 1000, 0.01, 34) 

""" Get the Confusion Matrix for Performance Evaluation (Training Set) """

#Confusion Matrix for the Training Set
vvtr = np.array([sum(ay2com[0,:]),sum(ay2com[1,:]),sum(ay2com[2,:])])
np.around((ay2com.T/vvtr).T, decimals=3)

""" Get the Confusion Matrix for Performance Evaluation (Test Set) """

vvte1 = np.array([sum(ax2com[0,:]),sum(ax2com[1,:]),sum(ax2com[2,:])])
np.around((ax2com.T/vvte1).T, decimals=3)


""" Plot the Loss Function for the Batch Size """
plt.figure(figsize=(7, 5))
plt.plot(range(8000),BACRE2,  color='b', label='Loss Function')
plt.ylabel('Batchsize Loss Function')
plt.xlabel('Cases')
plt.title('Plot of Batch-Size Loss  Function')
plt.legend()


""" Plot the Loss Function """
plt.figure(figsize=(7, 5))
plt.plot(range(8000),LOSS12,  color='b',  label='Loss Function')
plt.ylabel('Loss Function')
plt.xlabel('Cases')
plt.title('Plot of Loss  Function')
plt.legend()

""" Plot the change of norm of weights """
plt.figure(figsize=(7, 5))
plt.plot(range(8000),W_n12,  color='b', label='|| W (n+1) - W(n) || / || W(n) || ')
plt.ylabel('|| W (n+1) - W (n) || / || W (n) || ')
plt.xlabel('Cases')
plt.title('The change of the norm of weights ||W (n+1) - W(n) || / || W(n) || ')
plt.legend() 

""" Plot the average gradients """
plt.figure(figsize=(7, 5))
plt.plot(range(8000),G_ave12,  color='b', label=' || G_n || / d ')
plt.ylabel('|| G_n || / d ')
plt.xlabel('Cases')
plt.title(' || G_n || / d ')
plt.legend()


""" Plot the Performance of both training and test sets together"""

fig = plt.figure(figsize=(7,5))
ax = plt.subplot(111)
ax.plot( train_accu12, color='b',label='performance on training set')
ax.plot( test_accu12, color='r',label='performance on test set')
plt.ylabel('performance')
plt.title('performance on test set and training set')
ax.legend(loc='center')
plt.show()



""" Plot the loss function for both training and test sets together"""
fig = plt.figure(figsize=(7,5))
ax = plt.subplot(111)
ax.plot( train_loss2, color='b',label='Loss function  on training set')
ax.plot( test_loss2, color='r',label='Loss function on test set')
plt.ylabel('Loss Function')
plt.title('Loss function on test set and training set')
ax.legend(loc='center')
plt.show()



"""PART 5:
    Investigating the impact of various learning options on the MLP
    
"""

""" Effect of Change in Batch Size of Performance"""

for b in [2000,1000, 500,  200]:    
    train_accu1b, test_accu1b,train_lossb, test_lossb,axb,ayb,LOSS1b,W_n1b,G_ave1b,Wb1,Wb2,BACREb1 = fun (b, 1000, 0.01, 34) 
    plt.plot(train_accu1b, label='batch_size = %s '%b)
plt.ylabel('performance')
plt.title('Effect of Batch Size on performance of training set')
plt.legend(loc='center right')
plt.show()   


""" Effect of Initialization on Performance"""

for b in [2000,1000, 500,  200]:    
    train_accu1b, test_accu1b,train_lossb, test_lossb,axb,ayb,LOSS1b,W_n1b,G_ave1b, Wi1, Wi2,BACREi = fun (1000, b, 0.01, 34)    
    plt.plot(train_accu1b, label='ini = %s '%b)
plt.ylabel('performance')
plt.title('Effect of initialization on performance of training set')
plt.legend(loc='center right')
plt.show()   



""" Effect of gradient descent step size on Performance"""

for b in [0.05,0.01, 0.005, 0.001]:    
    train_accu1b, test_accu1b,train_lossb, test_lossb,axb,ayb,LOSS1b,W_n1b,G_ave1b, Wi3, Wi4, BACREh = fun (1000, 1000, b, 34)    
    plt.plot(train_accu1b, label='gdst = %s '%b)
plt.ylabel('performance')
plt.title('Effect of gradient descent step size on performance of training set')
plt.legend(loc='center right')
plt.show()   


""" Effect of number of neurons in the hidden layer (dimension h) on Performance"""

for b in [11,18,27,34]:    
    train_accu1b, test_accu1b,train_lossb, test_lossb,axb,ayb,LOSS1b,W_n1b,G_ave1b, Wh1, Wh2, BACREbh = fun (1000, 2000, 0.01, b)    
    plt.plot(train_accu1b, label='h= %s '%b)
    plt.ylabel('performance')
    plt.title('Effect of dimension h on performance of training set')
plt.legend(loc='center right')   
plt.show()



"""PART 5:
    Analysis of hidden layer behaviour on the MLP
    
"""

"""Get the global states of the hidden layer """
hidden_global_states = []
for a in range(10002):
    cc=np.reshape(W34, (15,34))
    dd= np.array([scaled_data.iloc[a,:]])
    ry = pd.DataFrame(np.matmul(dd,cc)+np.matrix(W35))  
    hidden_global_states.append(ry)
global_states=pd.concat(hidden_global_states)

"""Print the first 5 cases of the hidden layer global states"""
global_states.head(5)#First 5 cases


"""Scale the data and find the absolute values of the correlation for redundancy and pruning check
   show the first 10 cases
"""

global_states_std=StandardScaler().fit_transform(global_states) 
corr_global_states=np.corrcoef(np.transpose(global_states_std))
a=abs(pd.DataFrame(corr_global_states))
a.head(10) 

"""Print the Eigen values """
pca2l = PCA(n_components=34)
pca2l.fit(global_states)
print((pca2l.explained_variance_ratio_))


"""Plot the Eigen values"""
plt.figure(figsize=(7, 5))
plt.figure(1)
Le=sorted(pca2l.explained_variance_,reverse=True, )
plt.plot(Le,  marker='o', label='Eigen Values', color='r')
plt.ylabel('Eigen Values')
plt.xlabel('')
plt.title('Plot of Eigen Values for Hidden Layer Global States')
plt.legend()


dg=pca2l.explained_variance_ratio_ 
da1=[]
for i in range(34):    
    if i ==0:
        Ri = dg[i]
    else:
        Ri+=dg[i]        
    da1.append(Ri)
    
"Plot the cummulative sum of the explained variance"
plt.plot(da1,  marker='o', label='Lj', color='g')
plt.ylabel('Rj')
plt.xlabel('j')
plt.title('Plot of Lj vs Rj for Hidden Layer Global States')
plt.legend()

"""
The smallest number Ub that explains about 95% variance of the global states of the hidden layer
This number is estimated from the plot above and code below

"""
#Smallest Number uB
bb1=pca2l.explained_variance_
compare1bb=sum(bb1)*0.95
s1bb=0
count1bb=0
for i in bb1:
    count1bb=count1bb+1
    s1bb=s1bb+i
   # print(i)
    if s1bb>compare1bb:
       # print(count1)
        break
    
Ub=count1bb
print("The smallest number Ub is: " + str(Ub)) #Estimated as 10



"Get the classes of the global states"
ff=pd.concat([(global_states.reset_index(drop=True)),y_output], axis=1)
D111= ff[ff['Class']==1].iloc[:,0:34]
D211= ff[ff['Class']==2].iloc[:,0:34]
D311= ff[ff['Class']==3].iloc[:,0:34]
print ("Class 1: "+ str(D111.shape[0]),"Class 2: "+ str(D211.shape[0]), "Class 3: "+str(D311.shape[0]) )



"""Perform PCA Analysis on the Global States of the hidden layer"""
pca2l1 = PCA(n_components=3)
pca2l1.fit(global_states)
result11=pd.DataFrame(pca2l1.transform(D111), columns=['PCA%i' % i for i in range(3)], index=D111.index)
result12=pd.DataFrame(pca2l1.transform(D211), columns=['PCA%i' % i for i in range(3)], index=D211.index)
result13=pd.DataFrame(pca2l1.transform(D311), columns=['PCA%i' % i for i in range(3)], index=D311.index)
print(result12)



"""Projection in 3 Dimensions"""
fig = plt.figure(figsize=(8, 6))
axv = fig.add_subplot(1,1,1, projection='3d') 
axv.scatter(result11['PCA0'], result11['PCA1'], result11['PCA2'],s=8,marker='*', color='g', label='Class 1')
axv.scatter(result12['PCA0'], result12['PCA1'], result12['PCA2'],s=8,marker='o', color = 'b', label='Class 2')
axv.scatter(result13['PCA0'], result13['PCA1'], result13['PCA2'],s=8, marker='+', color='r', label='Class 3')
axv.set_xlabel("PC1")
axv.set_ylabel("PC2")
axv.set_zlabel("PC3")
axv.legend (loc='best')
axv.set_title("PCA on the global states of the hidden layer")


"""Compute the average hidden neurons activity profiles PROF1 , PROF2, PROF3"""
average_D111= (D111.sum(axis=0))/10002 #Class 1 , PROF1
average_D211= (D211.sum(axis=0))/10002 #Class 2, PROF2,
average_D311=(D311.sum(axis=0))/10002 #Class 3, PROF3
average_All = (global_states.sum(axis=0))/10002 #Class 3


""" Histogram of average activity for Class 1"""
n, bins, patches = plt.hist(average_D111, bins=12, color='b',alpha=0.7)
plt.show()


""" Histogram of average activity for Class 2"""
n, bins, patches = plt.hist(average_D211, bins=12)
plt.show()

""" Histogram of average activity for Class 3"""
n, bins, patches = plt.hist(average_D311, bins=12)
plt.show()


""" Histogram of average activity for the global states"""
n, bins, patches = plt.hist(average_All, bins=12)
plt.show()


"""
    Investigation of differentiation among class activities
    
"""    

""" PROF 1, PROF 2, PROF 3 """
plt.figure(figsize=(9, 7))
plt.plot(average_D111,color='r',  label='Class 1 Average Activity Profile (PROF1)')
plt.plot(average_D211,color='b',  label='Class 2 Average Activity Profile (PROF2)')
plt.plot(average_D311,color='g',  label='Class 3 Average Activity Profile (PROF3)')
plt.ylabel('Activities')
plt.xlabel('')
plt.xticks(np.arange(0, 34+1, 1.0))
plt.title('Plot of Activities for Class 1, Class 2 and Class 3')
plt.legend(loc='lower left')


""" PROF 1 and PROF 2"""
plt.figure(figsize=(9, 7))
plt.plot(average_D111,color='r',  label='Class 1 Average Activity Profile (PROF1)')
plt.plot(average_D211,color='b',  label='Class 2 Average Activity Profile (PROF2)')
plt.ylabel('Activities')
plt.xlabel('')
plt.xticks(np.arange(0, 34+1, 1.0))
plt.title('Plot of Profile Activities for Class 1 and Class 2')
plt.legend(loc='lower left')


""" PROF 1 and PROF 3"""
plt.figure(figsize=(9, 7))
plt.plot(average_D111,color='r',  label='Class 1 Average Activity Profile (PROF1)')
plt.plot(average_D311,color='g',  label='Class 3 Average Activity Profile (PROF3)')
plt.ylabel('Activities')
plt.xlabel('')
plt.xticks(np.arange(0, 34+1, 1.0))
plt.title('Plot of Profile Activities for Class 1 and Class 3')
plt.legend(loc='lower left')


""" PROF 2 and PROF 3"""
plt.figure(figsize=(9, 7))
plt.plot(average_D211,color='b',  label='Class 2 Average Activity Profile (PROF2)')
plt.plot(average_D311,color='g',  label='Class 3 Average Activity Profile (PROF3)')
plt.ylabel('Activities')
plt.xlabel('')
plt.xticks(np.arange(0, 34+1, 1.0))
plt.title('Plot of Profile Activities for Class 2 and Class 3')
plt.legend(loc='lower left')





