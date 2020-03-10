#
#
# DESCRIPTION:
#
#       This project involves the prediction of ExxonMobil Stock Prices using Kernel Ridge Regression algorithm
# 
# DATASET:
#
#       The dataset involves 10 stock prices (as explanatory variables) for the following companies;
#
#       Suncor Energy, Imperial Oil Limited, TC Energy Corporation, Cenovous Energy, Enbridge Energy, Encana Energy,
#
#       CNOOC, Chevron, Bp, and Total. 
#
#       Artificial Feature of 1 is included making the total number of  explanatory variables equals 11.
#
#       The target variable is the stock price of Exxonmobil which will be predicted when a new data is given using
#
#       the Kernel Ridge Regression technique. 
#
#        Also, this dataset was downloaded from Yahoo finance between the period  04 December 2015 to 04 December 2019.
#
#        This gives the total number of cases as 1,258. More so, the adjusted closing price was used for the stock price.
#
#
#   STEPS:
#
#        1) Download companies stock data from website and perform brief data analysis
#
#       2)  Perform Kernel Ridge Regression (KRR) with radial kernel using a Formula 
#       
#       3)  Improve the results through step by step tuning
#
#       4)  Implement KRR using a pre-existing function in R
#







#   PART 1:  
#
#       Read and import the respective companies stock price data 
#
#       Take only the adjusted closing price of the  stocks


options(digits=4)

SU1=droplevels(data.frame(read.csv("SU.csv", header=TRUE, sep=',', na.strings = "?")[,6]))  # Suncor Energy

IMO1= droplevels(data.frame(read.csv("IMO.csv", header=TRUE, sep=',', na.strings = "?") [,6])) # Imperial Oil Limited

TRP1= droplevels(data.frame(read.csv("TRP.csv", header=TRUE,sep=',', na.strings = "?") [,6])) # TC Energy Corporation

CVE1=droplevels(data.frame(read.csv("CVE.csv", header=TRUE, sep=',', na.strings = "?") [,6]))  # Cenovous Energy

ECA1=droplevels(data.frame(read.csv("ECA.csv", header=TRUE, sep=',', na.strings = "?") [,6]))      #Encana Corporation

CEO1=droplevels(data.frame(read.csv("CEO.csv", header=TRUE, sep=',', na.strings = "?") [,6]))     #CNOOC Limited

CVX1=droplevels(data.frame(read.csv("CVX.csv", header=TRUE, sep=',', na.strings = "?") [,6]))     #Chevron Limited

BP1=droplevels(data.frame(read.csv("BP.csv", header=TRUE, sep=',', na.strings = "?") [,6]))      #BP-British Petroleum PLC

TOT1=droplevels(data.frame(read.csv("TOT.csv", header=TRUE, sep=',', na.strings = "?") [,6]))  #Total  PLC

ENB1=droplevels(data.frame(read.csv("ENB.csv", header=TRUE, sep=',', na.strings = "?") [,6]))   #Enbridge Limited

XOM1=droplevels(data.frame(read.csv("XOM.csv", header=TRUE, sep=',', na.strings = "?") [,6]))  #ExxonMobil PLC - Target Variable


                
                           #   Merge the data together to give 11 explanatory variables and 1 target variable 

Rdata = cbind(SU1,ENB1, IMO1, TRP1, CVE1, ECA1, CEO1, CVX1, BP1, TOT1,COMX, XOM1)

colnames(Rdata)=c("Suncor E.","Enbridge E.", "Imperial Oil", "TC E.","Cenovous E.", "Encana E.","CNOOC","Chevron",
                  "BP","Total","ComVariable", "ExxonMobil") 

Rdata["ComVariable"]=1


#Display the first 6 cases

head(Rdata)



                                  # **Estimate and display the Mean and Standard deviation of explanatory variables**
  
Rdatax =Rdata[,1:11] #get the explanatory variable

meanRdatax=colMeans(Rdatax) #get the mean of the data

sdRdatax = sapply(Rdatax, sd) #get the standard deviation

stx = rbind(meanRdatax, sdRdatax) 

row.names(stx)=c('Mean','Standard Deviation')

stx # Display result


                              
                            #**Plot the Mean and Standard deviation of explanatory variables (i.e. stocks for each company)**

plot(sdRdatax, meanRdatax,main = "Mean vs StDev. for Explanatory Variables",col="red", pch=16, cex=1, ylab = "Mean",
     
     xlab="Standard Deviation", col.axis='red')


                              
                         
                              # **Mean and Standard deviation of Target Variable (i.e. ExxonMobil Stocks)**

Rdatay= data.frame(Rdata[,12])       #get the target variable

meanRdatay=sapply(Rdatay, mean)       #get the mean 

sdRdatay = sapply(Rdatay, sd)          #get the  standard deviation

sty = rbind(meanRdatay, sdRdatay)

colnames(sty)=c('ExxonMobil')

row.names(sty)=c('Mean','Standard Deviation')

sty   #display the mean and standard deviation of exxonmobil stocks for analysis


                         

                                    
                                     # **Split the data into 80% Training set and 20% Test Set**


set.seed(105688)                                           #set seed to fix randomization

explx = sort(sample(nrow(Rdata), nrow(Rdata)*0.8))  

train=Rdata[explx,]#creating training data set 80% 

test=Rdata[-explx,] #creating test data set 20%

com = cbind(nrow(train), nrow(test))

colnames(com)=c('Train', 'Test')

com #display number of cases in training and test sets




                         #  **Empirical correlations with ExxonMobil stocks and its absolute Value for analysis**

corx=cor(Rdatax$`Suncor E.`,Rdatay)            # Correlation of Suncor Energy Stocks with ExxonMobil Stocks

coree=cor(Rdatax$`Enbridge E.`,Rdatay)         # Correlation of Enbridge Stocks with ExxonMobil Stocks

corim=cor (Rdatax$`Imperial Oil`,Rdatay)       # Correlation of Imperial Oil Stocks with ExxonMobil Stocks

cortc=cor(Rdatax$`TC E.`,Rdatay)               # Correlation of TC Energy Stocks with ExxonMobil Stocks

corcen=cor(Rdatax$`Cenovous E.`,Rdatay)        # Correlation of Cenovous Energy Stocks with ExxonMobil Stocks

coren=cor(Rdatax$`Encana E.`,Rdatay)           # Correlation of Encana Stocks with ExxonMobil Stocks

corcn=cor(Rdatax$CNOOC,Rdatay)                 # Correlation of CNOOC Stocks with ExxonMobil Stocks

corchev=cor(Rdatax$Chevron,Rdatay)             # Correlation of Chevron Stocks with ExxonMobil Stocks

corbp=cor(Rdatax$BP,Rdatay);                   # Correlation of BP Stocks with ExxonMobil Stocks

cortot=cor(Rdatax$Total,Rdatay)                # Correlation of Total Stocks with ExxonMobil Stocks

correlation = cbind(corx,coree,corim,cortc,corcen,coren,corcn,corchev,corbp,cortot) #combine for comparison

abs_value = abs(correlation) #absolute value of correlation for analysis

ade= data.frame(rbind(correlation,abs_value))

colnames(ade)=c("Suncor E.","Enbridge E.", "Imperial Oil", "TC E.", "Cenovous E.", "Encana E.","CNOOC", "Chevron", "BP", "Total")

row.names(ade)=c('correlation', 'Absolute Value')

ade  ## Display the correlation and absolute values for analysis




                              #   **The Three Largest Value for absolute value of correlation** 

lg=sort(ade[2,], decreasing = TRUE)

lg[,1:3]  #Display the companies with largest absolute correlation for analysis



                   
                             #   **Display the three scatter plots for the largest absolute correlation Values** 

plot(Rdata$ExxonMobil,Rdata$`Encana E.`,main = "Stocks: Encana vs Exxon", col="red", pch=16, cex=1,
     ylab = "Encana Energy", xlab="ExxonMobil", col.axis='red')

plot(Rdata$ExxonMobil,Rdata$Chevron, main = "Stocks: Chevron vs Exxon",col="blue", pch=16, cex=1, 
     ylab = "Chevron", xlab="ExxonMobil", col.axis='blue')

plot(Rdata$ExxonMobil,Rdata$`Suncor E.`, main = "Stocks: Suncor E. vs Exxon",col="darkgreen", pch=16,
     cex=1, ylab = "Suncor Energy", xlab="ExxonMobil", col.axis='darkgreen')






#  PART 2 :
#        
#        Kernel Ridge Regression (KRR) with radial kernel using a prediction formula
#
#
                              


                                           # **Reordering to simplify cases**

rownames(train) = seq(length=nrow(train))#reordering the train data

rownames(test) = seq(length=nrow(test))#reordering the test data



             #  **Splitting the train and test data into input and output**

trainx1=train[,1:11] #Features for training set

trainy1=train[,12] #target variable for training set

testx1=test[,1:11] #Features for test set

testy1=test[,12]  #target variable for test set




              #    **COmpute Pairwise Distances to Choose Gamma**
                
library(rdist)

set.seed(105687)

pwd1= sort(as.numeric(rdist(trainx1,metric="euclidean",p=2L ))) 

quand1= quantile(pwd1, probs = c(0.1, 0.9)) # compute the percentile

quand1 # Display 10-90% quantile


 

                           #**Estimate Initial value of Gamma**

gamma1=as.numeric(1/quand1[1])

gamma1 #Display chosen intial value for gamma


 
                              #**Compute Matrix G (Gramian)**

library(rdetools)

G = rbfkernel(as.matrix(trainx1), sigma = 1/gamma1)

dim(G)    #Gramian Dimension


                            
                           
                           #**Gramian Eigen Values and Vectors**

cor_G=cor(G) #Estimating correlation of Gramian

eigenG=eigen(cor_G) #Estimating eigen values of G

eigen_Gvalue= eigenG$values  #Eigen values of G

eigen_Gvector=eigenG$vectors #Eigen vectors of G



                                       #     **Plot of Eigen Values **

plot(eigen_Gvalue,  main = "Lj versus j",col="red",cex=1, pch=16, col.axis='red', xlab = "Case j",
     
     ylab ="Eigen Values",col.axis='red', type='l', lwd=2.5)

legend(x=300, y=35, legend=c('Gramian Eigen Values'),col=c('red'), pch=c(16))




                                  # *Plot of Ratio j (cummulative sum of eigen values) vs j **


eigensum = sum(eigenG$values) # sum of the eigen values
datalist = list()
for (i in 1:1006)
{
  if (i==1)
  {
    Ri= (eigenG$values[1])
  }
  else 
  {
    Ri=(Ri + eigenG$values[i])
  }
  datalist[i] = Ri
  i=i+1
}
ratdata = do.call(rbind, datalist)
ratdata= ratdata/eigensum #Cummulative ratio of the sum of the eigen values for analysis

plot(ratdata, main = "Ratio Eigen ",col="red",col.axis='red', xlab = "j", ylab = "Ratio Eigen",col.axis='red',type='l', lwd=2)
legend(x=300, y=35, legend=c('Ratio j'), col=c('red'))




                             # Find the smallest number (j) that explain about 95% variance in the data

                             #This number is estimated from the plot above and code below


big1=data.frame(ratdata )

a=min(which(ratdata>0.95))
a                   # Display the smallest number that explaining 95% variance in the data


                              

                                   #**Corresponding Value of Lj**


lambda1=ratdata[a,]   #Set this as value of Lambda 

lambda1             # Display result 


                   
                                    #**Selecting Random Integers**

set.seed(108) #set the seed

list1= runif(100,min=1,max=nrow(trainx1))   #100 random numbers 

set.seed(1017) #set the seed

list2= runif(100, min=1, max=nrow(trainx1)) #100 random numbers



                               # *Estimating Distance Dij: Pairwise Distance**

pwd2= sort(as.numeric(cdist(list1,list2, metric = "euclidean", p=2L ))) 


                              

                                #**Histogram of 10,000 distances**

hist(pwd2, main = "Histogram of Dij",col="red",cex=1,pch=16,col.axis='darkblue', xlab = "Distance (Dij)",
     
     ylab = "Frequency",col.axis='darkblue')


                                #**Find the 10% quantile of the 10000 distances**

quand2= quantile(pwd2, probs = c(0.1, 0.9)) #Estimating 10-90% quantile


                     
                                 #**Set 10% quantile as Gamma**

gamma2=as.numeric(1/quand2[1]) #getting the ratio of 10% quantile

gamma2 

                                 #**Compute M  and its inverse**

M = G + (lambda1 * diag(1006)) 

Minv = solve(M)      #Inverse of M



                                  #** Compute Line Vector A**

Aij = t(trainy1) %*% Minv # Line vector


                 #**Prediction of Train Set**

datalist2 = list()

for (i in 1:1006)
  
{
  vi=rbfkernel(as.matrix(trainx1),sigma=1/(gamma1),  #sigma is used instead
               
               Y=t(as.matrix(as.integer(trainx1[i,]))))
  
  dd = Aij %*% vi
  
  datalist2[i] = dd
  
  i=i+1
}
compt1 = do.call(rbind, datalist2) #Prediction

errort= as.data.frame(trainy1) - compt1 #Error of Prediction

MSEt = sum(errort*errort)/1006 #MSE

RMSEt= sqrt(MSEt)             #RMSE

avet = sum(trainy1)/1006

RRMSEt=RMSEt/avet       #Ratio RMSE



                              #   **Prediction of test case**

datalist2 = list()

for (i in 1:252)
  
{
  vi = rbfkernel(as.matrix(trainx1), sigma = 1/(gamma1),
                 
                 Y=t(as.matrix(as.integer(testx1[i,]))))
  
  dd = Aij %*% vi
  
  datalist2[i] = dd
  
  i=i+1
  
}
compt2 = do.call(rbind, datalist2) #Prediction

errortt= as.data.frame(testy1) - compt1 #Error of Prediction

MSEtt = sum(errortt*errortt)/252 #MSE

RMSEtt= sqrt(MSEtt) #RMSE

avett = sum(testy1)/252

RRMSEtt=RMSEtt/avett #Ratio RMsE


                                          #**PLot of Prediction for Training Data**

plot( trainy1,xlim=range(0:1006),ylim=range(compt1,trainy1),col="red",pch=16, cex=0.5, xlab = "Case i", ylab="Exxon Stock Price (y)",
      
      main='Y vs y hat  for Training Data', col.axis='red', type='l', lwd=2.5)

points(compt1,col="blue", pch=16, cex=0.5, type='l', lwd=2.5)

legend(x=250, y=40, legend=c('Base Case', 'Predicted'), col=c('red', 'blue'), pch=c(16, 16))



                                              #**PLot of Prediction for Test Data**

plot( testy1,xlim=range(0:252),ylim=range(compt2,testy1),col="red",pch=16, cex=0.5, xlab = "Case i", ylab="Exxon Stock Price (y)",
      
      main='Y vs y hat  for Test Data',col.axis='red',type='l',lwd=2.5)

points(compt2,col="blue", pch=16, cex=0.5, type='l', lwd=2.5)

legend(x=85, y=36, legend=c( 'Base Case', 'Predicted'), col= c('red', 'blue'), pch=c(16, 16))


                   
                                                 #**Table of Results**

trainc1=c(MSEt,RMSEt, RRMSEt , 1/gamma1, gamma1, lambda1)

testc1= c(MSEtt,RMSEtt, RRMSEtt , 1/gamma1, gamma1, lambda1)

results1 = data.frame(rbind(trainc1, testc1))

rownames(results1)= c('Train Set', 'Test Set')

colnames(results1)= c('MSE', 'RMSE', 'Ratio RMSE','Sigma', 'Gamma', 'Lambda')
                     
results1 #Dsiplay result



### PART 3: 
#         
#         Improving the results through step by step tuning
#
                   

                                   #**Getting two more gammas and lamdas**

#lambda

a3=min(which(ratdata>0.9))#Smallest 90

lambda3=ratdata[a3,] #lambda3 

a4=min(which(ratdata>0.85))  #Smallest 85

lambda4=ratdata[a4,]   #lambda 4

#gamma

quand3= quantile(pwd2, probs = c(0.05, 0.95)) #Estimating 5-95% quantile

gamma3=as.numeric(1/quand3[1]) #getting the ratio of 5% quantile

quand4= quantile(pwd2, probs = c(0.2, 0.8)) #Estimating 20-80% quantile

gamma4=as.numeric(1/quand4[1]) #getting the ratio of 20% quantile


                  
                                    # **COmputing the results for each pair of the three values**

lambdas =c(lambda1, lambda3, lambda4)

gammas = c(1/gamma1, 1/gamma3, 1/gamma4)

ade = function(x,y){
  
  datalist4tr = list()
  
  datalist4te = list()
  
  
  MSEtr = list()
  
  RMsEtr = list()
  
  RRMSEtr = list()
  
  
  MSEte = list()
  
  RMsEte = list()
  
  RRMSEte = list()
  
  G = rbfkernel(as.matrix(trainx1), sigma = gammas[x])
  
  M3 = G + (lambdas[y] * diag(1006)) 
  
  Aij3 = t(trainy1) %*% solve(M3)     
  
  datalist3 = list()
  
  for (k in 1:1006)
    
  {
    vi1 = rbfkernel(as.matrix(trainx1), sigma = gammas[x],
                    
                    Y=t(as.matrix(as.integer(trainx1[k,]))))
    
    dd3 = Aij3 %*% vi1
    
    datalist4tr[k] = dd3
    
    k=k+1
  }
  
  compt3 = do.call(rbind, datalist4tr)
  
  errort3tr= as.data.frame(trainy1) - compt3
  
  MSEt3tr = sum(errort3tr*errort3tr)/1006 #MSE
  
  MSEtr[k]=MSEt3tr 
  
  RMSEt3tr= sqrt(MSEt3tr)             #RMSE
  
  RMsEtr[k]= RMSEt3tr
  
  avet3tr = sum(trainy1)/1006
  
  RRMSEt3tr=RMSEt3tr/avet3tr       #Ratio RMSE
  
  RRMSEtr[k]=RRMSEt3tr
  
  for (m in 1:252)
    
  {
    vi3t = rbfkernel(as.matrix(trainx1), sigma = gammas[x],Y=t(as.matrix(as.integer(testx1[m,]))))
                     
    dd3t = Aij3 %*% vi3t
    
    datalist4te [m] = dd3t
    
    m=m+1
    
  }
  
  compt4te = do.call(rbind, datalist4te)
  
  errortt3te= as.data.frame(testy1) - compt4te
  
  MSEtt3t = sum(errortt3te*errortt3te)/252 #MSE
  
  MSEte[m]= MSEtt3t
  
  RMSEtt3t= sqrt(MSEtt3t) #RMSE
  
  RMsEte[m]= RMSEtt3t
  
  
  avett3t = sum(testy1)/252
  
  RRMSEtt3t=RMSEtt3t/avett3t #Ratio RMsE
  
  RRMSEte[m] = RRMSEtt3t
  
  
  MSEtr1 = do.call(rbind, MSEtr )
  
  RMsEtr1 = do.call(rbind, RMsEtr)
  
  RRMSEtr1 = do.call(rbind, RRMSEtr)
  
  
  MSEte1 = do.call(rbind, MSEte)
  
  RMsEte1 = do.call(rbind, RMsEte )
  
  RRMSEte1 = do.call(rbind, RRMSEte)
  
  
  options(digits=4)
  
  dd1 = c(MSEtr1, RMsEtr1,RRMSEtr1, 1/gammas[x],gammas[x],lambdas[y]);
  
  dd2 = c(MSEte1, RMsEte1,RRMSEte1, 1/gammas[x],gammas[x],lambdas[y])
  
  
  dd3 = data.frame(rbind(dd1,dd2))
  
  colnames(dd3)= c('MSE:', 'RMSE', 'Ratio RMSE','Gamma','Sigma','Lambda')
  
  rownames(dd3)= c('Train', 'Test')
  
  return(dd3)
}


vars1=c(1,2,3)

vars2=c(1,2,3)

d = function (vars1,vars2)
  
{
  for (i in vars1){
    
    for (j in vars2){
      
      print(ade(i,j))
      
    }
  }
  
}


              #**Results for Each Pair of Gamma and Lamda**


d(vars1, vars2) #Display result



                #**Considering the Best Case Model**
  
                
                   #  **Training Set**

G = rbfkernel(as.matrix(trainx1), sigma = 102.8) #Gramian

Mb = G + (0.8522 * diag(1006)) 

Minvb = solve(Mb)               #Matrix inverse

Aijb = t(trainy1) %*% Minvb      # Line vector


datalist2b = list()

for (i in 1:1006)
  
{
  vib = rbfkernel(as.matrix(trainx1), sigma = 102.8,
                  
                  Y=t(as.matrix(as.integer(trainx1[i,]))))
  
  ddb = Aijb %*% vib
  
  datalist2b[i] = ddb
  
  i=i+1
}
compt1b = do.call(rbind, datalist2b) #Prediction

errortb= as.data.frame(trainy1) - compt1b #Error on prediction

MSEtb = sum(errortb*errortb)/1006 #MSE

RMSEtb= sqrt(MSEtb)             #RMSE

avetb = sum(trainy1)/1006


RRMSEtb=RMSEtb/avetb       #Ratio RMSE


                       



                                            #**Test case**

datalist2bt = list()

for (i in 1:252)
  
{
  vibt = rbfkernel(as.matrix(trainx1), sigma =102.8, Y=t(as.matrix(as.integer(testx1[i,]))))
                  
  
  ddbt = Aijb %*% vibt
  
  datalist2bt[i] = ddbt
  
  i=i+1
  
}

compt2bt = do.call(rbind, datalist2bt) #Prediction

errorttb= as.data.frame(testy1) - compt2bt #Error on prediction

MSEttb = sum(errorttb*errorttb)/252 #MSE

RMSEttb= sqrt(MSEttb) #RMSE

avettb = sum(testy1)/252

RRMSEttb=RMSEttb/avettb #Ratio RMsE


                    


                                    # **PLot of Prediction for Training Data**


plot(trainy1,xlim=range(0:1006),ylim=range(compt1b,trainy1), col="red",pch=16, cex=0.5, xlab = "Case i", 
     
       ylab="Exxon Stock Price (y)",main='Y vs y hat  for Best Case (Training Data)', col.axis='red', type='l', lwd=2.5)

points(compt1b,col="blue", pch=16, cex=0.5, type='l',lwd=2.5)

legend(x=350, y=60, legend=c( 'Base Case', 'Predicted'),   col=c('red', 'blue'), pch=c(16, 16))
     


                                     #**PLot of Prediction for Test Data**

plot(testy1, xlim=range(0:252),ylim=range(compt2,testy1),col="red",pch=16, cex=0.5, xlab = "Case i", ylab="Exxon Stock Price (y)",
     
     main='Y vs y hat  for Best Case(Test Data)', col.axis='red',type='l',lwd=2.5)

points(compt2bt,col="blue", pch=16, cex=0.5, type='l', lwd=2.5)

legend(x=85, y=36, legend=c('Base Case', 'Predicted'), col=  c('red', 'blue'), pch=c(16, 16))
        


                                     #**10 Cases of Squared Prediction Error Highest in Test Set**
  
testnew =test

testnew['SE']= errorttb*errorttb #Squared Error

sorterror = testnew[order(-testnew$SE),] #Sorted by error

sorted1 = sorterror[1:10,]

sorted1 


                        #**10 Cases of Squared Prediction Error Lowest in Test Set**

lowerror = sorterror[242:252,]

lowerror


                            #**Visualizing Lowest Errors Through PCA**

pctest = prcomp(testx1[,1:10],  scale=TRUE)

scores=pctest$x[,1:3]

class3 = c(52,53,238,1,183,239,192,22,177,92) #Index of highest error cases

class4= c(110,69,146,75,244,16,34,136,100,200) #Index of lowest error cases


sco_error=scores[class3,] #High error scores

real_score = scores[-class3,] 

lowscor_error=scores[class4,] #Low error scores

real_score1 = real_score[-class4,] #New Cases




                                  #**3D Projection of the erros

library(scatterplot3d)

s3d=scatterplot3d(real_score1,color="red",pch=16, box=TRUE,xlab="PC1", ylab="PC2",zlab="PC3", 
                  
                  main = "3 Dimenstional Scatter Plots for the 3 scores")

s3d$points3d(sco_error,col="blue",pch=16)

s3d$points3d(lowscor_error,col="black",pch=16)

legend(s3d$xyz.convert(-2, -3, -0.95), col= c("red","blue", "black"),pch=16,
       
legend = c("Normal Data", "Highest Error", "Lowest Error"), cex = 0.8)






                             #**Pairwise Distance Between Lowest and Highest Error**

dehigh= sort(as.numeric(rdist(sorted1, metric = "euclidean", p=2L ))) #Pairwise distance Error high

delow= sort(as.numeric(rdist(lowerror, metric = "euclidean",p=2L ))) #Pairwise distance Error low




                             #**Plot of Pairwise Distance between lowest and highest error to identify what went wrong**

plot(dehigh, xlim=range(0:60),ylim=range(delow,dehigh),col="red",pch=16,cex=0.5, xlab = "Number of Pairs", ylab="Pairwise Distance",
     
     main='Pairwise Distance', col.axis='red', type='l', lwd=2.5)

points(delow,col="blue", pch=16, cex=0.5, type='l', lwd=2.5)

legend(x=5, y=120, legend=c('Error High', 'Error Low'), col=c('red', 'blue'), pch=c(16, 16))





### PART 4 : 
#
#         Analysis of the best predicting formula pred(x)
#



               #Fix the best choice of parameters as found in the preceding part. 

                  
            
                #  **Best Choice Parameters are Lambda=0.8522 and sigma=102.8/ gamma=0.009727** 
  
                               
                                       #  **Reordering A1 etc**

G = rbfkernel(as.matrix(trainx1), sigma = 102.8)

Mb = G + (0.8522 * diag(1006)) 

Minvb = solve(Mb)               #Matrix inverse

Aijb = t(trainy1) %*% Minvb      # Line vector

B= sort(abs(Aijb), decreasing = TRUE) #Reordering in decreasing order

B[1:5] #sample


                                       #**Plot Bj vs j**

plot(B, main = "Bj",col="red",col.axis='red', xlab = "j", ylab = "Bj",col.axis='red',type='l', lwd=2)

legend(x=600, y=10, legend=c('Bj'), col=c('red'),  pch=c(16))




                    #**Compute the ratios bj = (B1 + ... + Bj)/(B1 + ...+Bm) and plot the increasing curve bj versus j**


Bi=as.data.frame(B)

datalistbb = list()

for (i in 1:1006)
  
{
  if (i==1)
  {
    Ri= (Bi[1,])
  }
  else 
  {
    Ri=(Ri + Bi[i,])
  }
  datalistbb[i] = Ri
  i=i+1
}
bbdata1 = do.call(rbind, datalistbb)

bbdata1=bbdata1/sum(Bi)

plot(bbdata1, main = "Ratio Bj",col="red", col.axis='red', xlab = "j", ylab = "Ratio Bj",col.axis='red',type='l', lwd=2)

legend(x=700, y=0.2, legend=c('Ratio Bj'), col=c('red'), pch=c(16))




                                     #**Compute the smallest value j such that bj > 99%. **

min_j=min(which(bbdata1>0.99))
min_j       #Smallest j


                            #**Corresponding Threshold THR**

THR=bbdata1[min_j]
THR


                       #**Run the Reduced Formula (PRED(x)  = AA1 K(x, X(1)) + ... + AAm K(x,X(m)) and Predict Training Case**

G = rbfkernel(as.matrix(trainx1), sigma = 102.8)

Mb = G + (0.8522 * diag(1006)) 

Minvb = solve(Mb)               #Matrix inverse

Aijb = t(trainy1) %*% Minvb      # Line vector




                                     #Prediction for Training Set

datalist21=list()

for (i in 1:1006)
  
{
  if (abs(Aij[i])>= THR)
    
  {
    AAi = Aijb
    
    vif = rbfkernel(as.matrix(trainx1), sigma = 102.8,Y=t(as.matrix(as.integer(trainx1[i,]))))
                    
    dd = AAi %*% vif
    
  }
  
  else 
    
  {
    AAi = 0
    
    dd = 0
  }
  
  datalist21[i] = dd
  
  i=i+1   
}

compt1r = do.call(rbind, datalist21) #Prediction

errort1r= as.data.frame(trainy1) - compt1r  #Error of Prediction

MSEt1r = sum(errort1r*errort1r)/1006 #MSE

RMSEt1r= sqrt(MSEt1r)             #RMSE

avet1r = sum(trainy1)/1006

RRMSEt1r=RMSEt1r/avet1r



                                    #**Prediction of Test Case**

datalist2tb = list()

for (i in 1:252)
  
{
  
  if (abs(Aij[i])>= THR)
    
  {
    AAi = Aijb
    
    vibt = rbfkernel(as.matrix(trainx1), sigma = 102.8,Y=t(as.matrix(as.integer(testx1[i,]))))
                     
    dd = AAi %*% vibt
  }
  
  else 
    
  {
    
    AAi = 0
    
    dd = 0
  }
  
  datalist2tb[i] = dd
  
  i=i+1   
  
}

compt21b = do.call(rbind, datalist2tb) #Prediction

errortt1b= as.data.frame(testy1) - compt21b #Error

MSEtt1b = sum(errortt1b*errortt1b)/252 #MSE

RMSEtt1b= sqrt(MSEtt1b) #RMSE

avett1b = sum(testy1)/252

RRMSEtt1b=RMSEtt1b/avett1b #Ratio RMsE


                                              #**PLot of Prediction for Training Data**

plot( trainy1,xlim=range(0:1006),ylim=range(compt1r,trainy1),col="red",pch=16, cex=0.5, xlab = "Case i", ylab="Exxon Stock Price (y)",
      
      main='Y vs y hat  for Reduced Case (Training Data)', col.axis='red', type='l', lwd=2.5)

points(compt1r,col="blue", pch=16, cex=0.5)

legend(x=350, y=35, legend=c( 'Base Case', 'Predicted'), col=c('red', 'blue'), pch=c(16, 16))



                                             #**PLot of Prediction for Test Data**

plot(testy1, xlim=range(0:252),ylim=range(compt21b,testy1),col="red",pch=16, cex=0.5, xlab = "Case i", ylab="Exxon Stock Price (y)",
     
     main='Y vs y hat  for Reduced Case(Test Data)',col.axis='red',type='l',lwd=2.5)

points(compt21b,col="blue", pch=16, cex=0.5)

legend(x=85, y=36, legend=c('Base Case', 'Predicted'), col=c('red', 'blue'), pch=c(16, 16))


                                                          

                                       # Compare these performances to the original formula pred(x) and interpret the results


                                                       #**Table of Results**

options(digits=4)

#Best Case

dd1bb = c(MSEtb, RMSEtb,RRMSEtb)

dd2bb = c(MSEttb, RMSEttb,RRMSEttb)

dd3bb = data.frame(rbind(dd1bb,dd2bb))

#Reduced Case

dd1rr = c(MSEt1r, RMSEt1r,RRMSEt1r)

dd2rr = c(MSEtt1b, RMSEtt1b,RRMSEtt1b)

dd3rr = data.frame(rbind(dd1rr,dd2rr))

options(digits=3)

combd1= data.frame(rbind(dd3bb,dd3rr))

colnames(combd1)= c('MSE:', 'RMSE', 'Ratio RMSE')

rownames(combd1)= c('Train_Best','Test_Best', 'Train_Reduced','Test_Reduced')
                    
combd1






# PART 5: 
#
#        Implementation of KRR using a pre existing function







library(krr)

                                           #Training Set

d= krr(as.matrix(trainx1), (trainy1), 0.8522, sigma = 0.009727)

predtrain = predict(d, as.matrix(trainx1)) #Prediction

MSEtrainform <- colMeans(((trainy1 - predtrain)^(2)))

RMSEformtrain= sqrt(MSEtrainform) #RMSE

avett1bformtr = sum(trainy1)/1006

RRMSEformtr=RMSEformtrain/avett1bformtr #Ratio RMsE

                                         


                                             # Test Set

predtest = predict(d, as.matrix(testx1))

MSEtestform <- colMeans(((testy1 - predtest)^(2)))

RMSEformtest= sqrt(MSEtestform) #RMSE

avett1bformte = sum(testy1)/252

RRMSEformte=RMSEformtest/avett1bformte #Ratio RMsE



                                            #**PLot of Prediction for Training Data**

plot( trainy1,xlim=range(0:1006),ylim=range(predtrain,trainy1),col="red",pch=16, cex=0.5, xlab = "Case i", ylab="Exxon Stock Price (y)",
      
      main='Y vs y hat  for Function (Training Data)', col.axis='red', type='l', lwd=2.5)

points(predtrain,col="blue", pch=16, cex=0.5, type='l', lwd=2.5)

legend(x=400, y=60, legend=c( 'Base Case', 'Predicted'), col=c('red', 'blue'), pch=c(16, 16))


                                              #**PLot of Prediction for Test Data**

plot(testy1, xlim=range(0:252),ylim=range(predtest,testy1),col="red",pch=16, cex=0.5, xlab = "Case i", ylab="Exxon Stock Price (y)",
     
     main='Y vs y hat  for Function (Test Data)',col.axis='red',type='l',lwd=2.5)

points(predtest,col="blue", pch=16, cex=0.5, type='l', lwd=2.5)

legend(x=85, y=36, legend=c('Base Case', 'Predicted'), col=c('red', 'blue'), pch=c(16, 16))
`



                                                        #**Comparision of Results**
  

trainbest= c(MSEtb, RMSEtb,RRMSEtb)

trainreduced= c(MSEt1r, RMSEt1r,RRMSEt1r)

trainfprefunc= c(MSEtrainform,RMSEformtrain, RRMSEformtr)

dd3train1 = data.frame(rbind(trainbest,trainreduced,trainfprefunc ))


testbest= c(MSEttb, RMSEttb,RRMSEttb)

testreduced= c(MSEtt1b, RMSEtt1b,RRMSEtt1b)

testfprefunc= c(MSEtrainform,RMSEformtrain, RRMSEformtr)

dd3test1 = data.frame(rbind(testbest,testreduced,testfprefunc))


options(digits=3)

combd4= data.frame(rbind(dd3train1,dd3test1))

colnames(combd4)= c('MSE:', 'RMSE', 'Ratio RMSE')

rownames(combd4)= c('Train_Best','Train_Reduced', 'Train_Function','Test_Best','Test_Reduced', 'Test_Function' )
                    
combd4




























