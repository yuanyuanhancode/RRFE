#!/usr/bin/bash
# -*- coding: utf-8 -*-
"""
Created on  2025

@author: yuanyuanhan
"""

 
 
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
 
#from sklearn.ensemble import HistGradientBoostingClassifier
import sys
 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
 
 
from sklearn.model_selection import cross_val_score
 
#from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
 
#from skfeature.function.similarity_based.reliefF import *
 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
#from minepy import MINE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
#from skfeature.function.similarity_based.lap_score import *
#from skfeature.function.similarity_based.fisher_score import *
#from skfeature.function.sparse_learning_based.RFS import *
#from skfeature.function.information_theoretical_based.MRMR import *
#from skfeature.function.sparse_learning_based.UDFS import *
#from skfeature.function.sparse_learning_based.MCFS import *
#from skfeature.utility.mutual_information import * 
# skfeature wrapper
#from skfeature.function.wrapper.decision_tree_backward import *
#from skfeature.function.wrapper.decision_tree_forward import *
#from skfeature.function.wrapper.svm_backward import *
#from skfeature.function.wrapper.svm_forward import *
# skfeature lap_score 修改 # W = kwargs['W']# 报错
from sklearn.naive_bayes import MultinomialNB
#from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from collections import Counter
from numpy import array
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
#from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
#from minepy import MINE
from sklearn.svm import NuSVC
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.svm import SVC
import itertools as it
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression       
from sklearn import preprocessing
#from minepy import MINE
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.svm import SVC
import itertools as it       
import warnings
from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
#from minepy import MINE
#from sklearn.ensemble import HistGradientBoostingClassifier
import numpy as np
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer


 
MaxRFE=7#sys.argv[2]#2
MaxRFE=int(MaxRFE)
randomseed=0
def z_score_normalization(X):
    """
    X: ndarray or list of values to be normalized
    """
    X = np.array(X, dtype=float)
    return (X - X.mean()) / X.std()
 


def delet(X,y,MaxRFE):
 
    clf1 = LinearSVC(penalty='l2') #LogisticRegression(solver='liblinear')
    #clf2 = LinearSVC(penalty='l2') 
    #clf3 =LogisticRegression(penalty="l1")
    clf1.fit(X, y)
    #clf2.fit(X, y)
    #clf3.fit(X, y)    
    cof1 = z_score_normalization(abs(clf1.coef_[0]))
    #cof2 = z_score_normalization(abs(clf2.coef_[0]))
    #cof3 = z_score_normalization(abs(clf3.coef_[0]))
    a= cof1#+cof3#  + cof2 +cof1
    print("cof",a)
    np.random.seed(randomseed)      
    
    a=list(a)
    x=np.argsort(a)
    x=x[0:MaxRFE] 
    return(x)


 
def delta(methodname,X1,y1,MaxRFE):
 
    if methodname=="NBayes": 
              
        cls = GaussianNB()         
    if methodname=="SVM": 
                
        cls = LinearSVC(penalty='l2')            
    if methodname=="LR":        
         
        cls = LogisticRegression()
    np.random.seed(randomseed)      
    cls.fit(X1,y1)
    a=abs(cls.coef_[0])
    a=list(a)
    x=np.argsort(a)
    x=x[0:MaxRFE] 
    
    return(x)

 
  
randomseed = 0
def k_fold_val(tmp_clf,x,y,k):
    sn = 0.0  # 敏感性
    sp = 0.0  # 特异性
    acc = 0.0  # 正确率
    TN=0
    FP=0
    FN=0
    TP=0   
    djix =0
    np.random.seed(randomseed)
    skf = StratifiedKFold(n_splits=k)
    skf.get_n_splits(x,y)
    ALL  =0 
    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        tmp_clf.fit(x_train, y_train)
        y_predict = tmp_clf.predict(x_test)
        conMatrix = confusion_matrix(y_test, y_predict)
        ALL=conMatrix.sum()+ALL
        con = conMatrix.shape[0] 
        for dj in range(0,con):
            djix = djix+conMatrix[dj][dj]
    acc=float(djix/ALL)#float(djix/ALL)
    #print("acc",acc) 
    return ( acc)


def Classifier(X,Y):
    #clflist = [GaussianNB(), LinearSVC(penalty='l2'), LogisticRegression(),DecisionTreeClassifier(), RandomForestClassifier()]
    np.random.seed(randomseed)   
    clf1 =  LogisticRegression(solver='liblinear')
    clf2 = LinearSVC(penalty='l2') 
    clf3 = KNeighborsClassifier(n_neighbors=5) 
    acc=k_fold_val( VotingClassifier(estimators=[('lr', clf1), ('SVM', clf2), ('knn', clf3)],voting='hard')     , X, Y, 10)      #VotingClassifier(estimators=[('lr', clf1), ('SVM', clf2)],voting='hard')  
    return (acc)
 
if __name__ == '__main__':
    
    
 
    
    txt1= sys.argv[1]#'./ALL4.csv'
    #egLabelFileI = sys.argv[3]
    txt='vote/'+sys.argv[2]#'./10.txt'
    embedclassifier="SVM" #ys.argv[5] 
   
    fl=open( txt, 'w')  

    csv_data = pd.read_csv(txt1,index_col=0)
    F=csv_data.values.T

    imputer = SimpleImputer(strategy="mean")  # 使用均值填充
    F = imputer.fit_transform(F)
    print(F,F.shape) 
    
    csv_data1 = pd.read_csv(txt1,index_col=0,header=None)
    
    n3=csv_data1.iloc[0,:].values 
    print(n3)
     
    C=np.round(n3).astype(int)

  
    
    def function_RRFE( ):
        fl.write('Algorithm\tRFE\tbacc\tacc\tsp\tsn\tnum\tFeatures\n')    
        xresult1=[]
        svc = LinearSVC(penalty='l1',dual=False, C=0.1) 
        svc.fit(F, C)
        coef = svc.coef_
        Subset = np.where(coef != 0)[1]  # 获取非零系数的索引
        print("print(len(Subset))",len(Subset))
        '''
        a=clf_func("ttest",F, C)
        Subset=a[start:end]
        print(len(Subset))
        '''     
        for i in range(len(Subset)):
            xresult1.append(F[:,Subset[i]])
     
        xresult1=np.array(xresult1)
        xresult1=xresult1.T
        Subset =list(Subset )
        print(Subset)
        
        
        RFEtimes=0  
                
    
        svmSubset =[]   
        
 
        print(Subset)
        nbacc=[]
        svmacc=[]
        lracc=[]
        MaxR=10
        
        while(len(Subset)> MaxR):
            MaxRFE=random.randint(2, 20)
            if (len(Subset)> MaxRFE+5):
                #MaxRFE=random.randint(2, 50)
                np.random.seed(randomseed)
                #嵌入的分类器:LinearSVC(penalty='l2') 加随机种子        
                RFEtimes=RFEtimes+1
                an=delet(xresult1,C,MaxRFE)#delta("SVM",xresult1,C,MaxRFE) 
                w4=[]
                accN=[]
                for j in range(1,MaxRFE+1):
                    x=an[0:j]
                    print(x)
                
                    xresult2= np.delete(xresult1,x,axis=1)
                    print(xresult2.shape)
                    acc=Classifier(xresult2, C)
                
                    accN.append(-acc) 
        
                w4=np.argsort(accN)
                
        
                number=w4[0]+1
                print("number",number)
                out=an[0:number]
                print("i",RFEtimes,"out",out)
                
                                                
                                                        
                out=an[0:number]
                xresult1= np.delete(xresult1,out,axis=1)  
                print(out,RFEtimes)
                acc=Classifier(xresult2, C)
                print("w4,baccN",w4,accN)
                print("bacc",acc)
        
                                
                
                #找到需要删除的特征
        
                
                delete=[]
                for  w in range(len(out)):
                    
                    print(RFEtimes,out[w])
                    delete.append(Subset[out[w]])
                print(delete)
                
                for  d in delete:
                
                    Subset.remove(d)    
                    #del Subset[0]
                print(Subset)
            
                sep=";"
                
                svmSubset.append(list(Subset))        
        
                #nbbacc,nacc,na,nb=Classifier("NBayes",xresult1, C) 
                #nbacc.append(nacc)        
                    
                #Sub_new = map(lambda x:str(x), Subset)
        
        
                
                #fl.write('nbayes\t'+str(RFEtimes)+'\t'+str(nbbacc)+'\t'+str(nacc)+'\t'+str(na)+'\t'+str(nb)+'\t'+str(len(Subset))+'\t'+sep.join(Sub_new)+'\n')
                    
                acc=Classifier(xresult1, C) 
        
                svmacc.append(acc)
                Sub_new = map(lambda x:str(x), Subset)
            
        
                fl.write('svm\t'+str(RFEtimes)+'\t'+str(acc)+'\t'+str(len(Subset))+'\t'+sep.join(Sub_new)+'\n')
                
                #lbacc,lacc,la,lb=Classifier("LR",xresult1, C) 
                
                #Sub_new = map(lambda x:str(x), Subset)
                #lracc.append(lacc)
                
                #fl.write('lr\t'+str(RFEtimes)+'\t'+str(lbacc)+'\t'+str(lacc)+'\t'+str(la)+'\t'+str(lb)+'\t'+str(len(Subset))+'\t'+sep.join(Sub_new)+'\n')
        index=[]       
        for i in range (len(svmacc)):
                
            if max(svmacc)==svmacc[i] in svmacc:
                index.append(i)
        ind=index[-1]   
            #fl.write('svm\t'+str(RFEtimes)+'\t'+str(sbacc)+'\t'+str(sacc)+'\t'+str(sa)+'\t'+str(sb)+'\t'+str(len(Subset))+'\t'+sep.join(Sub_new)+'\n')
             
        fl.write(str(svmSubset[ind])+'\n')      
        print(svmSubset[ind],"ind")
      
        fl.write('\nsvm\t'+str(max(svmacc)))   
        return  svmSubset[ind]                 
         
    Subset1=function_RRFE( ) 
    print("Subset1",Subset1)
