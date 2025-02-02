import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn import tree
from sklearn.metrics import f1_score
from Utils import downsamplenonrecsegments

def crossvalxgboost(combinations,X_trainval,Y_trainval,trainfoldinds,valfoldinds):
    """
    Performs Cross Validation of XGBoost model.

    Args:
        combinations (list): list of possible combinations of XGB n_estimators, max_depth, min_child_weight, max_delta_step, learning_rate, divsor for desired ratio of positive to non-positive samples, colsample_bynode.
        X_trainval (array): Input data of the training and vaidation folds
        Y_trainval (array): Labels of training and vaidation folds
        trainfoldinds (list): Indexes of input data and labels for each of the training folds to be used in each iteration of cross-validation
        valfoldinds (list): Indexes of input data and labels for each of the validation folds to be used in each itration of cross-validation

    Returns:
        np.array: Average F1 across validation folds for each combination of parameters for the XGB model
    """
    #print("Starting Cross Validation")
    #best_val_loss=float(inf)
    AverageF1=np.zeros(len(combinations))

    for i,(n_est,m_depth,m_cw,m_dst,lr,trainrecratio,colsamplerationode) in enumerate(combinations):
        F1forfold=np.zeros(len(trainfoldinds))
        if i%(round(len(combinations)/10))==0:
            print(i/len(combinations)*100,'% through Parameter Testing')
        for j in range(len(trainfoldinds)):
            #best_val_loss=float(inf)
            X_train=X_trainval[trainfoldinds[j]]
            Y_train=Y_trainval[trainfoldinds[j]]
            X_val=X_trainval[valfoldinds[j]]
            Y_val=Y_trainval[valfoldinds[j]]

            if trainrecratio > 0:
                trr=1/trainrecratio
                sampleids=downsamplenonrecsegments(X_train,Y_train,trr)
                X_train=X_train[sampleids,:]
                Y_train=Y_train[sampleids].astype(int)
            eval_set=[(X_val,Y_val.astype(int))]
            
            model = XGBClassifier(n_estimators=n_est,max_depth=m_depth,min_child_weight=m_cw,max_delta_step=m_dst,learning_rate=lr,colsample_bynode=colsamplerationode)
            model.fit(X_train, Y_train)
            break
            y_pred = model.predict(X_val)
            predictions = [round(value) for value in y_pred]

            F1= f1_score(Y_val, y_pred )
            F1forfold[j]=F1
        AverageF1[i]=np.mean(F1forfold)
    return(AverageF1)



def trainXGBoost(X_train,Y_train,n_est,m_depth,m_cw,m_dst,lr,trainrecratio,colsamplerationode):
    """
    Trains XGBoost model with input parameter values.

    Args:
        X_train (array): Whole training dataset input data 
        Y_train (array): Whole training dataset labels
        n_est (int): number of XGB estimators
        m_depth (int): max depth of estimators
        m_cw (int): max child weight
        m_dst (int): max delta step
        lr (float): learning rate
        trainrecratio (int): divsor for desired ratio of positive to non-positive samples
        colsamplerationode (float): colsample_bynode
    Returns:
        model object: Trained XGB model
    """
    if trainrecratio > 0:
        trr=1/trainrecratio
        sampleids=downsamplenonrecsegments(X_train,Y_train,trr)
        X_train=X_train[sampleids,:]
        Y_train=Y_train[sampleids]
    model = XGBClassifier(n_estimators=n_est,max_depth=m_depth,min_child_weight=m_cw,max_delta_step=m_dst,learning_rate=lr,colsample_bynode=colsamplerationode,random_state=1)
    model.fit(X_train, Y_train)
    return(model)

def crossdectree(combinationstree,X_trainval,Y_trainval,trainfoldinds,valfoldinds):
    
    """
    Performs Cross Validation of Decision Tree model.

    Args:
        combinations (list): list of possible combinations of Decision Tree max_depth, min_samples_split, min_samples_leaf, max_features, class_weight, divsor for desired ratio of positive to non-positive samples
        X_trainval (array): Input data of the training and vaidation folds
        Y_trainval (array): Labels of training and vaidation folds
        trainfoldinds (list): Indexes of input data and labels for each of the training folds to be used in each iteration of cross-validation
        valfoldinds (list): Indexes of input data and labels for each of the validation folds to be used in each itration of cross-validation

    Returns:
        np.array: Average F1 across validation folds for each combination of parameters for the Decision Tree model
    """

    print("Starting Cross Validation")
    #best_val_loss=float(inf)
    AverageF1=np.zeros(len(combinationstree))
    #(max_depth,min_samples_split,min_samples_leaf,trainrecratio)
    for i,(m_depth,m_samples_split,m_samples_leaf,max_f,cw,trainrecratio) in enumerate(combinationstree):
        F1forfold=np.zeros(len(trainfoldinds))
        if i%(round(len(combinationstree)/10))==0:
            print(i/len(combinationstree)*100,'% through Parameter Testing')
        for j in range(len(trainfoldinds)):
            #best_val_loss=float(inf)
            X_train=X_trainval[trainfoldinds[j]]
            Y_train=Y_trainval[trainfoldinds[j]]
            X_val=X_trainval[valfoldinds[j]]
            Y_val=Y_trainval[valfoldinds[j]]

            trr=sum(Y_train)/(len(Y_train))
            if trainrecratio > 0:
                trr=1/trainrecratio
                sampleids=downsamplenonrecsegments(X_train,Y_train,trr)
                X_train=X_train[sampleids,:]
                Y_train=Y_train[sampleids]
            model = tree.DecisionTreeClassifier(max_depth=int(m_depth),min_samples_split=int(m_samples_split),min_samples_leaf=int(m_samples_leaf),max_features=max_f,class_weight=cw)
            model.fit(X_train, Y_train)
            y_pred = model.predict(X_val)
            predictions = [round(value) for value in y_pred]
            #accuracy = accuracy_score(Y_te, predictions)
            #print("Accuracy: %.2f%%" % (accuracy * 100.0))
            F1= f1_score(Y_val, predictions)
            #print("Grid ",i," Fold ", j," F1: ", (F1))
                
            F1forfold[j]=F1
        AverageF1[i]=np.mean(F1forfold)
    return(AverageF1)

def trainDecTree(X_train,Y_train,m_depth,m_samples_split,m_samples_leaf,max_f,cw,trainrecratio):
    """
    Trains Decision Tree model with input parameter values.

    Args:
        X_train (array): Whole training dataset input data 
        Y_train (array): Whole training dataset labels
        m_depth (int): max_depth
        m_samples_split (int): min_samples_split
        m_samples_leaf (int): min_samples_leaf
        max_f (float): maximum fraction of features
        cw (str): balanced or default class_weight
        trainrecratio (int): divsor for desired ratio of positive to non-positive samples
    Returns:
        model object: Trained Decision Tree model
    """
    if trainrecratio > 0:
        trr=1/trainrecratio
        sampleids=downsamplenonrecsegments(X_train,Y_train,trr)
        X_train=X_train[sampleids,:]
        Y_train=Y_train[sampleids]
    model = tree.DecisionTreeClassifier(max_depth=int(m_depth),min_samples_split=int(m_samples_split),min_samples_leaf=int(m_samples_leaf),max_features=max_f,class_weight=cw)
    model.fit(X_train, Y_train)
    return(model)