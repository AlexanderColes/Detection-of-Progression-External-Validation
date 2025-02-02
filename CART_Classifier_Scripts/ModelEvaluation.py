import random
import sklearn
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np

import SoftED

def percrecwithin60days(reftable):
    
    """
    Calculates percentage of model indentified dates that are within 60 days of a true date in positively identifed patients.
    Also calculates the F1, PPV and Sensitivity of identifying patients with a progression using model identified dates of progression to infer that a patient has progressed.

    Args:
        reftable (pd.Dataframe): Reference dataframe inlcudng the true label and model label of each date of eahc patients treatment in the test dataset.

    Returns:
    perc (float): Percentage of model indentified dates that are within 60 days of a true date in positively identifed patients.
    f1 (float):F1 of identifying patients with a progression using modei identified dates of progression to infer that a patient has progressed
    sens (float): Sensitivity of identifying patients with a progression using modei identified dates of progression to infer that a patient has progressed
    ppv (float): Postive Predictive Value of identifying patients with a progression using modei identified dates of progression to infer that a patient has progressed
    """
    detswithin60=0
    ndets=0
    tp=0
    tn=0
    fn=0
    fp=0
    for i in reftable.PatientID.unique():
        truerectimes=np.array(reftable[(reftable.PatientID==i)&(reftable.event==1)].time)
        detrectimes=np.array(reftable[(reftable.PatientID==i)&(reftable.detection==1)].time)
        if (any(truerectimes)&any(detrectimes)):
            tp+=1
        elif any(truerectimes)&(any(detrectimes)==False):
            fn+=1
        elif (any(truerectimes)==False)&(any(detrectimes)):
            fp+=1
        else:
            tn+=1
        if any( truerectimes)&any(detrectimes):
            detswithin60+=sum([any([(i>=j-60)&(i<=j+60) for j in truerectimes]) for i in detrectimes])
            ndets+=len(detrectimes)
    perc=0
    if ndets>0:
        perc=detswithin60/ndets*100
    elif ndets==0:
        perc=0
    ppv=0    
    if (tp+fp)>0:
        ppv=tp/(tp+fp)
    sens=tp/(tp+fn)
    f1=0
    if ((sens+ppv))>0:
        f1=2*sens*ppv/(sens+ppv)
    elif (sens+ppv)==0:
        f1=0
    return(perc,f1,sens,ppv)

def truetestmetric(testinputdata,Y_test,testpids,model):
    
    """
    Calculates evaluation metrics on whole test set 

    Args:
    testinputdata (array): Whole training dataset input data
    Y_test (array): Whole training dataset labels
    testpids(array) : Patient ID of each date in input data
    model (object): Machine Learning Model

    Returns:
     F1 (float): F1 statistic of model identifying correct progression dates.
     Sensitivity (float): Sensitivity statistic of model identifying correct progression dates.
     PPV (float): PPV statistic of model identifying correct progression dates.
     softF1k14 (float): Soft F1 statistic of model identifying progression dates within a k=14 day tolerance window.
     softSensitivityk14 (float): Soft Sensitivity statistic of model identifying progression dates within a k=14 day tolerance window.
     softPPVk14 (float): Soft PPV statistic of model identifying progression dates within a k=14 day tolerance window.
     softF1k30 (float): Soft F1 statistic of model identifying progression dates within a k=30 day tolerance window.
     softSensitivityk30 (float): Soft Sensitivity statistic of model identifying progression dates within a k=30 day tolerance window.
     softPPVk30 (float): Soft PPV statistic of model identifying progression dates within a k=30 day tolerance window.
     softF1k60 (float): Soft F1 statistic of model identifying progression dates within a k=60 day tolerance window.
     softSensitivityk60 (float): Soft Sensitivity statistic of model identifying progression dates within a k=60 day tolerance window.
     softPPVk60 (float): Soft PPV statistic of model identifying progression dates within a k=60 day tolerance window.
     percwithin60 (float): Percentage of model indentified dates that are within 60 days of a true date in positively identifed patients.
     f1patient (float): F1 of identifying patients with a progression using modei identified dates of progression to infer that a patient has progressed
     senspateint (float): Sensitivity of identifying patients with a progression using modei identified dates of progression to infer that a patient has progressed
     ppvpatient (float): Postive Predictive Value of identifying patients with a progression using modei identified dates of progression to infer that a patient has progressed
     firstrecsidentified (float): Percentage of patients first progression dates identified correctly.
     softfirstrecsk14 (float): Percentage of patients first progression dates identified within 14 days of true date.
     softfirstrecsk30 (float): Percentage of patients first progression dates identified within 30 days of true date.
     softfirstrecsk60 (float): Percentage of patients first progression dates identified within 60 days of true date.

    """

    y_pred=model.predict(testinputdata)

    reftable=pd.DataFrame(columns=['GUPatientID','DaysAfterDiagnosis','detection','event'])
    reftable['GUPatientID']=testpids
    reftable['DaysAfterDiagnosis']=testinputdata[:,0]
    reftable['detection']=y_pred
    reftable['event']=Y_test

    reftable=reftable.rename(columns={'GUPatientID': 'PatientID', 'DaysAfterDiagnosis': 'time'})

    summedtable=reftable[reftable.event==1].groupby('PatientID')[['PatientID','time','event','detection']].transform("first").drop_duplicates().sum()
    firstrecsidentified=summedtable.iloc[3]/summedtable.iloc[2]*100
    
    percwithin60,f1patient,senspateint,ppvpatient=percrecwithin60days(reftable)
    
    F1=f1_score(reftable.event,reftable.detection)
    Sensitivity=sklearn.metrics.recall_score(reftable.event,reftable.detection)
    PPV=sklearn.metrics.precision_score(reftable.event,reftable.detection)

    softF1k14,softSensitivityk14,softPPVk14,softfirstrecsk14=SoftED.caclulatesofts(reftable,14)
    softF1k30,softSensitivityk30,softPPVk30,softfirstrecsk30=SoftED.caclulatesofts(reftable,30)
    softF1k60,softSensitivityk60,softPPVk60,softfirstrecsk60=SoftED.caclulatesofts(reftable,60)
    
    softfirstrecsk14=softfirstrecsk14/summedtable.iloc[2]*100
    softfirstrecsk30=softfirstrecsk30/summedtable.iloc[2]*100
    softfirstrecsk60=softfirstrecsk60/summedtable.iloc[2]*100
    
    return(F1,Sensitivity,PPV,softF1k14,softSensitivityk14,softPPVk14,softF1k30,softSensitivityk30,softPPVk30,softF1k60,softSensitivityk60,softPPVk60,percwithin60,f1patient,senspateint,ppvpatient,firstrecsidentified,softfirstrecsk14,softfirstrecsk30,softfirstrecsk60)


def empiricalbootcis(metric,bootmetricrange):
    """
    Calculates the upper and lower 95% empirical CIs for respective bootsrapped metrics

    Args:
   metric (float): The test statistics from the model performing on the whole test set.
   bootmetricrange (array): The sorted array of metrics from the model performing on bootsrapped test set.
    
    Returns:
     metric (lower CI,upper CI)(str): The test statisticform the model on the whole test set followed by the  95% empirical CIs. 
    """
    lower=2*metric-(bootmetricrange[974])
    upper=2*metric-(bootmetricrange[25])
    return(str(round(metric,3))+' ('+str(round(lower,3))+', '+str(round(upper,3))+')')

def bootmetrics(testinput,Y_test,testids,testpids,model,n):
    
    """
    Calculates evaluation metrics on n itreation of bootstrapped test set

    Args:
    testinputdata (array): Whole training dataset input data
    Y_test (array): Whole training dataset labels
    testids (list): List of unique patientIDs in the test set
    testpids(array) : Patient ID of each date in input data
    model (object): Machine Learning Model
    n (int): Number of iterations for bootstrapping

    Returns:
     F1 (float): Sorted ascending F1 statistics of model identifying correct progression dates in bootsrapped test sets.
     Sensitivity (float): Sorted ascending  Sensitivity statistic of model identifying correct progression dates in bootsrapped test sets.
     PPV (float): Sorted ascending PPV statistic of model identifying correct progression dates in bootsrapped test sets.
     softF1k14 (float): Sorted ascending Soft F1 statistic of model identifying progression dates within a k=14 day tolerance window in bootsrapped test sets.
     softSensitivityk14 (float): Sorted ascending Soft Sensitivity statistic of model identifying progression dates within a k=14 day tolerance window in bootsrapped test sets.
     softPPVk14 (float): Sorted ascending Soft PPV statistic of model identifying progression dates within a k=14 day tolerance window in bootsrapped test sets.
     softF1k30 (float): Sorted ascending  Soft F1 statistic of model identifying progression dates within a k=30 day tolerance window in bootsrapped test sets.
     softSensitivityk30 (float): Sorted ascending Soft Sensitivity statistic of model identifying progression dates within a k=30 day tolerance window in bootsrapped test sets.
     softPPVk30 (float): Sorted ascending Soft PPV statistic of model identifying progression dates within a k=30 day tolerance window in bootsrapped test sets.
     softF1k60 (float): Sorted ascending Soft F1 statistic of model identifying progression dates within a k=60 day tolerance window in bootsrapped test sets.
     softSensitivityk60 (float): Sorted ascending Soft Sensitivity statistic of model identifying progression dates within a k=60 day tolerance window in bootsrapped test sets.
     softPPVk60 (float): Sorted ascending Soft PPV statistic of model identifying progression dates within a k=60 day tolerance window in bootsrapped test sets.
     percwithin60 (float): Sorted ascending Percentage of model indentified dates that are within 60 days of a true date in positively identifed patients in bootsrapped test sets.
     f1patient (float): Sorted ascending F1 of identifying patients with a progression using modei identified dates of progression to infer that a patient has progressed in bootsrapped test sets
     senspateint (float): Sorted ascending Sensitivity of identifying patients with a progression using modei identified dates of progression to infer that a patient has progressed in bootsrapped test sets 
     ppvpatient (float): Sorted ascending Postive Predictive Value of identifying patients with a progression using modei identified dates of progression to infer that a patient has progressed in bootsrapped test sets
     firstrecsidentified (float): Sorted ascending Percentage of patients first progression dates identified correctly in bootsrapped test sets.
     softfirstrecsk14 (float): Sorted ascending Percentage of patients first progression dates identified within 14 days of true date in bootsrapped test sets.
     softfirstrecsk30 (float): Sorted ascending Percentage of patients first progression dates identified within 30 days of true date in bootsrapped test sets.
     softfirstrecsk60 (float): Sorted ascending Percentage of patients first progression dates identified within 60 days of true date in bootsrapped test sets.

    """
    F1s=np.zeros(n)
    Sensitivities=np.zeros(n)
    PPVs=np.zeros(n)
    FirstPercs=np.zeros(n)

    softF1sk14=np.zeros(n)
    softSensitivitiesk14=np.zeros(n)
    softPPVsk14=np.zeros(n)
    softF1sk30=np.zeros(n)
    softSensitivitiesk30=np.zeros(n)
    softPPVsk30=np.zeros(n)
    softF1sk60=np.zeros(n)
    softSensitivitiesk60=np.zeros(n)
    softPPVsk60=np.zeros(n)
    softfirstrecsidentifiedk14=np.zeros(n)
    softfirstrecsidentifiedk30=np.zeros(n)
    softfirstrecsidentifiedk60=np.zeros(n)
    
    percwithin60s=np.zeros(n)
    f1patients=np.zeros(n)
    senspateints=np.zeros(n)
    ppvpatients=np.zeros(n)

    y_pred=model.predict(testinput)

    reftable=pd.DataFrame(columns=['GUPatientID','DaysAfterDiagnosis','detection','event'])
    reftable['GUPatientID']=testpids
    reftable['DaysAfterDiagnosis']=testinput[:,0]
    reftable['detection']=y_pred
    reftable['event']=Y_test
    reftable=reftable.rename(columns={'GUPatientID': 'PatientID', 'DaysAfterDiagnosis': 'time'})
   
    random.seed(1)
    for i in range(n):
        bootids=random.choices(list(testids),k=len(testids))
        inds=[i for i in range(len(testpids)) if testpids[i] in list(set(bootids))]
        
        summedtable=reftable.loc[inds][reftable.loc[inds].event==1].groupby('PatientID')[['PatientID','time','event','detection']].transform("first").drop_duplicates().sum()
        FirstPercs[i]=summedtable.iloc[3]/summedtable.iloc[2]*100

        percwithin60s[i],f1patients[i],senspateints[i],ppvpatients[i]=percrecwithin60days(reftable.loc[inds])

        F1s[i]=f1_score(reftable.loc[inds].event,reftable.loc[inds].detection)
        Sensitivities[i]=sklearn.metrics.recall_score(reftable.loc[inds].event,reftable.loc[inds].detection)
        PPVs[i]=sklearn.metrics.precision_score(reftable.loc[inds].event,reftable.loc[inds].detection)

        softF1k14,softSensitivityk14,softPPVk14,softfirstrecsk14=SoftED.caclulatesofts(reftable.loc[inds],14)
        softF1k30,softSensitivityk30,softPPVk30,softfirstrecsk30=SoftED.caclulatesofts(reftable.loc[inds],30)
        softF1k60,softSensitivityk60,softPPVk60,softfirstrecsk60=SoftED.caclulatesofts(reftable.loc[inds],60)
                                        
        softfirstrecsidentifiedk14[i]=softfirstrecsk14/summedtable.iloc[2]*100
        softfirstrecsidentifiedk30[i]=softfirstrecsk30/summedtable.iloc[2]*100
        softfirstrecsidentifiedk60[i]=softfirstrecsk60/summedtable.iloc[2]*100

        softF1sk14[i]=softF1k14
        softSensitivitiesk14[i]=softSensitivityk14
        softPPVsk14[i]=softPPVk14
        softF1sk30[i]=softF1k30
        softSensitivitiesk30[i]=softSensitivityk30
        softPPVsk30[i]=softPPVk30
        softF1sk60[i]=softF1k60
        softSensitivitiesk60[i]=softSensitivityk60
        softPPVsk60[i]=softPPVk60
        if i % (100) == 0:
                print(i / 1000 * 100, '% through Bootstrapping')


    return(sorted(F1s),sorted(Sensitivities),sorted(PPVs),
           sorted(softF1sk14),sorted(softSensitivitiesk14),sorted(softPPVsk14),
           sorted(softF1sk30),sorted(softSensitivitiesk30),sorted(softPPVsk30),
           sorted(softF1sk60),sorted(softSensitivitiesk60),sorted(softPPVsk60),
           sorted(percwithin60s),sorted(f1patients),sorted(senspateints),sorted(ppvpatients),
           sorted(FirstPercs),sorted(softfirstrecsidentifiedk14),sorted(softfirstrecsidentifiedk30),
           sorted(softfirstrecsidentifiedk60)
          )

