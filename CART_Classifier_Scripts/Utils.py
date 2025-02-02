
# Import necessary libraries
import sklearn
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from lifelines.statistics import logrank_test
from sksurv.nonparametric import kaplan_meier_estimator

# Function to build cross-validation folds using StratifiedGroupKFold
def buildfolds(Y_trainval,trainvaldatapid):
    """
    Build cross-validation folds using StratifiedGroupKFold to maintain the distribution of patients with progression (Y_trainval)
    across different folds while considering the patient groupings.

    Args:
        Y_trainval (array-like): Array of target values for the training/validation set.
        trainvaldatapid (array-like): Array of patient IDs to ensure stratification by patient.

    Returns:
        tuple: A tuple containing two lists:
            - trainfoldinds: List of indices for each fold's training data.
            - valfoldinds: List of indices for each fold's validation data.
    """
    
    print("Building Cross Validation folds")
    trainfoldinds=[]
    valfoldinds=[]
    # Initialize StratifiedGroupKFold with 10 splits
    sgkf=StratifiedGroupKFold(n_splits=10, random_state=1, shuffle=True)
    # Generate stratified folds
    for i, (train_index, test_index) in enumerate(sgkf.split(np.zeros(len(Y_trainval)), Y_trainval,trainvaldatapid)):

        trainfoldinds.append(train_index)
        valfoldinds.append(test_index)

    return(trainfoldinds,valfoldinds)

# Function to generate a reference table for recurrence analysis
def producereftable(data,model,features):
    """
    Generate a reference table for survival analysis by making detections based on the provided features and model.
    
    Args:
        data (pandas.DataFrame): The dataset containing the patient information.
        model (sklearn model): The trained model to generate predictions for recurrence.
        features (list): List of feature columns to use for prediction.

    Returns:
        tuple: A tuple containing two DataFrames:
            - reftable: A reference table containing model proposed detections, and true event information.
            - tot_df: A summary table of detections and events per patient.
    """
    # Make model propsed detections on data
    X=data[features].values
    y_pred=model.predict(X)
    
    # Create a dataframe for reference table
    reftable=pd.DataFrame(columns=['GUPatientID','DaysAfterDiagnosis','detection','event'])
    reftable['GUPatientID']=data.GUPatientID.values
    reftable['DaysAfterDiagnosis']=data['DaysAfterDiagnosis'].values
    reftable['event']=data['LineofTherapyLabel'].values
    reftable['detection']=y_pred
    
    # Aggregate detections and events for each patient
    reftable=reftable.rename(columns={'GUPatientID': 'PatientID', 'DaysAfterDiagnosis': 'time'})
    tot_df=reftable[['PatientID','event','detection']].groupby('PatientID').sum()
    return(reftable,tot_df)

# Function to downsample non-progression events in the data
def downsamplenonrecsegments(X_train,Y_train,wantedfractionforrecdiagnoses):
    """
    Downsample non-recurrence segments in the training data to achieve the desired fraction of recurrence diagnoses.
    
    Args:
        X_train (array-like): Features of the training data.
        Y_train (array-like): Labels (1 for recurrence, 0 for non-recurrence) of the training data.
        wantedfractionforrecdiagnoses (float): Desired fraction of recurrence diagnoses in the training set.

    Returns:
        numpy.ndarray: A shuffled array of indices combining recurrence and non-recurrence samples.
    """
    totalrec=sum(Y_train)
    numnonrec=math.ceil(totalrec/wantedfractionforrecdiagnoses-totalrec)
    nonrecinds=np.random.choice(np.where(Y_train==0)[0],numnonrec)
    recinds=np.where(Y_train==1)[0]
    sampleids=np.append(nonrecinds, recinds)
    np.random.shuffle(sampleids)
    return(sampleids)

# Function to calculate TTNT survival times for patients
def producerecfreesurvivaltimes(recinstance,survdata,chartreview,tdf):
    
    """
    Calculate TTNT survival times for patients, considering the recurrence instance.

    Args:
        recinstance (int): The recurrence instance to analyze (1 for first recurrence, 2 for second, etc.).
        survdata (pandas.DataFrame): The survival data for the patients.
        chartreview (bool): Whether the data is from chart review (True) or model detection-based (False).
        tdf (pandas.DataFrame): A summary table of the number of true events and detections per patient.

    Returns:
        tuple: A tuple containing two arrays:
            - recstatus: Boolean array indicating whether the patient had recurrence.
            - dad: Array of time-to-next-treatment for each patient.
    """

    rec=recinstance
    
    # Choose column based on type on if the results are from chart rveiw or model detection
    if chartreview==True:
        chrev_or_det='event'
    else:chrev_or_det='detection'
    
    # Filter survival data for patients with the required number of recurrences/detections
    survdata1=survdata[survdata.PatientID.isin(tdf[tdf[chrev_or_det]>=(rec-1)].index.tolist())]
    recstatus=np.zeros(len(survdata1.PatientID.unique())) # Initialize recurrence status array
    dad=np.zeros(len(survdata1.PatientID.unique())) # Initialize time-to-next-treatment (TTNT) array
    
    # Iterate over unique patients in filtered data
    n=0
    for i in survdata1.PatientID.unique():
        df=[]
        df=survdata1[survdata1.PatientID==i]
        df=df.reset_index(drop=True)
        #assign dad (time of censorship) as last day in patients treatment
        dad[n]=df['time'].values[-1]
        
        # if we are looking for recuurences greater than 1, censor at time of last treatment relative to time of last true event
        if rec>1:
            ind=df[df[chrev_or_det]==1].index[rec-2]
            dad[n]=df['time'].values[-1]-df['time'].values[ind]
            
        # check to see if patient has number of recs designated by recinstance    
        if sum(df[chrev_or_det].values)>=rec:
            ind=df[df[chrev_or_det]==1].index[rec-1]
            #set recurrence status of patient for this recinstance
            recstatus[n]=1
            
            # for first recurrence set survival time relative to the start of treatment for the patient for TTNT.
            dad[n]=df['time'].values[ind]-df['time'].values[0]
            if rec>1:
                ind1=df[df[chrev_or_det]==1].index[rec-2]
                # for recurrence/progression folliwng after the first, set survival timer relative to the previous start of treatment for the previous instnce of the cancer for TTNT.
                dad[n]=df['time'].values[ind]-df['time'].values[ind1]
        n+=1
    return( recstatus.astype(bool),dad)

def getsurvcurve(Leedsreftable,Leedstotdf,Flatreftable,Flattotdf,trainedon,modtype):
    """
    Generate survival curves for different recurrence instances and compare datasets using a log-rank test.

    Args:
        Leedsreftable (pandas.DataFrame): Reference table for Leeds dataset.
        Leedstotdf (pandas.DataFrame): Summary table of the number of true events and detections per patient in Leeds.
        Flatreftable (pandas.DataFrame): Reference table for in Flatiron dataset.
        Flattotdf (pandas.DataFrame): Summary table of the number of true events and detections per patient Flatiron.
        trainedon (str): Dataset the model is trained on ('Leeds' or 'Flatiron').
        modtype (str): Model type (e.g., 'DecTree', 'XGB').

    Returns:
        pandas.DataFrame: A DataFrame with the log-rank test statistics and p-values.
    """
    
    if trainedon=='Leeds':
        testedon='Flatiron'
    else:testedon='Leeds'
    fig, ax = plt.subplots(nrows=3, ncols=1,figsize=(15, 20))

    fig.tight_layout(pad=10, h_pad=5, w_pad=None, rect=None)
    logrankdf=pd.DataFrame(columns=['Statistic','P-value'])
    z=[]
    p=[]
    for i in range(3):
        # Generate survival times for both datasets and recurrence instances
        Leedscr_recstatus,Leedscr_dad=producerecfreesurvivaltimes(i+1,Leedsreftable,True,Leedstotdf)
        Leedscr_time, Leedscr_survival_prob, Leedscr_conf_int = kaplan_meier_estimator(
        Leedscr_recstatus, Leedscr_dad, conf_type="log-log")

        Flatcr_recstatus,Flatcr_dad=producerecfreesurvivaltimes(i+1,Flatreftable,True,Flattotdf)
        Flatcr_time, Flatcr_survival_prob, Flatcr_conf_int = kaplan_meier_estimator(
        Flatcr_recstatus, Flatcr_dad, conf_type="log-log")

        Leedsmod_recstatus,Leedsmod_dad=producerecfreesurvivaltimes(i+1,Leedsreftable,False,Leedstotdf)

        Flatmod_recstatus,Flatmod_dad=producerecfreesurvivaltimes(i+1,Flatreftable,False,Flattotdf)
        
        # Perform log-rank test
        if trainedon=='Leeds':
            res=logrank_test(Flatcr_dad,Flatmod_dad,Flatcr_recstatus,Flatmod_recstatus)
        else:
            res=logrank_test(Leedscr_dad,Leedsmod_dad,Leedscr_recstatus,Leedsmod_recstatus)
        z.append(res.test_statistic)
        p.append(res.p_value)
        
        if any(Leedscr_recstatus):
            ax[i].step(Leedscr_time/30, Leedscr_survival_prob, where="post",lw=3)
            ax[i].fill_between(Leedscr_time/30, Leedscr_conf_int[0], Leedscr_conf_int[1], alpha=0.25, step="post")
        if any(Flatcr_recstatus):
            ax[i].step(Flatcr_time/30, Flatcr_survival_prob, where="post",lw=3)
            ax[i].fill_between(Flatcr_time/30, Flatcr_conf_int[0], Flatcr_conf_int[1], alpha=0.25, step="post")
        if any(Leedsmod_recstatus):
            Leedsmod_time, Leedsmod_survival_prob, Leedsmod_conf_int = kaplan_meier_estimator(
            Leedsmod_recstatus, Leedsmod_dad, conf_type="log-log")
            ax[i].step(Leedsmod_time/30,Leedsmod_survival_prob, where="post",lw=3)
        if any(Flatmod_recstatus):
            Flatmod_time, Flatmod_survival_prob, Flatmod_conf_int = kaplan_meier_estimator(
            Flatmod_recstatus, Flatmod_dad, conf_type="log-log")
            ax[i].step(Flatmod_time/30,Flatmod_survival_prob, where="post",lw=3)
            ax[i].set_ylim(0, 1)
            ax[i].set_xticks([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36])
            ax[i].set_yticks([0, 0.25, 0.5, 0.75,1])
            ax[i].set_xlim(0,25)
            ax[i].tick_params(axis='both', which='major', labelsize=20)
            if i==2:
                if trainedon=='Leeds':
                    ax[2].legend(["Leeds Curated\nChemotherapy Response Dates","Leeds 95% CI","Flatiron Imputed\nChemotherapy Response Dates","Flatiron 95% CI",""+modtype+" Trained on Leeds Applied to Leeds",""+modtype+" Trained on Leeds Applied to Flatiron"],fontsize=15,bbox_to_anchor=(1.05, -0.2),ncol=3,handlelength=1)
                else:ax[2].legend(["Leeds Curated\nChemotherapy Response Dates","Leeds 95% CI","Flatiron Imputed\nChemotherapy Response Dates","Flatiron 95% CI",""+modtype+" Trained on Flatiron Applied to Leeds",""+modtype+" Trained on Flatiron Applied to Flatiron"],fontsize=15,bbox_to_anchor=(1.05, -0.2),ncol=3,handlelength=1)
            
            if (i+1)==1:
                ax[i].set_xlabel("Time Since Start of Treatment (Months)",size=25)
                ax[i].set_ylabel("Probability of No Treatment\nfor First Progression",size=25)
            else:
                text=["First","Second","Third","Fourth","Fifth","Sixth","Seventh","Eighth"]
                ax[i].set_xlabel("Time Since Start of Treatment for "+ text[i-1]+" Progression (Months)",size=25)
                ax[i].set_ylabel(" Probability of No Treatment\nfor "+text[i]+" Progression",size=25)
            ax[i].tick_params(axis='both', which='major', labelsize=15)
    ax[0].tick_params(axis='both', which='major', labelsize=15)
    ax[1].tick_params(axis='both', which='major', labelsize=15)
    ax[2].tick_params(axis='both', which='major', labelsize=15)

    logrankdf['Statistic']=z
    logrankdf['P-value']=p
    return(logrankdf)