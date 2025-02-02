# This script produces classical metrics on the models' performance in identifying the correct timings of chemotherapy treatments susbequent to progression/recurrence diagoses and calculates SoftED metrics[2]

#to quantify its performance in getting dates within 14,30 and 60 dayy of true events.


import numpy as np



def soft_score_time(trueeventstime,detectiontime,k):
    
    """
    Computes the soft score matrix between true event times and detection times,
    where the score decreases linearly based on the temporal difference, bounded by `k`.
    
    Args:
        trueeventstime (list): List of true event times.
        detectiontime (list): List of detection times.
        k (float): Tolerance window for scoring.

    Returns:
        S_d (array): Scores for each detection.
        firstrectruefalse (int): Indicator if any detection has multiple matches.
    """

    E=trueeventstime
    D=detectiontime
    m=len(trueeventstime)
    n=len(detectiontime)
    
    # Compute the soft score matrix Mu based on the temporal proximity of events
    Mu=np.column_stack([[max(min((D[i]-(E[j]-k))/k, ((E[j]+k)-D[i])/k), 0) for i in range(len(D))] for j in range(len(E))])

    #Boolean Integer (1,0) inicating whether any detection was within the k window of the first true event time.
    firstrectruefalse=int(any([any([((i[0]>=i[j]) & (i[j]!=0))for j in range(len(i))]) for i in Mu]))
    
    # Determine the true event index associated with each detection
    E_d=[[a for a, b in enumerate(Mu[i]) if b == max(Mu[i])] for i in range(n)]

    # Determine the detection index associated with each true event
    ed=[[((j in E_d[i]) and (Mu[i,j]>0)) for i in range(n)] for j in range(m)]
    D_e=[[y for y, x in enumerate(ls) if x] for ls in ed]

     # Map detection to true events. 
    d_e=[]
    for j in range(m):
        if(len(D_e[j])==0):d_e.append(np.nan)
        else:
            d_e.append(np.argmax(Mu[:,j]))

    # Compute the scores for each true event. Each True event can only have on detction score associated with it.
    S_e=np.zeros(m)
    S_e[:]=np.nan
    #print(S_e)
    for j in range(m):
        if(len(D_e[j])>0):
            S_e[j]=max(Mu[:,j])

    # Compute the scores for each detection. A detection can only be attributed to one true event. 
    S_d=np.zeros(n)
    for i in range(n):
        se=S_e[[k for k in range(len(d_e)) if d_e[k] == i]]
        S_d[i]=max(float(np.array([max(se) if any(se) else 0])),0)
        
    return(S_d,firstrectruefalse)

def metrics(TP,FP,FN,TN,firstrecs,beta=1):
    """
    Computes evaluation metrics for binary classification.

    Args:
        TP (int): True Positives.
        FP (int): False Positives.
        FN (int): False Negatives.
        TN (int): True Negatives.
        firstrecs (int): Count of detections associated with multiple true events.
        beta (float): Weight for F1-score.

    Returns:
        tuple: Evaluation metrics including sensitivity, specificity, F1-score, etc.
    """
    accuracy = (TP+TN)/(TP+FP+FN+TN)
    sensitivity=0
    if(TP+FP)>0:      
        sensitivity = TP/(TP+FN)
    specificity = TN/(FP+TN)
    prevalence = (TP+FN)/(TP+FP+FN+TN)
    detection_rate = TP/(TP+FP+FN+TN)
    detection_prevalence = (TP+FP)/(TP+FP+FN+TN)
    balanced_accuracy = (sensitivity+specificity)/2
    NPV = (specificity * (1-prevalence))/(((1-sensitivity)*prevalence) + ((specificity)*(1-prevalence)))
    PPV=0
    if (TP+FP)>0:
        PPV = TP/(TP+FP)
    F1=0
    if (sensitivity+PPV)>0:
        F1 = (1+beta**2)*PPV*sensitivity/((beta**2 * PPV)+sensitivity)

    return(TP,FP,FN,TN,accuracy, specificity,
                    prevalence, NPV,
                    detection_rate, detection_prevalence,
                    balanced_accuracy, PPV,
                    sensitivity, F1,firstrecs)

def getsoftf1(reftable,k):
    
    """
    Calculates soft variants of TP,FP,TN and FN to calualte Soft metrics that accouunt for nearn misses within acceptable tolerance window.

    Args:
        reftable (DataFrame): Table containing detection and event information.
        k (float): Tolerance window for scoring.

    Returns:
        tuple: Various evaluation metrics.
    """

    TPS=0 
    FNS=0 
    TNS=0 
    FPS=0
    firstrecs=0
    pids=[]
    events=reftable[reftable.detection==1]
    
    for a in (reftable.PatientID.unique()):
        reference1=reftable[reftable.PatientID==a]
        events1=events[events.PatientID==a]

        reference_vec=reference1.event
        detected_vec=reference1.detection

        trueeventstime=reference1[reference1.event==1].time.values
        detectiontime=events1.time.values

        t=len(reference_vec)
        m=len(trueeventstime)
        n=len(detectiontime)

        firstrectruefalse=0
        TPs=0
        FPs=0
        if ((m>0)|(n>0)):
            FPs=n
            if((m>0)&(n>0)):
                softScores,firstrectruefalse=sS_d=soft_score_time(trueeventstime,detectiontime,k)
                TPs=sum(softScores)
                FPs=sum(1-softScores)
        FNs=m-TPs
        TNs=(t-m)-FPs


        TPS+=TPs
        FNS+=FNs 
        TNS+=TNs
        FPS+=FPs
        firstrecs+=firstrectruefalse

    return(metrics(TPS,FPS,FNS,TNS,firstrecs,beta=1))

# Wrapper function to compute and return selected soft metric
def caclulatesofts(reftable,k):
    """
    Wrapper function to calculate and return key soft metrics.

    Args:
        reftable (DataFrame): Reference table containing event and detection data.
        k (float): Tolerance window for scoring.

    Returns:
        tuple: Selected soft metrics.
    """
    TP,FP,FN,TN,softaccuracy,softspecificity,softprevalence,softNPV,softdetection_rate,softdetection_prevalence,softbalanced_accuracy, softPPV,softSensitivity,softF1,softfirstrecs=getsoftf1(reftable,k)
    return(softF1,softSensitivity,softPPV,softfirstrecs)