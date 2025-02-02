# Main.py call in the Leeds and Flatiron ovarian cancer treatment tables and trains CART models to identify the subsequent chemotherapy treatments to a progression/recurrence diagosis. Building on our previous work which used the treatment response to cancer progression as a  proxy for the diagnosis when detecting the timing of progression [1].

# The script cycles through the use of different levels of feature granularity. The lowest level being just the timing of chemotherapy treatements respective to patients' diagnoses and the previous chemotherapy treatment and the number of chemotherapy treatments in the last month. The next level of feautures uses all fom the previous level and the drug grouping of the chemotherapy drugs given on each date of chemotherapy and the previous chemotherapy date (grouped by a Leeds based trained chart reviewer). Finally the third level uses the previous features and the timing of blood tests and measured values of 6 cancer markers recorded in the patients EHRs.

# In addition the scrpit can be alterd to use features generated from using a sliding window LSTM of 10 EHR event dates and a stride of 3 trained to provide probabilities of chemotherapy treatments susbequent to progression/recurrence diagnoses being within the window.

# The scripts produces classical metrics on the models' performance in identifying the correct timings of chemotherapy treatments susbequent to progression/recurrence diagoses and calculates SoftED metrics[2], to quantify its performance in getting dates within 14,30 and 60 days of true events. It also measures the performance of the models identifying patients that have had a progression by inferring that model indentified dates of proxy progression infer the progression status of a patient.

#[1] Coles, A.D., McInerney, C.D., Zucker, K., Cheeseman, S., Johnson, O.A. and Hall, G., 2024. Evaluation of machine learning methods for the retrospective detection of ovarian cancer recurrences from chemotherapy data. ESMO Real World Data and Digital Oncology, 4, p.100038.

#[2] Salles, R., Lima, J., Reis, M., Coutinho, R., Pacitti, E., Masseglia, F., Akbarinia, R., Chen, C., Garibaldi, J., Porto, F. and Ogasawara, E., 2024. SoftED: Metrics for soft evaluation of time series event detection. Computers & Industrial Engineering, 198, p.110728.

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import pickle
from xgboost import XGBClassifier

# Import custom utility modules
import Utils
import CARTCrossValidationandTraining
import ModelEvaluation
from Utils import producereftable

# Define feature sets for Leeds dataset (all features, chemotherapy drugs, and frequency-related features)
LeedsAllFeatures=['DaysAfterDiagnosis',
       'OriginatingTable_PPMQuery.ddt.ChemoCycles',
       'OriginatingTable_PPMQuery.ddt.EHRReporting',
       'EventType_CancerMarkerTest',
       'DaysSinceLast_OriginatingTable_PPMQuery.ddt.ChemoCycles',
       'DaysSinceLast_OriginatingTable_PPMQuery.ddt.EHRReporting',
       'DaysSinceLast_EventType_CancerMarkerTest','DaysSinceLastSeen','CA 125', 'CA 19.9', 'CEA', 'CA 15.3', 'LDH',
       'AFP', 'PARPi',
       'Rucaparib/Nivolumab trial', 'Hormone', 'Carbo/Gem/Bev',
       'Cisplatin/Gem/Bev', 'Carbo/Paclitax/Bev', 'Bevacizumab',
       'Cyclophos / Bev', 'Carboplatin/Paclitaxel', 'Carboplatin', 'Carbo/Gem',
       'Carboplatin/Caelyx', 'CAV', 'Cisplatin', 'Cisplatin/Etoposide',
       'Other', 'Etoposide', 'Trial drug / placebo', 'Caelyx',
       'Hormone alternating', 'Cyclophosphamide', 'Paclitaxel',
       'Paclitaxel albumin', 'Cisplatin/Paclitaxel',
       'Pembrolizumab/Lenvantinib', 'Seq Doublet', 'Topotecan', 'Trametinib',
       'Trovax vaccine','Last_PARPi', 'Last_Rucaparib/Nivolumab trial', 'Last_Hormone',
       'Last_Carbo/Gem/Bev', 'Last_Cisplatin/Gem/Bev',
       'Last_Carbo/Paclitax/Bev', 'Last_Bevacizumab', 'Last_Cyclophos / Bev',
       'Last_Carboplatin/Paclitaxel', 'Last_Carboplatin', 'Last_Carbo/Gem',
       'Last_Carboplatin/Caelyx', 'Last_CAV', 'Last_Cisplatin',
       'Last_Cisplatin/Etoposide', 'Last_Other', 'Last_Etoposide',
       'Last_Trial drug / placebo', 'Last_Caelyx', 'Last_Hormone alternating',
       'Last_Cyclophosphamide', 'Last_Paclitaxel', 'Last_Paclitaxel albumin',
       'Last_Cisplatin/Paclitaxel', 'Last_Pembrolizumab/Lenvantinib',
       'Last_Seq Doublet', 'Last_Topotecan', 'Last_Trametinib',
       'Last_Trovax vaccine','NChemoInLastMonth']
LeedsChemoDrugsFeatures=['DaysAfterDiagnosis',
       'OriginatingTable_PPMQuery.ddt.ChemoCycles',
       'DaysSinceLast_OriginatingTable_PPMQuery.ddt.ChemoCycles', 'PARPi',
       'Rucaparib/Nivolumab trial', 'Hormone', 'Carbo/Gem/Bev',
       'Cisplatin/Gem/Bev', 'Carbo/Paclitax/Bev', 'Bevacizumab',
       'Cyclophos / Bev', 'Carboplatin/Paclitaxel', 'Carboplatin', 'Carbo/Gem',
       'Carboplatin/Caelyx', 'CAV', 'Cisplatin', 'Cisplatin/Etoposide',
       'Other', 'Etoposide', 'Trial drug / placebo', 'Caelyx',
       'Hormone alternating', 'Cyclophosphamide', 'Paclitaxel',
       'Paclitaxel albumin', 'Cisplatin/Paclitaxel',
       'Pembrolizumab/Lenvantinib', 'Seq Doublet', 'Topotecan', 'Trametinib',
       'Trovax vaccine','Last_PARPi', 'Last_Rucaparib/Nivolumab trial', 'Last_Hormone',
       'Last_Carbo/Gem/Bev', 'Last_Cisplatin/Gem/Bev',
       'Last_Carbo/Paclitax/Bev', 'Last_Bevacizumab', 'Last_Cyclophos / Bev',
       'Last_Carboplatin/Paclitaxel', 'Last_Carboplatin', 'Last_Carbo/Gem',
       'Last_Carboplatin/Caelyx', 'Last_CAV', 'Last_Cisplatin',
       'Last_Cisplatin/Etoposide', 'Last_Other', 'Last_Etoposide',
       'Last_Trial drug / placebo', 'Last_Caelyx', 'Last_Hormone alternating',
       'Last_Cyclophosphamide', 'Last_Paclitaxel', 'Last_Paclitaxel albumin',
       'Last_Cisplatin/Paclitaxel', 'Last_Pembrolizumab/Lenvantinib',
       'Last_Seq Doublet', 'Last_Topotecan', 'Last_Trametinib',
       'Last_Trovax vaccine','NChemoInLastMonth']
LeedsChemoFreqFeatures=['DaysAfterDiagnosis',
       'OriginatingTable_PPMQuery.ddt.ChemoCycles',
       'DaysSinceLast_OriginatingTable_PPMQuery.ddt.ChemoCycles','NChemoInLastMonth']

# Define similar feature sets for Flatiron dataset
FlatAllFeatures=['DaysAfterDiagnosis',
              'EventType_Chemotherapy',
              'EventType_LabReport',
              'EventType_CancerMarkerTest',
             'DaysSinceLast_EventType_Chemotherapy',
              'DaysSinceLast_EventType_LabReport',
              'DaysSinceLast_EventType_CancerMarkerTest',
              'DaysSinceLastSeen','CA 125','CA 19.9', 'CEA', 'CA 15.3', 'LDH', 'AFP','PARPi',
       'Rucaparib/Nivolumab trial', 'Hormone', 'Carbo/Gem/Bev',
       'Cisplatin/Gem/Bev', 'Carbo/Paclitax/Bev', 'Bevacizumab',
       'Cyclophos / Bev', 'Carboplatin/Paclitaxel', 'Carboplatin', 'Carbo/Gem',
       'Carboplatin/Caelyx', 'CAV', 'Cisplatin', 'Cisplatin/Etoposide',
       'Other', 'Etoposide', 'Trial drug / placebo', 'Caelyx',
       'Hormone alternating', 'Cyclophosphamide', 'Paclitaxel',
       'Paclitaxel albumin', 'Cisplatin/Paclitaxel',
       'Pembrolizumab/Lenvantinib', 'Seq Doublet', 'Topotecan', 'Trametinib',
       'Trovax vaccine','Last_PARPi', 'Last_Rucaparib/Nivolumab trial', 'Last_Hormone',
       'Last_Carbo/Gem/Bev', 'Last_Cisplatin/Gem/Bev',
       'Last_Carbo/Paclitax/Bev', 'Last_Bevacizumab', 'Last_Cyclophos / Bev',
       'Last_Carboplatin/Paclitaxel', 'Last_Carboplatin', 'Last_Carbo/Gem',
       'Last_Carboplatin/Caelyx', 'Last_CAV', 'Last_Cisplatin',
       'Last_Cisplatin/Etoposide', 'Last_Other', 'Last_Etoposide',
       'Last_Trial drug / placebo', 'Last_Caelyx', 'Last_Hormone alternating',
       'Last_Cyclophosphamide', 'Last_Paclitaxel', 'Last_Paclitaxel albumin',
       'Last_Cisplatin/Paclitaxel', 'Last_Pembrolizumab/Lenvantinib',
       'Last_Seq Doublet', 'Last_Topotecan', 'Last_Trametinib',
       'Last_Trovax vaccine','NChemoInLastMonth']
FlatChemoDrugsFeatures=['DaysAfterDiagnosis',
              'EventType_Chemotherapy',
             'DaysSinceLast_EventType_Chemotherapy','PARPi',
       'Rucaparib/Nivolumab trial', 'Hormone', 'Carbo/Gem/Bev',
       'Cisplatin/Gem/Bev', 'Carbo/Paclitax/Bev', 'Bevacizumab',
       'Cyclophos / Bev', 'Carboplatin/Paclitaxel', 'Carboplatin', 'Carbo/Gem',
       'Carboplatin/Caelyx', 'CAV', 'Cisplatin', 'Cisplatin/Etoposide',
       'Other', 'Etoposide', 'Trial drug / placebo', 'Caelyx',
       'Hormone alternating', 'Cyclophosphamide', 'Paclitaxel',
       'Paclitaxel albumin', 'Cisplatin/Paclitaxel',
       'Pembrolizumab/Lenvantinib', 'Seq Doublet', 'Topotecan', 'Trametinib',
       'Trovax vaccine','Last_PARPi', 'Last_Rucaparib/Nivolumab trial', 'Last_Hormone',
       'Last_Carbo/Gem/Bev', 'Last_Cisplatin/Gem/Bev',
       'Last_Carbo/Paclitax/Bev', 'Last_Bevacizumab', 'Last_Cyclophos / Bev',
       'Last_Carboplatin/Paclitaxel', 'Last_Carboplatin', 'Last_Carbo/Gem',
       'Last_Carboplatin/Caelyx', 'Last_CAV', 'Last_Cisplatin',
       'Last_Cisplatin/Etoposide', 'Last_Other', 'Last_Etoposide',
       'Last_Trial drug / placebo', 'Last_Caelyx', 'Last_Hormone alternating',
       'Last_Cyclophosphamide', 'Last_Paclitaxel', 'Last_Paclitaxel albumin',
       'Last_Cisplatin/Paclitaxel', 'Last_Pembrolizumab/Lenvantinib',
       'Last_Seq Doublet', 'Last_Topotecan', 'Last_Trametinib',
       'Last_Trovax vaccine','NChemoInLastMonth']
FlatChemoFreqFeatures=['DaysAfterDiagnosis',
              'EventType_Chemotherapy',
             'DaysSinceLast_EventType_Chemotherapy','NChemoInLastMonth']

# Define feature sets specific to those prodcued by sliding window LSTM models
leedslstmAllFeatures=['LSTM_Trained_on_Leeds_Features_AllFeatures_Window10StrideDivisor3MaxPred',
       'LSTM_Trained_on_Leeds_Features_AllFeatures_Window10StrideDivisor3MeanPred']
flatlstmAllFeatures=['LSTM_Trained_on_Flatiron_Features_AllFeatures_Window10StrideDivisor3MaxPred_y',
       'LSTM_Trained_on_Flatiron_Features_AllFeatures_Window10StrideDivisor3MeanPred_y']
leedslstmChemoDrugsFeatures=['LSTM_Trained_on_Leeds_Features_ChemoDrugsFeatures_Window10StrideDivisor3MaxPred',
       'LSTM_Trained_on_Leeds_Features_ChemoDrugsFeatures_Window10StrideDivisor3MeanPred']
flatlstmChemoDrugsFeatures=['LSTM_Trained_on_Flatiron_Features_ChemoDrugsFeatures_Window10StrideDivisor3MaxPred',
       'LSTM_Trained_on_Flatiron_Features_ChemoDrugsFeatures_Window10StrideDivisor3MeanPred']
leedslstmChemoFreqFeatures=['LSTM_Trained_on_Leeds_Features_ChemoFreqFeatures_Window10StrideDivisor3MaxPred',
       'LSTM_Trained_on_Leeds_Features_ChemoFreqFeatures_Window10StrideDivisor3MeanPred']
flatlstmChemoFreqFeatures=['LSTM_Trained_on_Flatiron_Features_ChemoFreqFeatures_Window10StrideDivisor3MaxPred',
       'LSTM_Trained_on_Flatiron_Features_ChemoFreqFeatures_Window10StrideDivisor3MeanPred']

leedslstmfeatures=[leedslstmAllFeatures,leedslstmChemoDrugsFeatures,leedslstmChemoFreqFeatures]
flatltsmfeatures=[flatlstmAllFeatures,flatlstmChemoDrugsFeatures,flatlstmChemoFreqFeatures]

# Load datasets for Leeds and Flatiron
Leedsdata=pd.read_csv(r'\\trust.leedsth.nhs.uk\Data\Users\AColes\FlatIron_Transformed\NewLeedsFlat.csv')
Flatdata=pd.read_csv(r'\\trust.leedsth.nhs.uk\Data\Users\AColes\FlatIron_Transformed\FlatLSTMLoTData.csv')

# Define input datasets for further processing
Leedsinput=Leedsdata
Flatinput=Flatdata

# Define hyperparameter ranges for XGBoost
n_estimators = range(100, 300, 100)
max_depth = range(4, 7, 2)
min_child_weight=range(4,12,4)
trainrecratio=[0,5,20/3,10,20]
max_delta_step=range(0,6,2)
learning_rate=list(np.array(range(1,4,1))/10)
colsample_bynodelist=[0.75,1]

# Generate combinations of hyperparameters for XGBoost
xgbcombinations=list(itertools.product(n_estimators,max_depth,min_child_weight,max_delta_step,learning_rate,trainrecratio,colsample_bynodelist))
print(len(xgbcombinations))  # Print total combinations

# Define hyperparameter ranges for Decision Trees
max_depth = list(range(5, 50,10))
min_samples_split=list(range(10, 20,5))
min_samples_leaf=list(range(5, 20,3))
max_features=[0.75,None]
class_weight=['balanced',None]
trainrecratio=[0,10,20]

# Generate combinations of hyperparameters for Decision Trees
combinationstree=list(itertools.product(max_depth,min_samples_split,min_samples_leaf,max_features,class_weight,trainrecratio))
print(len(combinationstree))  # Print total combinations

# Define feature sets for Leeds and Flatiron datasets
LeedsFeatures=[LeedsAllFeatures,LeedsChemoDrugsFeatures,LeedsChemoFreqFeatures]
FlatFeatures=[FlatAllFeatures,FlatChemoDrugsFeatures,FlatChemoFreqFeatures]
Features=['AllFeatures','ChemoDrugsFeatures','ChemoFreqFeatures']
LeedsFlat=['Leeds','Flatiron']

# Initialize an empty DataFrame to store results
results=pd.DataFrame(columns=['Features','Model','Trained_On','Tested_On','F1 (CIs)','Sensitivity (CIs)','PPV (CIs)','Soft F1 k=14 (CIs)','Soft Sensitivity k=14 (CIs)','Soft PPV k=14 (CIs)','Soft F1 k=30 (CIs)','Soft Sensitivity k=30 (CIs)','Soft PPV k=30 (CIs)','Soft F1 k=60 (CIs)','Soft Sensitivity k=60 (CIs)','Soft PPV k=60 (CIs)','Percentage of Detections within 60 days of a True Event in Identified Patients (CIs)','F1 for Identifying Patients (CIs)','Sensitivity for Identifying Patients (CIs)','PPV for Identifying Patients (CIs)','Percentage of First Recurrences Correctly Identified (CIs)','Percentage of First Recurrences Correctly Identified (Within 14 days) (CIs)','Percentage of First Recurrences Correctly Identified (Within 30 days) (CIs)','Percentage of First Recurrences Correctly Identified (Within 60 days) (CIs)'])

# Initialize lists to store feature names, model details, performance metrics, etc.
FeaturesUsed=[]
ModelUsed=[]
TrainUsed=[]
TestUsed=[]

fs=[]
sens=[]
ppvs=[]
softfsk14=[]
softsensk14=[]
softppvsk14=[]
softfsk30=[]
softsensk30=[]
softppvsk30=[]
softfsk60=[]
softsensk60=[]
softppvsk60=[]
percwithin60all=[]
f1patientall=[]
senspateintall=[]
ppvpatientall=[]
firstrecsperc=[]
firstrecsperck14=[]
firstrecsperck30=[]
firstrecsperck60=[]

# Boolean flag for LSTM usage and model type (Decision Tree)
LSTM=False
#Define Model Type to be Used
ModType='DecTree' # You can change to 'XGB' for XGBoost

#Name of Model
if LSTM ==True:
    modname=ModType+'LSTM'
else: modname=ModType

#modname='DecTreeLSTM' 

# Iterate through feature combinations
for i in range(len(LeedsFeatures)):
    
    for j in LeedsFlat:
        
        # Add current feature to list
        FeaturesUsed.append(Features[i])
        # Set Leeds dataset as training daatset and Flatiron as test dataset.
        if j=='Leeds':
            print('Features_Used: ',Features[i])
            print('Trained_On_Leeds')
            TrainUsed.append('Leeds')
            TestUsed.append('Flatiron')
            
            # Extract training and test data
            Y_trainval=Leedsinput.LineofTherapyLabel.values
            trainvalpids=Leedsinput.GUPatientID.values
            testids=Flatinput.GUPatientID.unique()
            testpids=Flatinput.GUPatientID.values
            trainfoldinds,valfoldinds=Utils.buildfolds(Y_trainval,trainvalpids)
            X_trainval=Leedsinput[LeedsFeatures[i]].values
            X_test=Flatinput[FlatFeatures[i]].values
            
             # Use LSTM generated features if necessary
            if LSTM==True:
                print("using:",leedslstmfeatures[i])
                X_trainval=Leedsinput[LeedsFeatures[i]+leedslstmfeatures[i]].values
                X_test=Flatinput[FlatFeatures[i]+leedslstmfeatures[i]].values
            Y_test=Flatinput['LineofTherapyLabel'].values
            testinput=Flatinput
            
        # Set Flatiron dataset as training daatset and Leeds as test dataset.
        elif j=='Flatiron':
            print('Features_Used: ',Features[i])
            print('Trained_On_Flatiron')
            TrainUsed.append('Flatiron')
            TestUsed.append('Leeds')
            
            # Extract training and test data
            Y_trainval=Flatinput.LineofTherapyLabel.values
            trainvalpids=Flatinput.GUPatientID.values
            testids=Leedsinput.GUPatientID.unique()
            testpids=Leedsinput.GUPatientID.values
            trainfoldinds,valfoldinds=Utils.buildfolds(Y_trainval,trainvalpids)
            X_trainval=Flatinput[FlatFeatures[i]].values
            X_test=Leedsinput[LeedsFeatures[i]].values
            
            # Use LSTM generated features if necessary
            if LSTM==True:
                print("using:",leedslstmfeatures[i])
                X_trainval=Flatinput[FlatFeatures[i]+flatltsmfeatures[i]].values
                X_test=Leedsinput[LeedsFeatures[i]+flatltsmfeatures[i]].values
            Y_test=Leedsinput['LineofTherapyLabel'].values
            testinput=Leedsinput
        print(X_trainval.shape,Y_trainval.shape)
        print(X_test.shape,Y_test.shape)
        
        # Choose and train the model (either XGBoost or Decision Tree)
        
        if ModType=='XGB':
            AverageF1= CARTCrossValidationandTraining.crossvalxgboost(xgbcombinations,X_trainval,Y_trainval,trainfoldinds,valfoldinds)
            # Get the best hyperparameters
            n_estimators,max_depth,min_child_weight,max_delta_step,learning_rate,trainrecratio,colsamplerationode=xgbcombinations[np.argmax(AverageF1)]  
            #once optmised train a new model on the training set with optnised parameters and set features
            print('Trainning XGBoost Model')
            model= CARTCrossValidationandTraining.trainXGBoost(X_trainval,Y_trainval,n_estimators,max_depth,min_child_weight,max_delta_step,learning_rate,trainrecratio,colsamplerationode)
            model.save_model('LeedsFlatResults/'+modname+'model_Trained_'+ j+'_Features'+Features[i]+'2.json')
        if ModType=='DecTree':
            print('Trainning Dec Tree Model')
            AverageF1=CARTCrossValidationandTraining.crossdectree(combinationstree,X_trainval,Y_trainval,trainfoldinds,valfoldinds)
            # Get the best hyperparameters
            max_depth,m_samples_split,m_samples_leaf,max_features,class_weight,trainrecratio=combinationstree[np.argmax(AverageF1)]
            model=CARTCrossValidationandTraining.trainDecTree(X_trainval,Y_trainval,max_depth,m_samples_split,m_samples_leaf,max_features,class_weight,trainrecratio)
            pickle.dump(model, open('LeedsFlatResults/'+modname+'model_Trained_'+ j+'_Features'+Features[i]+'.json', "wb"))
            #model.save_model('LeedsFlatXGBResults/DecTreemodel_Trained_'+ j+'_Features'+Features[i]+'.json')
#print(F1)
            
        # Evaluate the model on a bootstrapped test set
        print('Bootstrapping')
        F1s,Sensitivities,PPVs,softF1sk14,softSensitivitiesk14,softPPVsk14,softF1sk30,softSensitivitiesk30,softPPVsk30,softF1sk60,softSensitivitiesk60,softPPVsk60,percwithin60s,f1patients,senspateints,ppvpatients,firstrecpercs,firstrecpercsk14,firstrecpercsk30,firstrecpercsk60=ModelEvaluation.bootmetrics(X_test,Y_test,testids,testpids,model,1000)

        # Compute test metrics for the model on the whole test data set
        print('Getting Teststatic')
        F1,Sensitivity,PPV,softF1k14,softSensitivityk14,softPPVk14,softF1k30,softSensitivityk30,softPPVk30,softF1k60,softSensitivityk60,softPPVk60,percwithin60,f1patient,senspateint,ppvpatient,firstrecperc,firstrecperck14,firstrecperck30,firstrecperck60=ModelEvaluation.truetestmetric(X_test,Y_test,testpids,model)
        
        # Compute confidence intervals for metrics
        fs.append(ModelEvaluation.empiricalbootcis(F1,F1s))
        sens.append(ModelEvaluation.empiricalbootcis(Sensitivity,Sensitivities))
        ppvs.append(ModelEvaluation.empiricalbootcis(PPV,PPVs))
        softfsk14.append(ModelEvaluation.empiricalbootcis(softF1k14,softF1sk14))
        softsensk14.append(ModelEvaluation.empiricalbootcis(softSensitivityk14,softSensitivitiesk14))
        softppvsk14.append(ModelEvaluation.empiricalbootcis(softPPVk14,softPPVsk14))
        softfsk30.append(ModelEvaluation.empiricalbootcis(softF1k30,softF1sk30))
        softsensk30.append(ModelEvaluation.empiricalbootcis(softSensitivityk30,softSensitivitiesk30))
        softppvsk30.append(ModelEvaluation.empiricalbootcis(softPPVk30,softPPVsk30))
        softfsk60.append(ModelEvaluation.empiricalbootcis(softF1k60,softF1sk60))
        softsensk60.append(ModelEvaluation.empiricalbootcis(softSensitivityk60,softSensitivitiesk60))
        softppvsk60.append(ModelEvaluation.empiricalbootcis(softPPVk60,softPPVsk60))
        percwithin60all.append(ModelEvaluation.empiricalbootcis(percwithin60,percwithin60s))
        f1patientall.append(ModelEvaluation.empiricalbootcis(f1patient,f1patients))
        senspateintall.append(ModelEvaluation.empiricalbootcis(senspateint,senspateints))
        ppvpatientall.append(ModelEvaluation.empiricalbootcis(ppvpatient,ppvpatients))
        firstrecsperc.append(ModelEvaluation.empiricalbootcis(firstrecperc,firstrecpercs))
        firstrecsperck14.append(ModelEvaluation.empiricalbootcis(firstrecperck14,firstrecpercsk14))
        firstrecsperck30.append(ModelEvaluation.empiricalbootcis(firstrecperck30,firstrecpercsk30))
        firstrecsperck60.append(ModelEvaluation.empiricalbootcis(firstrecperck60,firstrecpercsk60))

# Add computed metrics to the results dataframe
results['Features']=FeaturesUsed
results['Model']=modname
results['Trained_On']=TrainUsed
results['Tested_On']=TestUsed
results['F1 (CIs)']=fs
results['Sensitivity (CIs)']=sens
results['PPV (CIs)']=ppvs
results['Soft F1 k=14 (CIs)']=softfsk14
results['Soft Sensitivity k=14 (CIs)']=softsensk14
results['Soft PPV k=14 (CIs)']=softppvsk14
results['Soft F1 k=30 (CIs)']=softfsk30
results['Soft Sensitivity k=30 (CIs)']=softsensk30
results['Soft PPV k=30 (CIs)']=softppvsk30
results['Soft F1 k=60 (CIs)']=softfsk60
results['Soft Sensitivity k=60 (CIs)']=softsensk60
results['Soft PPV k=60 (CIs)']=softppvsk60
results['Percentage of Detections within 60 days of a True Event in Identified Patients (CIs)']=percwithin60all
results['F1 for Identifying Patients (CIs)']=f1patientall
results['Sensitivity for Identifying Patients (CIs)']=senspateintall
results['PPV for Identifying Patients (CIs)']=ppvpatientall
results['Percentage of First Recurrences Correctly Identified (CIs)']=firstrecsperc
results['Percentage of First Recurrences Correctly Identified (Within 14 days) (CIs)']=firstrecsperck14
results['Percentage of First Recurrences Correctly Identified (Within 30 days) (CIs)']=firstrecsperck30
results['Percentage of First Recurrences Correctly Identified (Within 60 days) (CIs)']=firstrecsperck60


results.to_csv('LeedsFlatResults/LeedsFlat'+modname+'Results.csv',index=False)

# Evaluate and visualize survival curves and calculate log-rank test for best models

for j in LeedsFlat:
    # Find the best model based on F1 score
    f1s=results[results.Trained_On==j]['F1 (CIs)'].values
    ind=np.argmax([np.float(m.split(' ')[0]) for m in f1s])
    feat=results[results.Trained_On==j].reset_index().loc[ind].Features
    modtype=results[results.Trained_On==j].reset_index().loc[ind].Model
    
    # Load the best model for further evaluation
    if 'XGB' in modtype:
        model = XGBClassifier()
        model.load_model('LeedsFlatResults/'+modtype+'model_Trained_'+ j+'_Features'+feat+'2.json')
    if 'DecTree' in modtype:
        model = pickle.load(open('LeedsFlatResults/'+modtype+'model_Trained_'+ j+'_Features'+feat+'.json', "rb"))
    
    # Use model on approriate test set with its features
    if LSTM==True:
        if j=='Leeds':
            Leedsreftable,Leedstotdf=producereftable(Leedsinput,model,LeedsFeatures[Features.index(feat)]+leedslstmfeatures[Features.index(feat)])
            Flatreftable,Flattotdf=producereftable(Flatinput,model,FlatFeatures[Features.index(feat)]+leedslstmfeatures[Features.index(feat)])
        else:
            Leedsreftable,Leedstotdf=producereftable(Leedsinput,model,LeedsFeatures[Features.index(feat)]+flatltsmfeatures[Features.index(feat)])
            Flatreftable,Flattotdf=producereftable(Flatinput,model,FlatFeatures[Features.index(feat)]+flatltsmfeatures[Features.index(feat)])
    else:
        Leedsreftable,Leedstotdf=producereftable(Leedsinput,model,LeedsFeatures[Features.index(feat)])
        Flatreftable,Flattotdf=producereftable(Flatinput,model,FlatFeatures[Features.index(feat)])

    # Plot survival curves and compute log-rank test
    logrankdf=Utils.getsurvcurve(Leedsreftable,Leedstotdf,Flatreftable,Flattotdf,j,modtype)
    plt.savefig('LeedsFlatResults/SurvCurveBest'+modtype+'ModelTrainedOn_'+j+'_Features_'+feat+'.jpg')
    plt.show()

    logrankdf.to_csv('LeedsFlatResults/LogRank'+modtype+'ModelTrainedOn_'+j+'_Features_'+feat+'.csv',index=False)
