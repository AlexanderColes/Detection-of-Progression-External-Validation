# Detection-of-Progression-in-Flatiron-Data

This project aims to identify proxy dates for cancer progression and recurrence in Epithelial ovarian cancer patients.
It buils on our previous work[1]. 

In this project we intend to train models on LTHT data and evaluate their performance on Flatiron data.

The notebooks within the Pre-processing file are numbered in order to filter and assemble the Flatiron Cohort.

The CART Classifer scripts train CART models to identfiy the the susbequent chemotherapy treatments to a progression/recurrence diagnosis.The performance of models is evaluated using classical and soft metrics[2]. TTNT suvrvial curves are also produced along with log-rank tests scores comparing model generated and ground truth survival curves.

#[1] Coles, A.D., McInerney, C.D., Zucker, K., Cheeseman, S., Johnson, O.A. and Hall, G., 2024. Evaluation of machine learning methods for the retrospective detection of ovarian cancer recurrences from chemotherapy data. ESMO Real World Data and Digital Oncology, 4, p.100038.

#[2] Salles, R., Lima, J., Reis, M., Coutinho, R., Pacitti, E., Masseglia, F., Akbarinia, R., Chen, C., Garibaldi, J., Porto, F. and Ogasawara, E., 2024. SoftED: Metrics for soft evaluation of time series event detection. Computers & Industrial Engineering, 198, p.110728.

