# TSDR

TSDR: Two-Step Method for Drug Repurposing Based on Enhancing Drug features
Abstract
With the emergence of new diseases and the need for their treatment, the need to discover and develop new drugs has increased. While traditional drug discovery and development is a costly, time-consuming, and sometimes risky process. So, computational methods have been considered widely for drug repurposing tasks. In this study, a novel deep learning-based architecture for drug-disease association prediction is presented. The proposed model (TSDR) has two steps. In the first step, each subnetwork extracts a compact feature vector for the input drug-disease pair. For each input pair drug-disease, the initial drug features are constructed solely based on several primary drug datasets to assess separately the role of each initial drug dataset in the prediction of drug-disease association. In the second step, the individual drug feature vectors are combined and enhanced by NLP-based drug features to contract the final features of a drug. The enhanced drug vector beside a disease feature vector is used as input for training another convolutional neural network to predict the drug-disease association link. The final prediction is obtained at the output of the second step of the model. The experimental results show the efficiency of the proposed model in extracting a more efficient feature vector and improving the accuracy of the drug repurposing task so that the proposed method has achieved an accuracy of 96.34 in the AUC and 96.89 in the AUPR criterion.
