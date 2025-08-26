# model-extraction-attack
**Introduction**

This is something that I made back in postgrad -
Which based on this paper: https://arxiv.org/abs/1609.02943

This MEA (Model Extraction Attack) explores against a digit recognition system trained on the MNIST dataset.

has the aim to replicate a target model (LeNet CNN) by exploiting only black-box access (model predictions) without the access to the source codes, parameters or training checkpoints.

this work also has defense mechanisms to protect against model stealing by identifying adversarial queries by introducing noise into responses.


**Project Overview**

- Objective: steal/replicate a model used in national voting system's digit recognition model.
- Method: Interact with the targeet model as a benign user -> send adversarial queries -> Infer model Behaviour -> train the attack model.
- Dataset: MNIST


**Attack Overview**
1. Target model (LeNet):
 - Simulates a black-box digit classifier used in voting system.
 - Only predictions are available, no architecture and/or parameters.


2. Attack Model (LeNet_a):
 - Designed by the attacker to mimic the target model.
 - trained using:
    1. MNIST dataset labels
    2. predicted outputs from target model

3. Training setup:
 - Learning Rates; 0.0001 - 0.001
 - Batch size: 512
 - Epochs: 5 - 50
 - Optimizer: Adam

4. Evaluation Metrics:
 - Accuracy, loss
 - Precision, recall, F1 Score
 - Confusion Matrix
 - Computational cost (time (/s))


**Result**
| Run | LR     | Epochs | Accuracy (%) | F1-Score | Time (s)  |
| --- | ------ | ------ | ------------ | -------- | --------- |
| 1   | 0.0001 | 10     | 90.94        | 0.91     | 111.17    |
| 2   | 0.0001 | 20     | 93.98        | 0.94     | 217.25    |
| 3   | 0.0005 | 30     | 97.87        | 0.98     | 319.97    |
| 4   | 0.0005 | 50     | **98.11**    | 0.98     | 590.27    |
| 5   | 0.001  | 5      | 95.94        | 0.96     | **60.17** |

tl;dr: 
1. peak performance at 98.11% accuracy with balanced cost.
2. even with limited queries, the attack model replicted the target effectively.


**Defense overview**

Defense mechanism strategy that implemented are:
 1. Query classification: Detect in-Distribution (ID) vs. Out-of-Distribution (OOD) queries using Mahalanobis distance + probability thresholding.
 2. Response Modification:
    - ID queries -> return accurate predictions.
    - OOD queries -> inject noise to mislead adversaries.

 3. Result:
    - Succesfully reduced attacker's accuracy 
    - Increased attack cost significantly
    - maintained high accuracy for benign uses' queries.

