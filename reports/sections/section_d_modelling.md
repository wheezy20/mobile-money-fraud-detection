# Section D: Modelling and Evaluation

## Sub-section 1: Experimental Design

Model development followed a reproducible train–validation–test framework designed for severely imbalanced classification. The filtered dataset (2,770,409 transactions) was partitioned using a stratified 70/15/15 split, producing 1,939,285 training samples, 415,562 validation samples, and 415,562 test samples. Stratification preserved the fraud rate across all partitions, with each split maintaining approximately 0.2965% fraudulent transactions. The test set was held out throughout model development and used only once for final evaluation.

Class imbalance was handled using cost-sensitive weighting rather than synthetic resampling. class_weight='balanced' was applied to scikit-learn models, while XGBoost used scale_pos_weight = 336.3, matching the observed class ratio. Class weighting was preferred over SMOTE-style resampling to avoid introducing synthetic transaction patterns and to maintain deterministic preprocessing behaviour.

Hyperparameter tuning employed 3-fold stratified cross-validation on a 25% stratified subsample of the training data to reduce computational cost while preserving class proportions. A fixed random seed (random_state = 42) was used throughout, and all dataset splits, preprocessing artefacts, and trained models were persisted to disk for reproducibility.



## Sub-section 2: Baseline Models

Three baseline classifiers were trained to compare representative model families commonly used in fraud detection. Logistic Regression was included as an interpretable linear baseline, Random Forest as a non-linear ensemble method capable of modelling feature interactions, and XGBoost as a high-performance boosted-tree approach widely used in production fraud systems. All models were trained using the same stratified training set with imbalance-aware weighting and default hyperparameters.

| Model               | PR-AUC | ROC-AUC |     F1 | Precision | Recall | Training Time (s) |
| ------------------- | -----: | ------: | -----: | --------: | -----: | ----------------: |
| Logistic Regression | 0.4230 |  0.9738 | 0.0674 |    0.0350 | 0.8945 |              25.6 |
| Random Forest       | 0.9956 |  0.9987 | 0.9879 |    0.9816 | 0.9943 |             108.9 |
| XGBoost             | 0.9800 |  0.9986 | 0.8518 |    0.7454 | 0.9935 |               8.7 |

The most important finding is that ROC-AUC failed to meaningfully distinguish between models, with all three achieving scores above 0.97 despite vastly different operational behaviour. In contrast, PR-AUC produced a wide separation ranging from 0.4230 to 0.9956 (Figure 5), empirically validating the metric-selection rationale established in Section A for severely imbalanced classification tasks.

Although Logistic Regression achieved high recall, its extremely low precision produced over 30,000 false positives on the validation set (Figure 7), making it impractical for operational deployment. This suggests that fraud behaviour in PaySim is strongly non-linear and depends on feature interactions that linear coefficients cannot adequately capture. Random Forest emerged as the strongest baseline, achieving near-perfect PR-AUC with only 23 false positives at the default threshold. XGBoost achieved comparable PR-AUC while training approximately twelve times faster, but its lower F1-score indicates that the default decision threshold was poorly calibrated for the imbalance setting. he next sub-section explores how threshold choice affects each model's operational performance.


## Sub-section 3: Evaluation Metrics and Threshold Analysis

The default classification threshold of 0.5 is a modelling convention rather than a principled operating point. In severely imbalanced fraud detection settings, predicted probabilities are typically concentrated near zero, meaning that threshold selection can substantially alter operational behaviour. Threshold analysis therefore formed a critical component of model evaluation (Figure 8).

| Model | F1 @ 0.5 Threshold | Best F1 | Optimal Threshold |
|---|---|---|---|
| Logistic Regression | 0.0674 | 0.4542 | 0.997 |
| Random Forest | 0.9879 | 0.9955 | 0.845 |
| XGBoost | 0.8518 | 0.9235 | 0.928 |

The results show that the default threshold substantially underestimated the performance of some models, particularly XGBoost, whose F1-score increased from 0.8518 to 0.9235 after threshold optimisation. In contrast, Random Forest remained highly stable across a broad threshold range, indicating strong probability calibration and robust class separation.

To better reflect operational consequences, model evaluation also incorporated cost-weighted scoring using:

$\text{Total Cost} = 5 \cdot \text{FN} + 1 \cdot \text{FP}$

where missed fraud cases (FN) were weighted five times more heavily than false alarms (FP). This reflects the asymmetry between direct financial loss from undetected fraud and customer-friction costs arising from unnecessary transaction blocking.

Consistent with the stakeholder framework established in Section A, recall at 90% precision was also evaluated as the primary operational metric. Under this criterion, Random Forest achieved 99.51% recall, well above Logistic Regression (14.85%) and XGBoost (92.45%). Consequently, Random Forest emerged as the strongest candidate for further optimisation and detailed analysis.


## Sub-section 4: Hyperparameter Tuning

Hyperparameter optimisation was performed for the Random Forest model using RandomizedSearchCV with 3-fold stratified cross-validation. The search explored 30 randomly sampled configurations from a space of 648 possible parameter combinations and optimised for average_precision (PR-AUC), the primary evaluation metric established in Section A. To reduce computational cost, tuning was conducted on a 25% stratified subsample of the training data before refitting the best configuration on the full training set.

| Parameter | Best Value |
|---|---|
| n_estimators | 300 |
| max_depth | None |
| min_samples_split | 2 |
| min_samples_leaf | 2 |
| max_features | log2 |
| class_weight | balanced_subsample |


| Metric | Baseline RF | Tuned RF |
|---|---|---|
| PR-AUC | 0.9956 | 0.9951 |
| Precision | 0.9816 | 0.9992 |
| False Positives | 23 | 1 |
| Recall | 0.9943 | 0.9943 |

Although tuning produced no meaningful improvement in PR-AUC, it substantially improved operational precision at the default threshold by reducing false positives from 23 to 1 while preserving recall. This indicates that tuning primarily shifted the classifier along the precision-recall trade-off curve rather than expanding the achievable performance frontier. Given that the untuned baseline already operated near the apparent information ceiling of the dataset, this outcome is both expected and methodologically defensible.

Notably, the top-performing parameter configurations differed by less than 0.0003 PR-AUC, suggesting that model performance was highly robust to hyperparameter choice and driven primarily by the engineered feature set rather than fine-grained parameter optimisation.


## 5. Recipient Frequency Comparison Study

Section C hypothesised that a backward-looking recipient_count_24h feature could capture mule-account behaviour by identifying low-frequency recipient accounts. To evaluate this empirically, the tuned Random Forest configuration was trained twice under identical conditions: same hyperparameters, identical train/validation splits, and the same random seed. The only difference was the inclusion or exclusion of `recipient_count_24h`.

| Metric           |          9 Features |         10 Features |
| ---------------- | ------------------: | ------------------: |
| PR-AUC           |              0.9951 |              0.9951 |
| F1               |              0.9967 |              0.9967 |
| Precision        |              0.9992 |              0.9992 |
| Recall           |              0.9943 |              0.9943 |
| Confusion Matrix | TP=1225, FP=1, FN=7 | TP=1225, FP=1, FN=7 |

The two models produced identical predictions on every validation transaction. As shown in Figure 9, `recipient_count_24h` ranked only eighth out of ten features by Gini importance (~0.6%), substantially below the balance-derived features that dominated model decisions. This likely reflects the existing balance-pattern features (`drainage_ratio`, `error_orig`, and `oldbalanceOrg`) already capture nearly all separable fraud signal in the dataset.

Given the absence of measurable performance lift, the simpler 9-feature pipeline was retained for deployment, eliminating the additional rolling-window computation step.



## 6. Final Test-Set Performance

The final tuned Random Forest model was evaluated once on the held-out test set, which remained untouched throughout feature engineering, threshold analysis, and hyperparameter tuning. The resulting performance metrics are summarised below.

| Metric                    | Test Set |
| ------------------------- | -------: |
| PR-AUC                    |   0.9980 |
| ROC-AUC                   |   0.9992 |
| F1 (threshold = 0.5)      |   0.9972 |
| Precision                 |   0.9976 |
| Recall                    |   0.9968 |
| Recall @ 90% precision    |   0.9976 |
| Cost-weighted score (5:1) |       23 |

The final confusion matrix contained 1,228 true positives, 3 false positives, 4 false negatives, and 414,327 true negatives (Figure 11). Operationally, the model detected 99.68% of fraudulent transactions while generating only three false alarms across more than 414,000 legitimate transactions. The final precision-recall curve is shown in Figure 10.

Validation and test performance were also highly consistent, with test metrics marginally exceeding validation performance (PR-AUC: 0.9951 → 0.9980; F1: 0.9967 → 0.9972), indicating no evidence of overfitting or validation-set leakage.

However, these results should not be interpreted as production-level fraud-detection performance. PaySim is a synthetic dataset whose fraud behaviour follows deterministic simulator rules, including the near-perfect `drainage_ratio = 1.0` signature identified in Section B. Real fraud environments contain adaptive adversaries, evolving attack strategies, noisy transactional behaviour, and incomplete records that are absent from the simulator. The simulator artefacts observed earlier(including systematic balance inconsistencies and transactions exceeding sender balances) further simplify the discrimination task. Consequently, the methodological framework developed here (feature engineering, evaluation discipline, threshold optimisation, and reproducible pipelines) is more transferable to production settings than the specific numerical performance achieved on PaySim.
