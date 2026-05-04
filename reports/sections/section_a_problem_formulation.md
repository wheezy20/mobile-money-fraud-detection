Mobile money services such as MTN MoMo and Vodafone Cash have become the primary financial infrastructure across much of sub-Saharan Africa. In Ghana alone, mobile money transactions reached GH₵4.54 trillion in 2025, exceeding all traditional payment channels combined (Bank of Ghana, 2025). However, this rapid adoption has been accompanied by rising fraud schemes, SIM swap attacks, agent impersonation, and "wrong number" reversal scams, that exploit user trust. Unlike in developed markets, financial losses are often borne directly by low-income users, amplifying economic vulnerability. This project frames mobile money fraud detection as a supervised binary classification problem: given the features available at the moment a transaction is initiated, predict the probability that the transaction is fraudulent.
## 2. Target Variable
The target variable is isFraud, a binary indicator where 1 denotes a fraudulent transaction and 0 denotes a legitimate transaction. The dataset additionally contains isFlaggedFraud, a rule-based flag triggered by transactions exceeding a predefined threshold (e.g., unusually large transfers). To prevent label leakage, isFlaggedFraud is excluded from the feature set, as it encodes prior knowledge not available at inference time. However, it may serve as a useful baseline for comparing model performance.
## 3. Type of ML Problem
This task is formulated as a supervised binary classification problem, where the model learns to distinguish fraudulent from legitimate transactions using labelled data. A defining characteristic of this dataset is severe class imbalance, with only 8,213 fraudulent transactions out of 6,362,620 total (~0.13%). This imbalance is critical because a naive model could achieve over 99% accuracy by predicting all transactions as legitimate, yet fail to detect any fraud. Consequently, model design and evaluation must prioritise imbalance-aware techniques (such as SMOTE resampling, class-weighted loss functions, or threshold tuning) and metrics beyond accuracy. and metrics beyond accuracy.
## 4. Stakeholders

Mobile money fraud detection systems affect multiple stakeholders with competing objectives, making model design inherently a trade-off rather than a single optimisation problem:

- **Operators** (e.g., MTN Ghana, Vodafone Ghana): seek to maximise fraud detection (high recall) to minimise financial losses, but must also limit false positives to avoid customer dissatisfaction and operational costs.
- **Customers**: require strong protection against fraud, especially in low-income contexts, but also need reliable access to their own money. False positives can block critical transactions (e.g., medical or transport payments), creating severe real-world consequences.
- **Regulators** (e.g., Bank of Ghana): prioritise consumer protection, transparency, and auditability, often requiring interpretable and explainable decision-making processes.
- **Fraud investigation teams**: operate under limited review capacity, needing high-precision alerts and interpretable outputs to efficiently prioritise cases.

These conflicting requirements (recall vs precision, automation vs explainability) mean the system must balance trade-offs, directly shaping model selection and evaluation strategies.
## 5. Success Metrics

Given the conflicting stakeholder requirements identified above, particularly the trade-off between maximising fraud detection (recall) and maintaining high-quality alerts (precision), model performance will be evaluated using metrics designed for severely imbalanced classification with asymmetric costs:

- **Precision–Recall AUC (PR-AUC)**, primary metric: This measures performance across all thresholds while focusing on the minority (fraud) class. Unlike ROC-AUC, it does not give inflated scores for correctly predicting the dominant legitimate transactions, making it more informative in this highly imbalanced setting.
- **Recall at fixed precision** (e.g., 90% precision): This evaluates how much fraud can be captured while ensuring that most flagged transactions are genuinely fraudulent. It reflects real operational constraints, where investigation teams require high-confidence alerts due to limited review capacity.
- **F1-score**: As the harmonic mean of precision and recall, F1 provides a balanced single-value summary. It is useful for benchmarking against prior studies and ensuring neither metric is disproportionately sacrificed.
- **Cost-weighted confusion matrix**: Errors are weighted by transaction value, translating false negatives into direct financial loss and false positives into customer friction costs. This aligns evaluation with real business impact.

While accuracy will be reported for completeness, it will not guide model selection due to its misleading nature under extreme class imbalance.
## 6. Assumptions and Constraints

This analysis proceeds under the following assumptions and constraints:

- **Synthetic data limitation**: The PaySim dataset is simulated rather than derived from live production systems. While it is based on real African mobile money patterns, it cannot fully capture evolving fraud behaviours or adversarial adaptation. Findings should therefore be interpreted as a methodological demonstration, not deployment-ready results.
- **Limited temporal coverage**: The dataset spans only 30 days of transactions. This constrains the model's ability to learn seasonal trends, long-term behavioural shifts, and concept drift, which are critical in real-world fraud detection systems.
- **Feature availability at inference time**: It is assumed that all selected features can be computed in real time at transaction initiation. Features requiring future or post-transaction information are excluded, which may limit predictive power but ensures deployment feasibility.
- **Geographic specificity**: The dataset reflects patterns from a single mobile money ecosystem within sub-Saharan Africa. Fraud dynamics may differ across countries, constraining the generalisability of results.
- **Ethical considerations**: False positives may disproportionately affect low-income users who rely on mobile money for essential transactions. This necessitates careful threshold selection and highlights the importance of fairness and human oversight in real deployments.

Overall, these constraints frame the project as an exploration of robust fraud detection methodology under realistic but simplified conditions.