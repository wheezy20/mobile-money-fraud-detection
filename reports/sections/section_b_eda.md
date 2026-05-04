The analysis was conducted using the PaySim dataset, a synthetic mobile money transaction simulator calibrated against transaction patterns observed in African mobile money ecosystems. The dataset contains 6,362,620 transactions spanning approximately 31 days (744 hourly time steps), occupying approximately 1.68 GB of memory across 11 variables. Variables comprise eight numerical attributes (e.g., transaction amount and account balances) and three categorical attributes: transaction type, sender identifier, and recipient identifier. No missing values were observed in any column, reflecting the controlled nature of the simulation process rather than the realities of production financial systems, where incomplete or inconsistent records are common. Consistent with the considerations outlined in Section A, the variable isFlaggedFraud was excluded from modelling to prevent label leakage, while the sender identifier (nameOrig) was excluded due to its near-unique cardinality, as discussed in §4.

## Sub-section 2: Class Distribution and Transaction Type Structure

The dataset exhibits severe class imbalance, with only 8,213 fraudulent transactions out of 6,362,620 total observations (0.1291%), corresponding to approximately one fraudulent transaction for every 774 legitimate transactions. This confirms the imbalance concerns discussed in Section A and reinforces the need for evaluation metrics such as PR-AUC, recall at fixed precision, and cost-sensitive analysis rather than raw accuracy alone.

However, examination of fraud occurrence by transaction type reveals a strong structural pattern: fraud is confined entirely to only two of the five transaction categories in the dataset (see Figure 1).

| Transaction Type | Total Transactions | Fraud Cases | Fraud Rate | Share of Total Fraud |
|---|---|---|---|---|
| CASH_OUT | 2,237,500 | 4,116 | 0.18% | 50.12% |
| TRANSFER | 532,909 | 4,097 | 0.77% | 49.88% |
| CASH_IN | 1,399,284 | 0 | 0.00% | 0.00% |
| DEBIT | 41,432 | 0 | 0.00% | 0.00% |
| PAYMENT | 2,151,495 | 0 | 0.00% | 0.00% |

Notably, the two fraud-affected categories—`CASH_OUT` and `TRANSFER`—both represent outbound fund movements, consistent with operational patterns associated with account-takeover and unauthorised withdrawal fraud. Although `TRANSFER` transactions exhibit the highest fraud rate (0.77%), `CASH_OUT` contributes slightly more fraud cases in absolute terms due to its substantially larger transaction volume.

The remaining transaction types (`CASH_IN`, `DEBIT`, and `PAYMENT`) contain no fraud cases across more than 3.5 million transactions, contributing little predictive value while substantially increasing class imbalance. Consequently, subsequent modelling stages restrict analysis to `CASH_OUT` and `TRANSFER` transactions only, reducing the dataset to approximately 2.77 million rows and increasing the effective fraud rate from 0.13% to approximately 0.30%. This preprocessing strategy is consistent with established PaySim research methodology (Lopez-Rojas, Elmir & Axelsson, 2016).



## Sub-section 3: Fraud Signatures — Amount and Balance Patterns

Beyond the structural transaction-type patterns identified above, two additional analyses reveal strong distributional signals separating fraudulent from legitimate transactions: transaction amount and account balance behaviour.

### Transaction Amount

Fraudulent transactions are systematically larger than legitimate transactions. The mean transaction amount for legitimate activity is approximately 178,197, compared to 1,467,967 for fraudulent transactions—an increase of roughly 8.2×. Visual inspection of the transaction amount distributions (Figure 2) further shows that fraud transactions are concentrated at substantially higher values, while legitimate transactions exhibit a broader spread across lower-value transfers. Both classes display extreme right skewness, with transaction amounts ranging up to approximately 92.4 million. This skew suggests that logarithmic transformation of the `amount` variable will be necessary during feature engineering to stabilise variance and reduce the influence of extreme outliers.

### Balance Behaviour and Engineered Fraud Signatures

For a standard mobile money transfer, the sender and recipient balances should satisfy the following accounting relationships:

\text{oldbalanceOrg} - \text{newbalanceOrig} = \text{amount}

\text{oldbalanceDest} + \text{amount} = \text{newbalanceDest}

Deviations from these relationships were captured using three engineered features: `drainage_ratio`, `error_orig`, and `error_dest`.

| Feature          | Legitimate (Median) | Fraud (Median) |
| ---------------- | ------------------: | -------------: |
| `drainage_ratio` |              605.54 |           1.00 |
| `error_orig`     |          144,200.83 |           0.00 |
| `error_dest`     |                0.00 |       2,231.46 |

The strongest signal is the `drainage_ratio`, defined as the proportion of the sender’s balance transferred during a transaction. For at least half of all fraud cases, the ratio equals exactly 1.0, indicating that the entire available balance is drained in a single operation. Figure 3 confirms this visually, showing a sharp fraud-specific spike at a drainage ratio of 1.0 that is largely absent in legitimate transactions.

Similarly, fraudulent transactions exhibit near-zero sender balance error (`error_orig = 0`), meaning the sender’s balance decreases by exactly the transferred amount. In contrast, legitimate transactions display systematic discrepancies, likely reflecting artefacts introduced by the PaySim simulator rather than realistic accounting behaviour. Recipient-side balance behaviour (error_dest) shows the swapped pattern: legitimate destinations behave correctly (median = 0), while fraud destinations show small but consistent deviations. Suggesting that fraudulent destination accounts behave differently from normal recipient accounts, potentially reflecting low-activity “mule” accounts used for laundering stolen funds.

Collectively, these balance-derived variables provide the strongest discriminatory signals in the dataset and form the foundation of the feature engineering strategy developed in Section C.


## Sub-section 4: Temporal and Identity Patterns

Two additional patterns merit consideration: temporal fraud concentration and the behaviour of identity-related variables.

### Temporal Patterns

Fraud occurrence is strongly concentrated during late-night and early-morning hours. Within the filtered transaction set (`CASH_OUT` and `TRANSFER` only), the baseline fraud rate is approximately 0.30%; however, fraud rates increase dramatically between 2 AM and 6 AM.

| Hour of Day | Total Transactions | Fraud Rate |
| ----------- | -----------------: | ---------: |
| 5 AM        |                632 |     57.91% |
| 4 AM        |                512 |     53.52% |
| 3 AM        |                780 |     41.79% |
| 6 AM        |                904 |     39.60% |
| 2 AM        |              1,922 |     19.35% |

At 5 AM, fraud is approximately 193 times more likely than the baseline rate. This pattern is consistent with operational account-takeover behaviour, where fraudulent activity is performed during hours when victims are less likely to monitor their accounts or respond to suspicious transactions. Consequently, temporal indicators such as `hour_of_day` or a derived `is_unusual_hour` feature are likely to provide strong predictive value in Section C.

### Identity Columns

The sender identifier (`nameOrig`) exhibits extremely high cardinality, with 2,768,630 unique senders across 2,770,409 filtered transactions, meaning sender accounts almost never repeat. As a result, the variable provides little reusable predictive signal and is excluded from modelling.

In contrast, the recipient identifier (`nameDest`) demonstrates greater repetition, with 509,565 unique recipients averaging approximately five transactions each. Interestingly, the most frequently occurring recipient accounts were associated exclusively with legitimate activity, suggesting that fraud destinations are typically low-frequency or short-lived “mule” accounts rather than established recipients. This indicates that recipient-frequency features may carry predictive signal; however, to avoid label leakage, such features must be computed using backward-looking rolling counts available only up to the time of transaction initiation. The implementation of these features is discussed further in Section C.
