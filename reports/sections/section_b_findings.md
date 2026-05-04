# Section B: EDA Findings

## Dataset overview
- 6,362,620 transactions across 30+ days (max step = 743 hours, ~31 days)
- 11 columns; 8 numeric, 3 categorical (type, nameOrig, nameDest)
- 1.68 GB in memory
- No missing values across any column (note: this reflects synthetic data; production data would have gaps)

## Class imbalance (verifying Section A)
- 8,213 fraudulent transactions (0.1291%)
- 6,354,407 legitimate transactions (99.8709%)
- Imbalance ratio: 1 fraud per 774 legitimate transactions

## Transaction amount distribution
- Range: 0.00 to 92,445,516.64
- Median: 74,871.94; Mean: 179,861.90
- Heavy right skew — implies log transformation needed for modelling

## Sender balance pattern (PaySim quirk)
- Median of newbalanceOrig is 0 — sender frequently emptied after transaction
- Possible signal for fraud detection (verify in next cell)

## isFlaggedFraud
- Almost never triggered (mean ~0); effectively useless as baseline

## Fraud signatures (key signals identified)

### Transaction amount
- Legitimate mean: 178,197
- Fraud mean: 1,467,967
- Fraud transactions are **8.2× larger on average**
- Implication: `amount` is a strong feature, but log-transform for skew

### Sender balance pattern
- Fraud transactions drain ~88% of sender's balance
- Legitimate transactions show normal balance growth
- Implication: engineer `balance_drain_ratio` as a feature in Section C

### Recipient balance pattern
- Fraud recipients have lower starting balances (likely "mule accounts")
- Receive disproportionately large credits
- Implication: engineer recipient-side balance features

## Rule-based baseline (isFlaggedFraud)
- Triggered only 16 times across 6.3M transactions
- 100% precision (all 16 flags are fraud)
- 0.19% recall (16 / 8,213 actual fraud cases caught)
- Acts as a meaningful but extremely weak baseline

## Critical finding: Fraud occurs in only 2 of 5 transaction types

| Type | Total | Fraud | Fraud Rate | % of all fraud |
|---|---|---|---|---|
| CASH_OUT | 2,237,500 | 4,116 | 0.18% | 50.12% |
| TRANSFER | 532,909 | 4,097 | 0.77% | 49.88% |
| CASH_IN | 1,399,284 | 0 | 0.00% | 0.00% |
| DEBIT | 41,432 | 0 | 0.00% | 0.00% |
| PAYMENT | 2,151,495 | 0 | 0.00% | 0.00% |

### Implications
- 56% of transactions (CASH_IN, DEBIT, PAYMENT) contribute zero fraud signal
- Modelling will be restricted to CASH_OUT and TRANSFER only (~2.77M rows)
- Filtering raises the effective fraud rate from 0.13% to ~0.30%
- Decision aligns with published PaySim research methodology
- Both fraud-affected types are *outflow* transactions (cashing out to an agent or transferring to another account) — consistent with the operational pattern of account-takeover fraud


## Balance-pattern fraud signatures

Three engineered features expose strong fraud signals:

### Drainage ratio: amount / (oldbalanceOrg + 1)
- Legitimate median: 605.54
- **Fraud median: 1.00**
- At least half of fraud cases transfer exactly the entire sender's balance.

### Sender balance error: amount - (oldbalanceOrg - newbalanceOrig)
- Legitimate median: 144,200.83
- **Fraud median: 0.00**
- For fraud, the sender's balance drops by exactly the transferred amount; legitimate transactions exhibit systematic (simulator-induced) discrepancies.

### Recipient balance error: amount - (newbalanceDest - oldbalanceDest)
- **Legitimate median: 0.00**
- Fraud median: 2,231.46
- Mirror image of the previous: legitimate destinations behave correctly; fraud destinations show small but consistent deviations.

These three features will be carried forward to Section C as core engineered features.


## Visual confirmation: the "full drain" fraud signature

Figure fig_03_drainage_pattern.png plots the density of drainage_ratio
(amount / oldbalanceOrg) for transactions with non-zero sender balance:

- Out of 8,213 total fraud cases, 8,172 had a non-zero sender balance and
  appear in this plot. The remaining 41 had oldbalanceOrg = 0, which
  represents a different fraud pattern (already-emptied accounts being
  drained again, or simulator artefacts).
- Fraud transactions form a sharp spike at ratio = 1.0, indicating the
  entire sender balance is consistently transferred.
- A secondary, larger spike of legitimate transactions appears at the
  clipped upper bound of 2.0, indicating that PaySim allows transfers
  exceeding the sender's balance — a known simulator quirk that further
  underscores the limitations of generalising results to production data.
- Apart from these spikes, legitimate transactions show a broad, low
  distribution across all drainage ratios.

This figure provides the strongest single-feature visual evidence of the
fraud signature in PaySim and motivates the use of drainage_ratio as a
core engineered feature in Section C.



## Time-of-day pattern

Fraud rates per hour-of-day (top 5):

| Hour | Total txns | Fraud rate |
|---|---|---|
| 5 AM | 632 | 57.91% |
| 4 AM | 512 | 53.52% |
| 3 AM | 780 | 41.79% |
| 6 AM | 904 | 39.60% |
| 2 AM | 1,922 | 19.35% |

Compared to the baseline rate of 0.30%, fraud is up to 193× more likely
during late-night hours (2-6 AM). This reflects a typical account-takeover
pattern where fraudsters operate while victims are unlikely to notice.
Implication: hour_of_day or a binary "unusual hour" indicator should be
engineered as a feature in Section C.

## Identity columns

- nameOrig (sender): 2,768,630 unique IDs across 2,770,409 transactions.
  Senders almost never repeat. Column carries no signal as-is and will be
  dropped from the feature set.
- nameDest (recipient): 509,565 unique IDs across 2,770,409 transactions.
  Recipients average ~5 transactions each. The 5 most-frequent recipients
  (60-75 transactions each) had zero fraud cases, suggesting fraud destinations
  are typically low-frequency accounts. A recipient frequency feature could
  carry signal but must be computed as a backward-looking rolling count to
  prevent label leakage at inference time.