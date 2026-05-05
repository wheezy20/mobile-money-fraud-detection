Sub-section 1: Feature Selection and Justification

Following the exploratory analysis in Section B, the final feature set was constrained to a compact group of high-signal, leakage-free variables designed to maximise interpretability and operational relevance. Feature selection prioritised variables exhibiting strong discriminatory behaviour during EDA while avoiding redundancy and information unavailable at inference time.

| Feature | Source | Justification |
|---|---|---|
| `amount_log` | Log-transformed `amount` | Reduces extreme right skew observed in transaction amounts |
| `drainage_ratio` | Engineered | Strongest fraud signature; median = 1.0 for fraud cases |
| `error_orig` | Engineered | Captures sender-side balance inconsistencies |
| `error_dest` | Engineered | Captures recipient-side balance anomalies linked to mule accounts |
| `hour_of_day` | Engineered from `step` | Fraud heavily concentrated during late-night hours |
| `is_unusual_hour` | Engineered binary feature | Encodes off-hour fraud behaviour in simplified form |
| `oldbalanceOrg` | Raw variable | Preserves sender balance context |
| `oldbalanceDest` | Raw variable | Preserves recipient balance context |
| `type_encoded` | Encoded transaction type | Distinguishes `TRANSFER` from `CASH_OUT` operations |
| `recipient_count_24h` *(optional)* | Rolling engineered feature | Captures recipient reuse patterns while avoiding leakage |


Several variables were excluded from modelling. The step variable was replaced by interpretable temporal features (hour_of_day and is_unusual_hour), while newbalanceOrig and newbalanceDest were omitted because their information content is already embedded within the engineered balance-error variables. The sender identifier (nameOrig) was removed due to its near-unique cardinality and negligible predictive utility. Similarly, nameDest was used only to derive rolling recipient-frequency statistics before being discarded. Finally, isFlaggedFraud was excluded to prevent label leakage, as discussed in Section A.

The engineering procedures used to construct these derived variables are detailed in the following sub-section.



## Sub-section 2: Engineered Features

Engineered features were grouped into three categories: balance-derived features, temporal features, and recipient-frequency features. These variables were designed to capture behavioural patterns identified during the exploratory analysis while remaining computable at transaction initiation.

### Balance-Derived Features

Three engineered variables were constructed from sender and recipient balance relationships:

\text{drainage_ratio} = \frac{\text{amount}}{\text{oldbalanceOrg} + 1}

The `+1` term prevents division-by-zero errors for accounts with zero prior balance. This feature captures the proportion of the sender’s available funds transferred during the transaction and was motivated by the Section B finding that fraudulent transactions frequently drained the sender’s balance entirely.

\text{error\_orig} = \text{amount} - (\text{oldbalanceOrg} - \text{newbalanceOrig})

\text{error_dest} = \text{amount} - (\text{newbalanceDest} - \text{oldbalanceDest})

These variables measure deviations from the expected accounting relationships governing sender and recipient balances. As shown in Section B, fraudulent and legitimate transactions exhibited distinct balance-error patterns, making these variables strong discriminative features.

### Temporal Features

Temporal behaviour was extracted from the `step` variable, which records elapsed transaction hours:

\text{hour_of_day} = \text{step} \bmod 24

\text{is_unusual_hour} = \begin{cases}1,&0 \leq \text{hour_of_day} \leq 6\0,&\text{otherwise}\end{cases}

The continuous `hour_of_day` feature allows models to learn fine-grained temporal variation, while `is_unusual_hour` provides a simplified indicator of late-night behaviour. The threshold range (0–6 hours) was selected based on the elevated fraud rates observed during these periods in Section B.

### Recipient Frequency Feature

An optional rolling-frequency feature was also engineered:

$$
\text{recipient\_count\_24h}(t) = \left| \{ s : \text{nameDest}_s = \text{nameDest}_t \,\land\, \text{step}_s \in [\text{step}_t - 24,\, \text{step}_t - 1] \} \right|
$$

This variable counts the number of prior transactions received by the same destination account within the previous 24-hour window. The strict upper bound (`step_t - 1`) ensures that only historical information available at prediction time is used, preventing temporal leakage. Implementation required sorting transactions chronologically and maintaining recipient-level transaction histories for efficient rolling-window computation. Section B suggested that fraudulent recipient accounts are typically low-frequency or short-lived accounts; this feature was designed to capture that behaviour. Its computational overhead and predictive contribution are evaluated empirically in Section D.



## Sub-section 3: Encoding and Scaling

Three classes of preprocessing transformation were applied prior to modelling: categorical encoding, logarithmic transformation, and numeric scaling.

The transaction-type variable (`type`) was converted into a binary feature (`type_encoded`) following the Section B filtering stage, where only `CASH_OUT` and `TRANSFER` transactions remained. Transactions were encoded as 1 for `TRANSFER` and 0 for `CASH_OUT`. Binary encoding was preferred over one-hot encoding because only two categories remained, making an additional dummy variable unnecessary.

To address the heavy right skew observed in transaction amounts (Figure 2), the raw `amount` variable was transformed using:

\text{amount\_log} = \log(1 + \text{amount})

The `log1p` formulation safely handles zero-value transactions while compressing extreme outliers and stabilising variance.

Finally, z-score standardisation using `StandardScaler` was applied to all continuous numerical variables (`amount_log`, `drainage_ratio`, `error_orig`, `error_dest`, `hour_of_day`, `oldbalanceOrg`, `oldbalanceDest`, and optionally `recipient_count_24h`). Scaling was necessary because feature magnitudes varied across several orders of magnitude; oldbalanceOrg reaches the millions while hour_of_day is bounded by [0, 23]. Binary indicators (`type_encoded` and `is_unusual_hour`) were passed through without scaling.



## Sub-section 4: Reproducible Preprocessing Pipeline

All preprocessing logic was encapsulated within a reusable `scikit-learn` `Pipeline` defined in `src/preprocessing.py`, ensuring consistent behaviour across training, evaluation, and deployment environments.

```python
Pipeline([
    ("engineer", FunctionTransformer(engineer_features)),
    ("preprocess", ColumnTransformer([
        ("scale_numeric", StandardScaler(), NUMERIC_FEATURES),
        ("passthrough_binary", "passthrough", BINARY_FEATURES),
    ])),
])
```

The first stage applies deterministic feature-engineering transformations, while the second stage uses a `ColumnTransformer` to apply scaling selectively to continuous variables and pass binary indicators through unchanged. All remaining columns are excluded automatically, preventing accidental inclusion of identity variables or leakage-prone attributes.

This architecture improves reproducibility by packaging all preprocessing operations into a single serialisable object that can be saved and reloaded without manual intervention. Importantly, the same fitted pipeline used during model training is reused unchanged during inference, ensuring consistency between offline evaluation and deployed predictions. Because scaling parameters are learned only from the training data during `fit()`, the design also prevents preprocessing leakage into validation or inference stages.

An optional `use_recipient_frequency` flag allows the rolling recipient-frequency feature to be included or excluded while preserving the same overall pipeline structure. This supports the controlled comparison study presented in Section D and provides a deployment-ready preprocessing artefact for the FastAPI inference service described in Section E.
