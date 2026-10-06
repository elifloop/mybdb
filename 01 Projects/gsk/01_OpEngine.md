# Opportunity Engine: Notebook Guide

## Purpose and Workflow

This workspace contains four Databricks notebooks for estimating HCP-level outcomes and turning those estimates into operational priority groups. The notebooks are related, but they do not all train models:

1. `Learning_from_Aggregates_V3.ipynb` trains a LightGBM regression model to estimate average sales for records marked as non-initiation candidates. It writes actual and predicted sales to an aggregate output table.
2. `Growth_model_metrics_V3.ipynb` reads those aggregate predictions, calculates relative sales, joins channel-plan information, assigns HCP priority tiers, and reports segment/rank distributions across selected months.
3. `Initiation_model_RF_V3.ipynb` trains a Random Forest classifier to estimate the probability of the `Switch_toF2F` outcome and writes high-probability records to an initiation output table.
4. `Initiation_model_metrics_V3.ipynb` reads initiation probabilities, joins channel-plan information, creates monthly/yearly priority tiers, and summarizes the resulting target population.

The intended high-level flow is:

```text
Prepared opportunity-engine source tables
	+--> Learning_from_Aggregates_V3 --> aggregate_new_* --> Growth_model_metrics_V3
	+--> Initiation_model_RF_V3 -------> initiation_new_* --> Initiation_model_metrics_V3
```

The two branches are related operationally but are not sequential dependencies of one another. The notebooks use Spark/Databricks tables for input and output, and convert selected data to pandas/scikit-learn or LightGBM for modeling and analysis.

## Notebook Details

### Learning from Aggregates

**File:** `Learning_from_Aggregates_V3.ipynb`

- Reads a configured country/product source table and filters to `Initiation_Candidate = 0`.
- Uses records through the configured end month for training/validation and later months for the holdout test set.
- Predicts `AVG_SALES` from HCP attributes, product/share/volume measures, engagement fields, customer segments, and elapsed-time/switch fields.
- Imputes numeric missing values with zero, standardizes numeric features, imputes categorical values as `missing`, and one-hot encodes categories. Unknown test categories are ignored by the encoder.
- Fits a LightGBM gradient-boosted-tree regressor, reports test $R^2$, and runs shuffled five-fold cross-validation on the training period.
- Computes SHAP values for the test data and produces global feature-importance plots.
- Writes account attributes, observed `AVERAGE_SALES`, and `PREDICTED_SALES` to `aggregate_new_<country>_<product>` in the configured database.

The notebook defines mean/normal and sum/Poisson gradient-Hessian helper functions. In the current fit calls, however, the model is configured with `objective='regression'` and the mean/normal helper is supplied as `eval_metric`; it is not installed as the model's training objective. The sum/Poisson helper is defined but not used in the shown training path. Treat the notebook as standard LightGBM regression unless that configuration is deliberately changed and validated.

### Growth Metrics

**File:** `Growth_model_metrics_V3.ipynb`

- Reads aggregate output records, including `PREDICTED_SALES` and product-share fields.
- Calculates relative Trelegy sales as `PREDICTED_SALES * SHARE_SITT * SHARE_TRELEGY`.
- Joins HCPs to multi-cycle channel-plan data, then ranks relative sales by percentile for each configured month/cycle.
- Assigns the tiers `Low`, `Medium`, `High`, and `Critical` using percentile bands of 0-75%, 75-92%, 92-99%, and 99-100%.
- Writes monthly priority/rank tables and examines segment/rank overlap and the number/distribution of HCPs selected for targeting.

These tiers are relative rankings within the data being ranked, not calibrated estimates of absolute sales or business value. Confirm that share fields and the channel-plan cycle used are the intended ones for the reporting period.

### Initiation Random Forest

**File:** `Initiation_model_RF_V3.ipynb`

- Reads the configured country/product source table, derives `Switch_toF2F` from the face-to-face switching time fields, and selects training and later-period test records.
- Uses customer/consent attributes, product shares and volumes, average sales, engagement, and switching-history variables as predictors.
- Applies numeric imputation/scaling and categorical imputation/one-hot encoding, then fits a `RandomForestClassifier` with 2,000 trees.
- Produces class-1 probabilities for `Switch_toF2F`, reports ROC-AUC, applies a 0.5 cutoff for a confusion matrix and predicted-positive count, and retains records with probability above 0.5 for output.
- Writes the selected records and their `INITIATION_PROBABILITY` to `initiation_new_<country>_<product>`.

The output table contains only records above the probability cutoff in the shown code. It is therefore a selected target list, not a complete table of probabilities for every test record. The notebook's K-fold example is commented out, so it does not currently perform that cross-validation.

### Initiation Metrics

**File:** `Initiation_model_metrics_V3.ipynb`

- Reads the initiation output table produced by the classifier notebook.
- Joins the probability-ranked HCPs with channel-plan data for configured months/cycles.
- Calculates probability percentiles and assigns `Low`, `Medium`, `High`, and `Critical` tiers using the same 75/92/99 percentile boundaries as the growth metrics notebook.
- Summarizes the composition of selected HCPs by prescription segment and analyzes the number and distribution of targets by rank.
- Writes ranked outputs for monthly and yearly channel-plan views.

This is a post-model reporting and prioritization notebook; it does not train or evaluate the Random Forest. The output priorities are relative percentiles among records present in the input table, which has already been filtered by the classifier notebook's 0.5 probability cutoff.

## Data Science Methodology: Theory and Interpretation

### 1. Define the prediction question

Supervised learning estimates a target $y$ from observed features $X$. The aggregate-sales branch is a regression problem because `AVG_SALES` is numeric. The initiation branch is a binary classification problem because `Switch_toF2F` indicates whether a switch outcome occurred. A business ranking is a later decision step; it is not itself the model target.

The intended use determines what counts as a valid label and prediction time. Every predictor must be available before the outcome period. Features measured after, or derived from, the outcome can leak information and inflate offline performance.

### 2. Split data to represent future use

The notebooks use a month-based training/test boundary: earlier periods are used to fit models and later periods are held out. This temporal holdout is more representative of future deployment than a purely random split when observations change over time. The aggregate notebook additionally reports shuffled five-fold cross-validation within its training data. Random folds can mix months or repeated HCPs across folds, so their scores may be optimistic when records are longitudinal; use time-based folds or group-aware folds when that better matches deployment.

Preprocessing should be learned from training data only and then applied unchanged to validation/test data. This prevents information from the holdout set influencing imputation, scaling, or category definitions. The notebooks use fit-on-training/transform-on-test for their primary preprocessing path.

### 3. Prepare mixed feature types

Numeric features are imputed and standardized; categorical features are imputed and one-hot encoded. Imputation avoids dropping records solely because some predictors are missing. Standardization places numeric variables on comparable scales and is particularly important for distance- or gradient-based models; it is generally not essential for tree splits, though it does not ordinarily change their ordering-based split decisions. One-hot encoding represents categories without imposing a false numeric order, while ignoring unseen categories makes scoring more robust to new values.

The current numeric imputation value of zero is an explicit modeling assumption, not a neutral treatment of missingness. Check whether zero is meaningful for each field and whether missingness itself should be represented separately.

### 4. Fit models for each target type

LightGBM builds an ensemble of decision trees in sequence, where later trees correct residual patterns from earlier trees. It can capture nonlinearities and interactions in tabular data. The sales model is configured for regression and is evaluated with $R^2$, which compares squared prediction error with the error of predicting the observed mean. $R^2=1$ is perfect, $R^2=0$ matches the mean baseline, and negative values are worse than that baseline.

Random Forest trains many decision trees on randomized samples/features and aggregates their predictions. For classification, the notebook uses estimated class probabilities. ROC-AUC measures how well scores rank positive cases above negative cases across all possible cutoffs; it does not choose an operational cutoff and does not establish that probabilities are calibrated. The 0.5 cutoff creates a particular confusion matrix and target count, but should be chosen based on the relative costs of missed opportunities and unnecessary outreach, as well as available capacity.

### 5. Evaluate and explain

Cross-validation estimates how results vary across resampled training folds; a separate temporal holdout estimates performance on later data. These answer different questions and both should be interpreted alongside a simple baseline and the actual business objective. For imbalanced classification, ROC-AUC should be supplemented with precision, recall, PR-AUC, and calibration checks at candidate operating thresholds. For sales regression, consider MAE/RMSE and residual checks in addition to $R^2$, especially by month, segment, and sales range.

SHAP assigns feature contributions to individual model predictions relative to a reference prediction. Aggregating absolute SHAP values gives a global measure of how much features influence predictions in the analyzed sample. It explains model behavior; it is not evidence that a feature causes sales or switching.

### 6. Convert scores to actions

The reporting notebooks use percentile bands to divide HCPs into operational tiers. Percentiles are useful when a fixed fraction of a population should receive attention, but tier sizes depend on the input population and selected period. The same tier label in two months need not represent the same absolute probability or expected sales. Validate that the input population, deduplication grain, cutoff, and available channel capacity match the intended decision. A ranked association is not a causal estimate of the effect of contacting an HCP.

## Operational Notes and Limitations

- The notebooks are parameterized with Databricks widgets for country, product, database/table prefix, and reporting months. Confirm these values and the availability/schema of source and channel-plan tables before running.
- Outputs use Spark `saveAsTable` with overwrite mode. Running a notebook can replace an existing output table for that country/product. Review target names and existing data before execution.
- The initiation metrics notebook includes hard-coded `DROP TABLE` statements for example Germany/Trelegy yearly output tables. Review those statements before running with shared or production tables.
- Several notebook cells are not executed in the checked-in files. Results described here are based on the code paths, not a fresh end-to-end run or verified model performance.
- The aggregate-learning custom gradient/Hessian functions should not be described as an active custom training loss in the current configuration. If aggregate-level learning is a requirement, verify the mathematical objective, LightGBM API contract, and evaluation design, then compare it with a documented baseline.
- Confirm feature availability dates, duplicate HCP/month handling, target definitions, class balance, threshold selection, and month-over-month stability before using model scores for field decisions.
