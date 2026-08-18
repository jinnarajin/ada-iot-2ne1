# Research protocol

## Objective

Determine whether explicitly representing frequency information improves simulated-tremor detection under both subject shift and sampling-condition shift in a small IMU dataset.

## Confirmatory hypotheses

- **H1 - Subject-independent:** Adding spectral features to a time-domain baseline improves mean Leave-One-Subject-Out balanced accuracy.
- **H2 - Cross-dataset:** Adding spectral features improves the mean of the two directional cross-dataset balanced accuracies.
- **H3 - Sensor contribution:** Gyroscope features provide complementary information to accelerometer features.
- **H4 - Aggregation:** Attention pooling improves recording-level performance over mean pooling only when tremor evidence is temporally intermittent.

H1 and H2 are primary. H3 and H4 are secondary analyses.

## Unit of analysis

- Recording: one approximately 20-second CSV file and the unit used for splitting and final evaluation.
- Instance: a 3-second window with 50% overlap, used for feature extraction or MIL.
- Bag: all valid windows from one recording.

Overlapping windows are correlated observations and are not independent samples. Final confidence intervals and tests must therefore operate on recordings or subjects, not windows.

## Preprocessing

1. Validate schema and strictly increasing timestamps.
2. Resample acceleration and gyroscope channels to a common 50 Hz grid.
3. Record missing intervals and reject or mask windows crossing gaps above a prespecified threshold (initially 100 ms).
4. Remove the per-window mean from each raw axis.
5. Use acceleration and gyroscope axes as separate input groups; derive vector magnitude after centering.
6. Exclude `angle_*_deg` from confirmatory experiments because these are fused, orientation-dependent signals with wrap-around discontinuities.
7. Fit scaling and any learned preprocessing on training data only.

Do not expose dataset ID, subject ID, filename, timestamp pattern, or original sampling rate to the classifier.

## Explicit frequency representation

For each sensor and axis/magnitude, calculate Welch power spectral density on the uniformly resampled signal. Prespecified spectral features are:

- log power in 3-12 Hz;
- dominant frequency within 3-12 Hz;
- peak prominence or peak-to-band-power ratio;
- spectral entropy;
- power ratios for 3-6, 6-9, and 9-12 Hz.

The 3-12 Hz interval is broad enough to cover the different simulated oscillation frequencies observed across the two datasets. Frequency-band alternatives must be reported as sensitivity analyses rather than selected using test performance.

## Model ladder

### B0 - amplitude sanity baseline

Logistic regression using acceleration RMS and gyroscope RMS. This reveals how much of the task is explained by simulated-tremor amplitude alone.

### B1 - time-domain baseline

Regularized logistic regression or a shallow tree ensemble using RMS, variance, median absolute deviation, jerk RMS, and zero-crossing features.

### B2 - explicit-frequency baseline

The same estimator and tuning budget as B1, with the prespecified spectral features added. The controlled B1-versus-B2 comparison is the primary test of H1 and H2.

### D1 - compact raw-signal model

A small dual-branch 1D CNN processes acceleration and gyroscope windows separately before feature fusion.

### D2 - dual-view frequency-aware model

Fuse the raw-window encoder with a log-PSD or spectrogram encoder. Model capacity must be constrained and reported.

### D3 - attention MIL

Use the D1 or D2 window encoder followed by gated-attention pooling over a recording. Compare against mean and max pooling with an identical encoder. Attention weights are explanatory diagnostics, not proof of causal importance.

## Evaluation

### Subject-independent evaluation

Use six Leave-One-Subject-Out folds. All recordings and windows from the held-out participant are test data. Hyperparameters must be chosen inside the remaining participants using nested grouped validation or fixed in advance.

### Cross-dataset evaluation

Run both directions independently:

- train and tune on `dataset_a`; test once on `dataset_b`;
- train and tune on `dataset_b`; test once on `dataset_a`.

Do not normalize each dataset independently using its complete statistics.

### Metrics

Primary metric: recording-level balanced accuracy.

Report sensitivity, specificity, macro F1, AUROC, AUPRC, confusion matrices, and per-subject results. Include bootstrap confidence intervals resampled at subject level where feasible. With six subjects, emphasize effect sizes and fold-level uncertainty rather than asymptotic significance claims.

## Required ablations and controls

- acceleration only, gyroscope only, and both;
- time features versus time plus frequency features;
- raw axes versus centered vector magnitude plus axes;
- 2-, 3-, and 5-second windows as a sensitivity analysis;
- mean, max, and attention pooling with the same encoder;
- a dataset-ID prediction control to quantify residual domain information;
- label-shuffling and majority-class sanity checks.

## Interpretation boundary

Positive results support robustness for detecting voluntarily simulated tremor under the represented collection conditions. They do not establish clinical validity, disease classification, or performance on naturally occurring tremor.

