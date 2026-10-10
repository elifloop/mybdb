# TCN Architecture

This document describes the Temporal Convolutional Network implemented in this repository. It is based on the current source in `src/model.py`, `src/data.py`, `src/trainer.py`, `src/pipeline.py`, and `src/config.py`.

## 1. Layers and Architecture Components

### Model family

The forecasting model is a **Temporal Convolutional Network (TCN)** for multi-step forecasting across many parallel time series.

At a high level, the model does this:

1. Takes a fixed-length history window for one series.
2. Runs that window through a stack of **causal dilated Conv1d residual blocks**.
3. Collapses the time axis with **adaptive average pooling**.
4. Concatenates a learned **series ID embedding**.
5. Produces a multi-step forecast with a final **linear layer**.

### Exact layers used

The implemented network uses the following layers.

#### Input

- Input tensor shape before the network: `(batch, window_size, n_features)`
- Default `window_size = 36`
- Default engineered feature count: `24` plus optional exogenous columns

#### Temporal block

Each `TemporalBlock` contains:

1. `Conv1d(in_ch, out_ch, kernel_size, dilation=d, padding=(kernel_size - 1) * d)`
2. `Chomp1d(padding)` to remove future-looking padded positions and preserve causality
3. `ReLU`
4. `Dropout`
5. `Conv1d(out_ch, out_ch, kernel_size, dilation=d, padding=(kernel_size - 1) * d)`
6. `Chomp1d(padding)`
7. `ReLU`
8. `Dropout`
9. Residual addition
10. Final `ReLU`

Both convolutions inside the block use **weight normalization**.

If `in_ch != out_ch`, the residual branch uses a `1 x 1` convolution:

- `Conv1d(in_ch, out_ch, kernel_size=1)`

#### TCN stack

The model stacks multiple temporal blocks in sequence.

Default training config:

- `num_channels = (64, 64, 64)`
- `kernel_size = 3`
- `dropout = 0.2`

This gives three temporal blocks with dilations:

- Block 1: dilation `1`
- Block 2: dilation `2`
- Block 3: dilation `4`

#### Output head

After the TCN stack:

1. `AdaptiveAvgPool1d(1)` reduces `(batch, channels, time)` to `(batch, channels, 1)`
2. `squeeze(-1)` gives `(batch, channels)`
3. `Embedding(n_series_ids, id_emb_dim)` maps the series ID to `(batch, id_emb_dim)`
4. Concatenation gives `(batch, channels + id_emb_dim)`
5. `Linear(channels + id_emb_dim, horizon)` outputs the forecast horizon

Default values:

- `id_emb_dim = 8`
- `horizon = 3`

### Causality

The convolutions are **causal** in effect because the implementation pads the sequence on the right length needed for same-length convolution and then removes the extra tail with `Chomp1d`. That means the output at time index `t` only depends on positions `<= t`.

### Dilation

For block index `i`, the dilation is:

$$d_i = 2^i$$

So with three blocks the dilations are `1, 2, 4`.

This lets deeper layers see farther into the past without increasing the kernel size.

### Receptive field

Because each temporal block has **two** dilated convolutions with the same dilation, the receptive field of the full stack is:

$$
R = 1 + 2 (k - 1) \sum_{i=0}^{L-1} d_i
$$

Where:

- $k$ is the kernel size
- $L$ is the number of temporal blocks
- $d_i = 2^i$

For the default configuration:

- $k = 3$
- $L = 3$
- dilations are $1, 2, 4$

So:

$$
R = 1 + 2(3 - 1)(1 + 2 + 4)
$$

$$
R = 1 + 4 \cdot 7 = 29
$$

So the default network has a receptive field of **29 time steps**.

Compared with the default `window_size = 36`:

- The model receives 36 historical rows per sample.
- The deepest convolutional output can directly depend on at most the most recent 29 time steps of that window.
- The final average pooling still aggregates the block outputs across all 36 positions.

## 2. Sequential Flow

The end-to-end flow is:

1. Raw event data is grouped by series ID.
2. Each series is resampled to monthly frequency.
3. Missing timestamps are filled and values are forward-filled or interpolated where needed.
4. Outliers are capped or flagged during preprocessing.
5. Feature engineering produces lag, difference, momentum, slope, rolling statistics, ratio, EMA, and optional exogenous features.
6. Rows with incomplete feature histories are dropped.
7. Sliding windows are created.
8. Windows are split chronologically into train, validation, and test.
9. Input features are scaled with one global `StandardScaler`.
10. Targets are log-transformed and scaled with one `StandardScaler` per series.
11. Training batches are formed as `(X_window, y_horizon, series_id)`.
12. The network predicts the next `horizon` values.
13. Predictions are inverse-scaled back to the original target scale.

### Tensor flow through the neural network

For a batch of size `B`:

1. Input: `(B, 36, F)`
2. Transpose for Conv1d: `(B, F, 36)`
3. TCN stack output: `(B, 64, 36)` with default channels
4. Adaptive average pooling: `(B, 64, 1)`
5. Squeeze: `(B, 64)`
6. Series embedding: `(B, 8)`
7. Concatenate: `(B, 72)`
8. Linear head: `(B, 3)`

### Mermaid flow diagram

```mermaid
flowchart TD
    A[Raw event rows] --> B[Group by series ID]
    B --> C[Monthly resampling and regularization]
    C --> D[Outlier handling and missing value filling]
    D --> E[Feature engineering]
    E --> F[Drop rows with incomplete lag or rolling history]
    F --> G[Sliding windows: X and y]
    G --> H[Chronological split: train val test]
    H --> I[Feature scaling and per-series target scaling]
    I --> J[Batch tuples: X window, y horizon, series ID]
    J --> K[TCN forward pass]
    K --> L[Scaled horizon forecast]
    L --> M[Inverse scaling and expm1]
    M --> N[Final forecast rows]
```

## 3. Worked Example: 70 Rows per Series and 721 Series

This section shows how the pipeline behaves for a dummy case with:

- `721` series
- `70` monthly rows per series before feature dropping
- default `window_size = 36`
- default `horizon = 3`
- default feature set with no extra exogenous columns

### Step 1: Raw panel size

If every series has 70 monthly rows, then the raw input panel has:

$$
721 \times 70 = 50{,}470
$$

raw rows.

### Step 2: Feature warm-up reduces usable rows

Feature engineering uses lags up to `12`, plus rolling features and ratios. The strictest warm-up requirement comes from the `lag_12` feature, so the first 12 rows of each series do not have a complete feature vector.

That leaves:

$$
70 - 12 = 58
$$

feature-complete rows per series.

Across all 721 series:

$$
721 \times 58 = 41{,}818
$$

feature-complete rows.

### Step 3: Sliding windows

`create_windows` builds samples with:

- one input window of length `36`
- one target horizon of length `3`

For a sequence of length `N`, the number of windows is:

$$
N - W - H + 1
$$

Where `W` is the window size and `H` is the forecast horizon.

Using `N = 58`:

$$
58 - 36 - 3 + 1 = 20
$$

So each eligible series produces **20 supervised samples**.

If all 721 series remain eligible, the total number of windows is:

$$
721 \times 20 = 14{,}420
$$

### Step 4: Chronological split per series

The code splits each series independently.

For `n = 20` windows:

- train end: `floor(0.70 x 20) = 14`
- validation end: `floor(0.85 x 20) = 17`
- test gets the remainder

So per series:

- train windows = `14`
- validation windows = `3`
- test windows = `3`

Across 721 series:

- train windows = $721 \times 14 = 10{,}094$
- validation windows = $721 \times 3 = 2{,}163$
- test windows = $721 \times 3 = 2{,}163$

### Step 5: Dummy single-series example

Assume one series has monthly target values:

$$
y_1, y_2, y_3, \dots, y_{70}
$$

After feature engineering, the first usable row is at original month 13. So the model-ready sequence is based on:

$$
t = 13, 14, \dots, 70
$$

For one training sample, the first window uses model-ready rows 1 to 36 and predicts the next 3 rows.

In original time indexing, that corresponds to:

- input rows: original months `13` to `48`
- target rows: original months `49` to `51`

So one sample is:

$$
X^{(1)} =
\begin{bmatrix}
x_{13} \\
x_{14} \\
\vdots \\
x_{48}
\end{bmatrix}
\in \mathbb{R}^{36 \times F}
$$

$$
y^{(1)} =
\begin{bmatrix}
y_{49} \\
y_{50} \\
y_{51}
\end{bmatrix}
\in \mathbb{R}^{3}
$$

Where each feature row `x_t` contains the engineered features for that month, for example:

$$
x_t = [\text{lag}_1, \text{lag}_2, \text{lag}_3, \text{lag}_6, \text{lag}_{12}, \text{diff}_1, \dots, \text{ema}_6]_t
$$

With default settings and no exogenous variables, `F = 24`.

### Step 6: Feature scaling and target scaling

The input window is globally standardized feature by feature:

$$
\tilde{x}_{t,f} = \frac{x_{t,f} - \mu_f}{\sigma_f}
$$

The target values are first log-transformed:

$$
z = \log(1 + y)
$$

Then standardized using the scaler fitted on the training targets of the same series:

$$
\tilde{z}_{t} = \frac{z_t - \mu_{\text{series}}}{\sigma_{\text{series}}}
$$

So the network learns on scaled horizon targets, not on raw units.

### Step 7: Neural processing of one batch item

Take one scaled sample with shape:

$$
X \in \mathbb{R}^{36 \times 24}
$$

The model first transposes it to channels-first form:

$$
X^T \in \mathbb{R}^{24 \times 36}
$$

For a single output channel in a dilated 1D convolution, the computation at time `t` is:

$$
h_t = b + \sum_{j=0}^{k-1} w_j \cdot x_{t - j d}
$$

Where:

- $k$ is the kernel size
- $d$ is the dilation
- $w_j$ are the learned kernel weights

With default `kernel_size = 3`:

- in block 1, dilation `d = 1`, so each convolution looks at `t, t-1, t-2`
- in block 2, dilation `d = 2`, so each convolution looks at `t, t-2, t-4`
- in block 3, dilation `d = 4`, so each convolution looks at `t, t-4, t-8`

Because there are two convolutions per block plus residual additions, the final representation at each time step aggregates information from up to 29 historical positions.

The default forward path for one sample is:

$$
(36, 24) \rightarrow (24, 36) \rightarrow (64, 36) \rightarrow (64, 36) \rightarrow (64, 36)
$$

Then temporal pooling gives:

$$
(64, 36) \rightarrow (64, 1) \rightarrow (64)
$$

If the series ID embedding is 8-dimensional, and this series has index `s`, then:

$$
e_s \in \mathbb{R}^{8}
$$

The concatenated representation is:

$$
[z ; e_s] \in \mathbb{R}^{72}
$$

The final dense layer computes:

$$
\hat{y}_{\text{scaled}} = W [z ; e_s] + b
$$

Where:

$$
\hat{y}_{\text{scaled}} \in \mathbb{R}^{3}
$$

Finally, the predictions are inverse-scaled per series:

$$
\hat{z} = \hat{y}_{\text{scaled}} \cdot \sigma_{\text{series}} + \mu_{\text{series}}
$$

$$
\hat{y} = \exp(\hat{z}) - 1
$$

So the model outputs the next 3 forecasts in the original unit scale.

### Important interpretation note

There are two valid ways to talk about the 70-row example:

1. If `70` means **raw monthly rows before feature warm-up**, this implementation yields `58` usable rows and `20` windows per series.
2. If `70` means **already feature-complete rows**, then the window count would be:

$$
70 - 36 - 3 + 1 = 32
$$

This document uses the first interpretation because that is how the code processes raw input data.

## 4. What the Architecture Learns

The architecture learns parameters in the neural network. It does **not** learn the preprocessing rules.

### Learned parameters

The following are learned during training.

#### Convolution kernels

Every `Conv1d` layer learns:

- kernel weights
- bias terms when present
- weight-normalization parameters associated with the layer

These filters learn temporal patterns such as:

- short-term changes
- delayed effects
- repeated local motifs
- interactions across engineered features

#### Residual projection

When a block changes channel width, the `1 x 1` residual convolution learns how to project the input into the new channel space.

#### Series embedding

The embedding table learns a dense vector for each series ID.

This captures series-specific behavior that is not fully explained by the engineered features alone, such as:

- baseline level tendencies
- persistent shape differences between series
- series-specific offsets in forecast behavior

#### Final linear head

The output layer learns how to combine:

- pooled temporal features from the TCN
- the series embedding

to produce the final `horizon` outputs.

### What is not learned by the neural network

The following are fixed pipeline choices, not learned neural parameters:

- monthly resampling
- missing-value filling rules
- outlier handling rules
- feature engineering definitions
- lag choices
- rolling-window definitions
- `window_size`
- `horizon`
- train validation test split boundaries
- feature scaling procedure
- per-series target scaling procedure

Those choices shape the learning problem, but the neural network does not optimize them.

### Mermaid neural-network diagram

```mermaid
flowchart LR
    A[Input window<br/>batch x 36 x F] --> B[Transpose<br/>batch x F x 36]
    B --> C[TemporalBlock 1<br/>Conv1d dil=1 x2<br/>Residual]
    C --> D[TemporalBlock 2<br/>Conv1d dil=2 x2<br/>Residual]
    D --> E[TemporalBlock 3<br/>Conv1d dil=4 x2<br/>Residual]
    E --> F[AdaptiveAvgPool1d 1]
    F --> G[Squeeze<br/>batch x 64]
    H[Series ID] --> I[Embedding<br/>batch x 8]
    G --> J[Concatenate]
    I --> J
    J --> K[Linear head<br/>72 to 3]
    K --> L[Scaled forecast horizon]
```

## Summary

This repository uses a multi-series TCN with:

- causal dilated `Conv1d` residual blocks
- exponential dilation growth
- adaptive temporal pooling
- learned series embeddings
- a linear multi-step forecast head

Under the current default configuration, the model sees 36 time steps per sample, has a receptive field of 29 steps, and predicts the next 3 monthly values.
