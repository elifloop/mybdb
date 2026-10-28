

### hypothesis to test

not clear yet

### HL components

##### Architecture

```mermaid
flowchart LR
    A[1. Data Layer<br/>market + alt data<br/>point-in-time] --> B[2. Research Platform<br/>features, signals,<br/>hypotheses]
    B --> C[3. Backtesting & Simulation<br/>costs, slippage,<br/>Monte Carlo]
    C --> D[4. Portfolio & Risk<br/>sizing, limits,<br/>drawdown control]
    D --> E[5. Execution<br/>routing, cost model,<br/>broker connectivity]
    E --> F[6. Post-trade Analysis<br/>P&L attribution,<br/>signal decay]

```
##### 1. data Layer
Bloomberg / Refinitiv / CRSP / Compustat / FRED / alternative data / adhoc docs

##### 2. QR Layer
Stats/ DL/ ML
Factor model / alpha model / volatility model / portfolio optimizer

##### 3. backtest Layer
- Scenario engine, causal relationships, feedback loops, parameter uncertainty, external shocks,scenario clustering, sensitivity analysis
- ** Monte Carlo simulation **

##### 4. risk layer
Expected return, Volatility, VaR, Expected Shortfall, Drawdown, Scenario probabilities, Tail exposure, Key risk drivers

##### 5. execution layer
- NL based output
- trade trigger zerodha apis

##### 6. retrospective layer
- Reconciliation, P&L attribution and signal decay tracking. It shows whether live results match backtests and feeds back into research.

##### 7. 
