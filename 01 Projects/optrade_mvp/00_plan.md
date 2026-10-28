

### hypothesis 

time bounded correlation/ cointegration between returns & market data

### HL components

##### Architecture

'''mermaid
flowchart LR
    A[1. Data Layer<br/>market + alt data<br/>point-in-time] --> B[2. Research Platform<br/>features, signals,<br/>hypotheses]
    B --> C[3. Backtesting & Simulation<br/>costs, slippage,<br/>Monte Carlo]
    C --> D[4. Portfolio & Risk<br/>sizing, limits,<br/>drawdown control]
    D --> E[5. Execution<br/>routing, cost model,<br/>broker connectivity]
    E --> F[6. Production Infra<br/>deploy, schedule,<br/>monitor]
    F --> G[7. Post-trade Analysis<br/>P&L attribution,<br/>signal decay]
    G -->|feedback| B

    H[8. Governance & Compliance<br/>validation, audit trail,<br/>kill switch] -.-> C
    H -.-> D
    H -.-> E
    H -.-> F

    I[9. People & Process<br/>idea → test → review →<br/>deploy → retire] -.-> B
    I -.-> G
'''

Layer 1 — Data
Bloomberg / Refinitiv / CRSP / Compustat / FRED / alternative data

Layer 2 — Statistical research
Python
- pandas
- numpy
- statsmodels
- sklearn
- scipy
- PyMC
- vectorbt / Backtrader
  
Layer 3 — Quant model
Factor model / alpha model / volatility model / portfolio optimizer

Layer 4 — Grey Machines
Scenario engine
- causal relationships
- feedback loops
- parameter uncertainty
- external shocks
- Monte Carlo
- scenario clustering
- sensitivity analysis

Layer 5 — Decision
Expected return
Volatility
VaR
Expected Shortfall
Drawdown
Scenario probabilities
Tail exposure
Key risk drivers


## LL Components

Layer 1 — Data
	- ohlc data 
	- market data
		- bcg {in progress}
		
- data science 
	- research algorithm - in progress - /home/harsh/Documents/projects/optrade/optrade-datascience/study

