## Section - 0 - My Intro

I'm Harsh, and I have over 14 years of experience across data analytics, data science and AI engineering. I'm currently a Principal AI Engineer at GSK, where I lead cross-functional pods of AI engineers, data engineers and full-stack engineers to take AI problem statements from ideation to MVP, typically within two to three weeks.
Some of my significant contribution was building multi agentic AI chatbot where I built components for RAG pipeline, designing as well as implemented the multi agentic workflow; then I built a forecasting framework using TCN which forecasts 700+ time series at scale & speed + hyperparameter tuning step is completely autonomously managed using karpathy loop. I have also performed finetuning of LLMs - BioBert to train language & qwen to train procedure. Other than this my prior experiences include cx segmentation, image classification, some statistical analysis like Kaplain Meiers to study email consent attrition. My domains experience include mainly commercial side of business like supplychain, marketing and sales but in general i am interested in understanding different areas business a lot.

gsk 
- agentified tcn forecasting 
- built multiagentic systems for patient analytics
- qwen llm ft for payload generation
- bert ft
- consent attrition using kaplan meier 

volvo 
- customer segmentation using xgb
- cnn image based dataset

## Section - 1 - todos & questiosn

- HL concepts in detail - add to `03 Knowledge`
- apply langraph to aim+ 
- apply streaming to aim+ 
- `apply collaborative filtering & add to resume`
- `apply mmm`
- system design

  
## Section - 2 - RAG

### HL components

  * ingestion - text quality scoring(mvoc quality); category classification model; NER; sentiment analysis using fuzzy wuzzy ; twitter/ social listening apis
  * retrieval - HyDE, Hybrid Fusion Scoring, reranking algos, HNSW, IVF, retrieval metrics
  * generation - AI models, context management, guardrails caching, genEval metrices
  * architecture - naive vs advanced vs modular, streaming SSE, agent evals, langgraph

### Application 🈸

* multi-agentic projects - askAIM (Advanced RAG) | PatientPlus (Modular RAG)
* automation projects - Forecasting Karpathy Loops (ReAct agentic)
* blenrep mvoc (naive RAG)


### Links
🔗 https://www.datacamp.com/blog/langchain-interview-questions

## Section - 2 - ML

### HL components

- [[Random Forest]]
- [[LightGBM]] 
- [[XGBoost]]
- [[CNN]]
- [[TCN]]

### Application 🈸

* Sales Panel forecasting (TCN)
* Cx Segmentation - growth (RF) + initiation (LightGBM) - [[01_OpEngine]]
* Cx propensity  - (XGBoost)
* Image classification - (CNN)
* Recommendation engine - 

## Section - 3 - AI 

### HL Components 

- architecture: intent classification -> query rewriter -> orchestrator -> analytics agent + web search agent + ... + self reflection agent -> response formatter
- finetuning: todo

### Application 

* patient+
* aim+
* dtrainer
* aim - fine tuning

## Section - 4 - NeGD Interview

#### Q's
- system design - weather app
- Why is memory required in langchain or langraph?
- When we should go for AI vs not AI ? - answer - procedural kb; cost factor
- write fetch vector index class to retrieve data from AI search or any other vector db (components - fetch, similarity search, rerank, eval logs etc.)
- explain a typical llm finetuning pipeline

