                    AZURE
                      │
          ┌───────────┴────────────┐
          │                        │
     APP SERVICE                 AKS
          │                        │
    App Service Plan          Kubernetes
          │                        │
     ┌────┴────┐             ┌─────┴─────┐
     ↓         ↓             ↓           ↓
   Web App   Function      Pods        Pods
              App

## A concrete example

Suppose you build:

> `POST /predict` → your ML model → prediction

#### Using App Service:

Azure Subscription
│
└── Resource Group: ml-prod
     │
     ├── App Service Plan
     │     │
     │     └── Web App: prediction-api
     │
     ├── Storage Account
     │
     └── Key Vault

So, Your Python code:
FastAPI
   │
   └── /predict
          │
          └── ML model

runs inside: 

Web App
   ↓
App Service Plan
   ↓
Azure-managed compute

#### Using Azure Functions:

Azure Subscription
│
└── Resource Group: ml-prod
     │
     ├── Function App: prediction-functions
     │      │
     │      ├── predict()
     │      ├── health()
     │      └── retrain()
     │
     ├── Hosting / App Service Plan
     │
     ├── Storage Account
     │
     └── Key Vault

That's the **serverless** idea.

#### Using AKS

Resource Group
│
├── AKS Cluster
│    │
│    ├── Node 1
│    │    ├── Pod: API
│    │    └── Pod: model-server
│    │
│    └── Node 2
│         ├── Pod: agent
│         └── Pod: worker
│
├── Container Registry
│
├── Key Vault
└── Storage

Another node is trying to represent if your application has another micorservice too.

