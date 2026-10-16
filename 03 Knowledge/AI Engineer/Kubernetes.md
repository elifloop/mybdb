# Kubernetes in AIM+ (AKS)

Scope: manifests, pipelines and variables across `backend_aim_plus`, `aim_plus`, `ui_aim_plus`, `de_aim_plus`, `aim_documents/patient_plus_help`. `agent_aim_plus` contains no Kubernetes content.
Platform: **Azure Kubernetes Service (AKS)**, images in **Azure Container Registry (ACR)**, deployed by **Azure DevOps** pipelines using `KubernetesManifest@1`. Secrets/values are token-substituted (`#{var}#`) at deploy time; no Helm, no Ingress manifests in these repos (the `/help` manifest states it is routed by the host app's ingress, which lives elsewhere).

---

## 1. THEORETICAL FOUNDATIONS

### The "Why" and "What"
- **Problem:** run ~12 independently releasable services (APIs, frontend, timer, function app, GenAI backend) with self-healing, horizontal scale, zero-downtime rollouts and per-environment config (DEV/UAT/PROD).
- **Kubernetes primitives used here:** `Deployment` (desired replica state, rolling updates), `Service` (ClusterIP stable virtual IP), `HorizontalPodAutoscaler` (autoscaling/v2), `ConfigMap` (env config), `ServiceAccount` + Azure Workload Identity (keyless Azure auth), probes (liveness/readiness/startup), `securityContext` (hardening), `emptyDir`/projected volumes.

### Formulations (control loops)
- **HPA desired replicas:**
  $$\text{desired} = \left\lceil \text{current} \times \frac{\text{currentMetric}}{\text{targetMetric}} \right\rceil$$
  clamped to `[minReplicas, maxReplicas]`; with multiple metrics (CPU and memory here) the **max** of the per-metric results wins. Utilization is relative to the container **request**, not the limit.
- **Rolling update bounds:** with `replicas=N`, `maxSurge=25%` and `maxUnavailable=25%` → up to $\lceil 1.25N \rceil$ pods exist and at least $\lfloor 0.75N \rfloor$ are available during rollout. For N=2: surge=1, unavailable=0 (rounding: surge rounds up, unavailable rounds down).
- **Probe worst-case detection time:** `initialDelaySeconds + periodSeconds × failureThreshold`.

### Architectural Breakdown
```
Azure DevOps CI (az acr build) ──► ACR image + docker-tag.txt artifact
Azure DevOps CD:  ENVIRONMENT param ► variables/<env>-deploy-variables.yml
   ► AzureKeyVault@2 (managed identity client id) ► replace-token-in-files.yml
   ► KubernetesManifest@1 deploy ► AKS namespace
Pod:  Deployment ─ ReplicaSet ─ Pod(s) ◄── Service (ClusterIP :80 → targetPort)
        ▲ HPA scales replicas        ▲ envFrom ConfigMap   ▲ SA ↔ Azure Workload Identity
```

### Trade-offs & Alternatives
| Choice | Strength | Weakness / alternative |
|---|---|---|
| Raw manifests + `#{token}#` replacement | Simple, transparent | No templating logic/validation; Helm/Kustomize give overlays and diffing |
| ClusterIP only | Minimal attack surface | Needs an external ingress/gateway for traffic (not in these repos) |
| Resource-metric HPA | Zero extra infra | Poor for queue/LLM-latency-bound load; KEDA scales on events/queues |
| Deployment for timer | Easy | A scheduler as a Deployment (replicas=1) is a SPOF; `CronJob` is the native alternative |

---

## 2. PRACTICAL & ENGINEERING IMPLEMENTATION

### 2.1 Workload inventory

| Workload (name) | Repo / file | Port | Replicas / HPA (CPU%, Mem%) | Health path | Resources (req → limit) |
|---|---|---|---|---|---|
| `aim-plus-genai-api` | `backend_aim_plus/AIMServices/GenAIService/devops/k8s/deployment-genai.yaml` | 7071 | 2 / 2–5 (80, 75) | `/api/health` | tokens `cpu_request`… |
| Workbook, DataTabColumn, InsightsLibrary, Filter, Reporting, ServiceAlerts, Admin services | `backend_aim_plus/AIMServices/*/devops/k8s/deployment-*.yaml` | per service | 2 / 2–5 (80, 75) | `/api/health` | tokens |
| `aim-plus-schedule-timer` | `…/ScheduleTimerService/devops/k8s/deployment-dynamic-prompts-timer.yaml` | – | 1 / **no HPA, no Service, no probes** | – | tokens |
| `aim-plus-function-api` | `backend_aim_plus/AIMFunctionApp/DevOps/k8s/deployment-function.yaml` | 9091 | 2 / 2–5 (80, 75) | `/api/health-check/health` | tokens |
| `aim-plus-genai-backend-ds-api-v2` | `aim_plus/GenAI/devops/k8s/deployment-genai-v2.yaml` | 8000 | 2 / 2–8 (65, 70) | `/health` | 1000m/1Gi → 1500m/2Gi |
| GenAI v1 (`aim_plus`) | `aim_plus/GenAI/devops/k8s/deployment-genai.yaml` | – | 2 / 2–5 (70, 75) | `/health` | 500m/1Gi → 1000m/1.5Gi |
| `aim-plus-frontend` (Next.js) | `ui_aim_plus/devops/k8s/deployment-frontend.yaml` | 3000 | 2 / 2–5 (70, 75) | `/` | 500m/1Gi → 1000m/1.5Gi |
| `patient-plus-dev-help` | `aim_documents/patient_plus_help/linked_services/devops/k8s/deployment-dev-help.yaml` | 8080 | 1 / `hpa_min_replicas`–3 (70, CPU only) | `/help/api/health` | 100m/256Mi → 500m/512Mi |
| Self-hosted ADO agent | `de_aim_plus/DevOps/dockeragent/aimplus-adoagent-aks-deploy.yaml` | – | 1 | – | – |

Every Service is `ClusterIP`, port `80` → container port. Backend services request/limit values come from per-env variable files (DEV example: `cpu 150m→400m`, `memory 500Mi→768Mi`).

### 2.2 Code walkthrough with excerpts

**Workload Identity (keyless Azure access)** — `deployment-genai.yaml` L16-21, L42-44:
```yaml
        app: aim-plus-genai-api
        azure.workload.identity/use: "true"

    spec:
      serviceAccountName: #{service_account_managed_identity}#
...
          env:
            - name: AZURE_CLIENT_ID
              value: #{AZURE-MANAGED-IDENTITY-CLIENT-ID}#
```
The label makes the AKS webhook inject federated token env vars; `AZURE_CLIENT_ID` is fetched from Key Vault by the pipeline. `patient-plus-dev-help` additionally projects the token explicitly (`audience: api://AzureADTokenExchange`).

**Pod hardening** — `deployment-genai.yaml` L23-27, L37-40:
```yaml
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        seccompProfile:
          type: RuntimeDefault
...
          securityContext:
            allowPrivilegeEscalation: false
            capabilities:
              drop: ["ALL"]
```
The dev-help manifest goes further: `readOnlyRootFilesystem: true` with `emptyDir` at `/tmp`. GenAI v2 mounts `emptyDir` at `/tmp` and `/app/cache` (`sizeLimit: 1Gi`).

**Rollout strategy** — `deployment-genai.yaml` L83-90:
```yaml
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 25%
      maxSurge: 25%

  revisionHistoryLimit: 10
  progressDeadlineSeconds: 600
```
Dev-help uses `maxUnavailable: 0`, `maxSurge: 1` (strict zero-downtime for a single replica) and `terminationGracePeriodSeconds: 60` so streamed LLM answers finish.

**HPA** — `deployment-genai.yaml` L103-117 (CPU 80%, memory 75%, 2–5). GenAI v2 adds slow scaling (`aim_plus/GenAI/devops/k8s/deployment-genai-v2.yaml`):
```yaml
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 25
        periodSeconds: 60
```

**Probes for slow-starting LLM service** — GenAI v2 uses `startupProbe` (`failureThreshold: 10`, `periodSeconds: 30`), readiness `initialDelaySeconds: 120`, liveness `initialDelaySeconds: 180`, all on `/health:8000`. Dev-help startup probe allows ~190 s (`failureThreshold: 18`, period 10 + delay 10) for Key Vault secret loading.

**Config** — services use `envFrom: configMapRef: name: #{configMap}#` (e.g. DEV `aim-plus-backend-config-tst`). The frontend manifest *creates* its own `ConfigMap` containing `NEXT_*`/`GRAPH_*` values, **including `NEXT_CLIENT_SECRET` and `GRAPH_CLIENT_SECRET`** (see gotchas).

**CD pipeline** — `backend_aim_plus/AIMServices/GenAIService/devops/pipeline/deploy/aim-plus-genai-core.yml`:
```yaml
parameters:
  - name: ENVIRONMENT
    ...
    values:
      - DEV
      - UAT
      - PROD
...
                - task: KubernetesManifest@1
                  inputs:
                    action: 'deploy'
                    connectionType: 'kubernetesServiceConnection'
                    kubernetesServiceConnection: $(k8s_service_connection_name)
                    namespace: $(k8s_namespace)
```
Guard (L59-70): UAT/PROD cannot be deployed with `DockerTag=latest` when the CI branch is `refs/heads/develop`. Build side: `az acr build -t "$(DOCKER_TAG)" --registry $(acrName)`; the tag is passed via the `dockerTag` pipeline artifact. Shared templates: `backend_aim_plus/shared-components/replace-token-in-files.yml`.

**Environment variables (DEV)**: `k8s_namespace: md2i-aim-plus-test`, `k8s_service_connection_name: md2i-aim-plus-test`, `acr_base_path: rxgblacrdevmd2i.azurecr.io`, `service_account_managed_identity: aimplus-workload-sa-dev`, agent pool `AIMPLUS-AKS-POOL-TST`.

**Build agents on AKS** — `de_aim_plus/DevOps/dockeragent/Dockerfile`: installs Docker CLI, `kubectl` (`ver_kubernetes="v1.29"` from `pkgs.k8s.io`) and Azure CLI; targets cluster `rx-g-all-md2i-AKS-devtest`, namespace `aimplus-aks-data-pipeline-pool`. The Databricks deploy pipelines in `de_aim_plus/DevOps/Databricks/` run on this pool.

### 2.3 Data/traffic flow ("shapes")
```
Client ► (external ingress, not in repo) ► Service :80 ► Pod :7071/:8000/:9091/:3000/:8080
Pod ► Azure resources via Workload Identity token (SA JWT → AAD token exchange)
Replicas: 2 baseline ► HPA 2..5 (v2 GenAI 2..8)  | capacity = replicas × limit
```

### 2.4 Performance, scale & bottlenecks
- Memory-heavy GenAI v2 (2Gi limit): memory HPA at 70% of a 1Gi request = 700Mi triggers scaling; OOMKill risk when limit/request ratio is 2×.
- Probe `timeoutSeconds: 30–60` on backend services masks hung pods for up to `30 + 20×3` s.
- `imagePullPolicy: Always` + mutable tags (`latest`) adds pull latency and breaks reproducibility.
- The cluster autoscaler must have headroom: max 5–8 replicas × ~12 services.

### 2.5 Production considerations & telemetry
Monitor: pod restarts/OOMKills, HPA `currentReplicas` vs `maxReplicas` (pinned at max = under-provisioned), p95 latency, probe failures, rollout `ProgressDeadlineExceeded`, ACR pull errors, node pressure. Frontend wires Application Insights (`NEXT_PUBLIC_APPLICATIONINSIGHTS_CONNECTION_STRING`).

Useful commands:
```bash
kubectl -n <ns> get deploy,hpa,svc,pods
kubectl -n <ns> rollout status deploy/aim-plus-genai-api
kubectl -n <ns> describe pod <pod>        # probe/OOM events
kubectl -n <ns> logs deploy/aim-plus-genai-api --tail=200
kubectl -n <ns> rollout undo deploy/aim-plus-genai-api
kubectl -n <ns> top pods
```

---

## 3. TECHNICAL INTERVIEW PREPARATION

### System design
1. **Scale the GenAI API to 10k req/s.** Checklist: profile per-pod RPS and set requests accordingly; HPA on custom/external metrics (RPS, queue depth via KEDA) not just CPU/memory; raise `maxReplicas`, enable cluster autoscaler/node pools (GPU or high-mem); add ingress/Application Gateway with rate limiting; cache and async/queue long LLM calls; PodDisruptionBudgets and topology spread across zones; startup probes so new pods are not killed while loading; load-test and set `behavior` to scale up fast and down slowly.
2. **Zero-downtime multi-environment release with safe rollback.** Checklist: immutable image tags (digest) instead of `latest`; same manifest per env with env variable files; `maxUnavailable: 0`; readiness gating; progressive delivery (blue/green or canary — v1/v2 GenAI already coexist with `version` labels); automated `rollout undo` on failed `progressDeadlineSeconds`; approvals on `DEPLOY_AIM_PLUS_PROD` environment; smoke tests post-deploy.

### Deep-dive questions
1. **Why can HPA flap even with CPU/memory targets?** *Star:* Situation: bursty LLM traffic. Task: stable scaling. Action: HPA takes max across metrics, so memory (not freed after load) can pin replicas high; use `behavior.stabilizationWindowSeconds` (300 s in v2) and percent policies. Result: no thrash, slower scale-down.
2. **Utilization of what?** *Star:* HPA % is of the container **request**. Services here use tokenized requests (150m DEV); a too-low request inflates utilization and over-scales. Fix: right-size from observed usage (VPA recommendations).
3. **How does a pod authenticate to Azure without secrets?** *Star:* Pod label `azure.workload.identity/use: "true"` + ServiceAccount federated to a managed identity; webhook injects `AZURE_FEDERATED_TOKEN_FILE`; SDK exchanges the projected SA token (audience `api://AzureADTokenExchange`) for an AAD token. No client secret stored.

### Gotchas & edge cases
1. **Secrets in a ConfigMap.** The frontend `ConfigMap` holds `NEXT_CLIENT_SECRET`/`GRAPH_CLIENT_SECRET` in plain text (and the rendered manifest is printed in the CD log by "Show Final Manifest"). Troubleshoot/fix: move to Kubernetes Secrets/CSI Key Vault driver; stop echoing manifests.
2. **Unreplaced `#{token}#`.** A missing pipeline variable yields an invalid manifest or literal values (e.g. empty `cpu_request`). Check the "Show Final Manifest" step and variable template for the chosen `ENVIRONMENT`.
3. **Probe/startup mismatch & single-replica timer.** Slow LLM/Key Vault startup kills pods in a loop without `startupProbe` (only GenAI v2 and dev-help have one); `aim-plus-schedule-timer` has 1 replica with no probes/HPA, so a hung pod silently stops scheduled work, and scaling it >1 would double-fire jobs. Diagnose via `kubectl describe pod` events and restart counts.
