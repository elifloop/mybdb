

#### EPIC-001 ####

Goal - ft an SLM like qwen 7B which is not built to do turns, loops & tool calls to do all

Procedure - 
Step 1: Define the trajectory format
Step 2: Build the trajectory dataset (the hard part)
Step 3: Supervised fine-tuning (SFT)
Step 4: Preference tuning (optional but common)
Step 5: Format alignment with your specific harness
Step 6: Evaluate on held-out multi-step tasks, not just loss

Bottomline - For a 7B model:** the ceiling here is real — a 7B model has limited capacity to hold long reasoning chains and recover from its own mistakes across many turns, so even a well-executed version of this pipeline will plateau well below what Claude gets you on the same harness. It's a reasonable and increasingly well-trodden path (this is literally how open agentic coding models like GLM-4.x-flash or Qwen3-Coder's tool-enabled checkpoints get built) — but expect diminishing returns past a certain trajectory volume for a model this small.

Details - **Trajectory format & environment**

- ReAct format (thought/action/observation)
- Tool-call schema design (JSON vs code-style function calls)
- Sandboxed execution environments (Docker/gVisor for safe code running)

**Dataset construction**

- Task/prompt collection (seed tasks, Self-Instruct, Evol-Instruct)
- Knowledge distillation from teacher LLMs
- Multi-agent trajectory synthesis (role-based generation, e.g. MapCoder-style)
- Rejection sampling / pass-based filtering (unit-test verification)
- LLM-as-judge filtering
- Existing public agentic corpora (AgentInstruct, FireAct, AgentBank, Agent-FLAN, ToolACE, APIGen)

**Supervised fine-tuning (SFT)**

- Loss masking (mask observations/tool outputs, train only on model turns)
- LoRA / QLoRA / parameter-efficient fine-tuning
- Long-context/window handling for multi-turn trajectories
- Training frameworks (Hugging Face TRL, Unsloth, Axolotl)

**Preference tuning**

- DPO / IPO preference optimization
- Preference-pair generation via judge models
- Reward modeling / RLHF basics
- RL for agents (PPO/GRPO on task success reward)

**Harness/format alignment**

- Function-calling API conventions (OpenAI/Anthropic-style schemas)
- Ollama/vLLM tool-calling compatibility layers
- Prompt/template matching between training and inference harness

**Evaluation**

- Berkeley Function-Calling Leaderboard (BFCL) methodology
- Agentic coding benchmarks (SWE-bench, terminal-agent benchmarks)
- Multi-turn/tool-use eval metrics (task success rate, calls-to-completion, error recovery)


#### EPIC-002 ####

Goal - train an SLM from scratch 


data/processed/
├── tinystories_train.jsonl
├── tinystories_val.jsonl
├── python_train.jsonl
├── python_val.jsonl
├── instruction_train.jsonl
├── instruction_val.jsonl
├── codealpaca_train.jsonl
└── codealpaca_val.jsonl

             ↓
        1. Inspect data
             ↓
        2. Build tokenizer
             ↓
        3. Tokenize + pack
             ↓
        4. Build PyTorch Dataset
             ↓
        5. Build tiny GPT
             ↓
        6. Train
             ↓
        7. Evaluate
             ↓
        8. Generate text
             ↓
        9. Add Python/code training
             ↓
       10. Instruction tuning
             ↓
       11. Tool calling