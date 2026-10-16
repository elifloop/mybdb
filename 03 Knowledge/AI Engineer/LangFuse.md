# Langfuse Integration Guide

## Overview

This application supports **Langfuse** integration for comprehensive LLM observability, tracing, and monitoring. Langfuse helps you track:
- LLM API calls and responses
- Token usage and costs
- Latency and performance
- Prompt versions
- User sessions
- Error tracking

## What is Langfuse?

**Langfuse** is an open-source LLM engineering platform that provides:
- 🔍 **Tracing**: Track every LLM call with full context
- 📊 **Analytics**: Monitor usage, costs, and performance
- 🧪 **Prompt Management**: Version and test prompts
- 👥 **User Tracking**: Associate calls with users/sessions
- 💰 **Cost Tracking**: Monitor token usage and expenses
- ⚡ **Performance**: Track latency and bottlenecks

**Website**: https://langfuse.com  
**Documentation**: https://langfuse.com/docs

## Installation

### 1. Install Langfuse Package

```powershell
pip install langfuse
```

Or add to your `requirements.txt`:
```txt
langfuse==2.6.3
```

### 2. Get Langfuse Credentials

#### Option A: Langfuse Cloud (Recommended for getting started)
1. Sign up at https://cloud.langfuse.com
2. Create a new project
3. Get your API keys from Project Settings

#### Option B: Self-Hosted
```bash
# Using Docker
docker run -d \
  -p 3000:3000 \
  --name langfuse \
  langfuse/langfuse
```

Visit http://localhost:3000 and create an account.

### 3. Configure Environment Variables

Add to your `.env` file:

```env
# Enable Langfuse
LANGFUSE_ENABLED=true

# Add your credentials
LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key-here
LANGFUSE_SECRET_KEY=sk-lf-your-secret-key-here

# Optional: Custom host (defaults to https://cloud.langfuse.com)
# LANGFUSE_HOST=https://your-langfuse-instance.com
```

## Features

### 1. **Automatic Log Integration**

All application logs (INFO level and above) are automatically sent to Langfuse when enabled:

```python
from utils.logging.logger import default_logger as logger

# This log will appear in both files and Langfuse
logger.info(
   "User action",
   extra={
      "request_id": "abc-123",
      "user_id": "user456"
   }
)
```

### 2. **LLM Call Tracking**

Use the `LLMLogger` for detailed LLM tracing:

```python
from utils.logging.logger import llm_logger

# Log LLM call with full context
llm_logger.log_llm_call(
   model="gpt-3.5-turbo",
   messages=[
      {"role": "user", "content": "Hello!"}
   ],
   response={
      "content": "Hi there!",
      "usage": {
         "prompt_tokens": 10,
         "completion_tokens": 5,
         "total_tokens": 15
      }
   },
   request_id="abc-123",
   user_id="user456",
   session_id="session789",
   metadata={
      "temperature": 0.7,
      "max_tokens": 500,
      "context_type": "pharma"
   }
)
```

### 3. **Error Tracking**

Track LLM-specific errors:

```python
try:
    result = llm_service.chat(messages)
except Exception as e:
    llm_logger.log_error(
        error=e,
        context={
            "model": "gpt-3.5-turbo",
            "message_count": len(messages)
        },
        request_id=request_id
    )
```

### 4. **Custom Logger with Langfuse**

Create a custom logger with Langfuse enabled:

```python
from utils.logging.logger import setup_logger_with_langfuse

# Create logger with Langfuse
logger = setup_logger_with_langfuse(
   name="my_service",
   log_level="INFO",
   enable_langfuse=True
)

logger.info("This goes to both files and Langfuse")
```

## Integration Examples

### Example 1: Enhanced Chat Endpoint

Update your chat endpoint to use LLM logging:

```python
from utils.logging.logger import llm_logger


@app.post("/api/chat")
async def chat(request: ChatRequest, req: Request):
   request_id = getattr(req.state, "request_id", "unknown")

   try:
      # Your LLM call
      llm_service = LLMService()
      response = await llm_service.chat(
         messages=messages,
         model=model,
         temperature=temperature
      )

      # Log to Langfuse
      llm_logger.log_llm_call(
         model=model,
         messages=messages,
         response=response,
         request_id=request_id,
         session_id=request.session_id,  # if available
         metadata={
            "temperature": temperature,
            "max_tokens": max_tokens,
            "context_type": request.context_type
         }
      )

      return response

   except Exception as e:
      llm_logger.log_error(e, request_id=request_id)
      raise
```

### Example 2: User Session Tracking

```python
from utils.logging.logger import LLMLogger


class ChatSession:
   def __init__(self, session_id: str, user_id: str):
      self.session_id = session_id
      self.user_id = user_id
      self.llm_logger = LLMLogger()

   async def send_message(self, message: str):
      response = await llm_service.chat([{"role": "user", "content": message}])

      # Track in Langfuse with session context
      self.llm_logger.log_llm_call(
         model="gpt-3.5-turbo",
         messages=[{"role": "user", "content": message}],
         response=response,
         session_id=self.session_id,
         user_id=self.user_id
      )

      return response
```

## Langfuse Dashboard Features

### What You'll See in Langfuse:

1. **Traces Tab**
   - All LLM calls
   - Request/response pairs
   - Token usage
   - Latency metrics
   - Error rates

2. **Sessions Tab**
   - User conversations
   - Session duration
   - Message count
   - User activity

3. **Users Tab**
   - User-specific metrics
   - Cost per user
   - Usage patterns

4. **Analytics**
   - Cost trends
   - Token usage over time
   - Model performance
   - Error analysis

5. **Prompts**
   - Prompt versions
   - A/B testing
   - Performance comparison

## Benefits for PharmaDA Chat

### 1. **Cost Monitoring**
Track API costs per:
- User
- Session
- Context type (pharma, clinical_trial, etc.)
- Model

### 2. **Performance Optimization**
- Identify slow prompts
- Optimize token usage
- Compare model performance

### 3. **Quality Assurance**
- Track response quality
- Monitor error rates
- A/B test prompts

### 4. **Compliance & Auditing**
- Full audit trail of LLM interactions
- User activity tracking
- Session replay capability

### 5. **Development Insights**
- Debug production issues
- Understand user behavior
- Improve prompts based on data

## Configuration Options

### Environment Variables

```env
# Enable/Disable
LANGFUSE_ENABLED=true              # true/false

# Authentication
LANGFUSE_PUBLIC_KEY=pk-lf-...     # Your public key
LANGFUSE_SECRET_KEY=sk-lf-...     # Your secret key

# Endpoint (optional)
LANGFUSE_HOST=https://cloud.langfuse.com  # Or self-hosted URL
```

### Programmatic Configuration

```python
from utils.logging.logger import LLMLogger

# Create logger with custom Langfuse instance
llm_logger = LLMLogger()
llm_logger.langfuse.host = "https://my-langfuse.com"
```

## Best Practices

### 1. **Privacy & Security**
```python
# DON'T log sensitive data
llm_logger.log_llm_call(
    messages=[{"role": "user", "content": "Patient data: ..."}],  # ❌ Bad
    ...
)

# DO sanitize before logging
sanitized_messages = sanitize_phi(messages)
llm_logger.log_llm_call(
    messages=sanitized_messages,  # ✅ Good
    ...
)
```

### 2. **Use Session IDs**
Always track session_id for conversation continuity:
```python
llm_logger.log_llm_call(
    ...,
    session_id=session_id,  # Track conversation flow
)
```

### 3. **Add Rich Metadata**
```python
llm_logger.log_llm_call(
    ...,
    metadata={
        "context_type": "clinical_trial",
        "user_role": "researcher",
        "environment": "production",
        "feature": "adverse_event_analysis"
    }
)
```

### 4. **Cost Tracking**
Monitor costs by adding tags:
```python
metadata = {
    "department": "research",
    "project": "trial_analysis_2026",
    "budget_code": "R-123"
}
```

## Troubleshooting

### Issue: Langfuse not sending data

**Check:**
1. `LANGFUSE_ENABLED=true` in .env
2. Valid API keys
3. Network connectivity to Langfuse host
4. Check console for initialization messages

**Solution:**
```powershell
# Test connection
python -c "from langfuse import Langfuse; lf = Langfuse(); print('✅ Connected')"
```

### Issue: "Module not found: langfuse"

**Solution:**
```powershell
pip install langfuse
```

### Issue: Duplicate traces

**Solution:**
Ensure you're not calling `log_llm_call` multiple times for the same request.

### Issue: Missing user/session data

**Solution:**
Always pass `user_id` and `session_id` when available:
```python
llm_logger.log_llm_call(
    ...,
    user_id=user_id,      # Add this
    session_id=session_id  # Add this
)
```

## Performance Impact

- **Overhead**: < 5ms per log entry
- **Async**: Langfuse sends data asynchronously
- **Retry**: Automatic retry on network failures
- **No blocking**: Errors in Langfuse don't affect your app

## Disabling Langfuse

To disable Langfuse:

**Option 1: Environment Variable**
```env
LANGFUSE_ENABLED=false
```

**Option 2: Comment out in .env**
```env
# LANGFUSE_ENABLED=true
```

**Option 3: Uninstall**
```powershell
pip uninstall langfuse
```

The application will work normally without Langfuse installed.

## Advanced Features

### Custom Traces

```python
if llm_logger.langfuse:
    trace = llm_logger.langfuse.trace(
        name="complex_workflow",
        user_id=user_id,
        metadata={"workflow": "multi_step"}
    )
    
    # Step 1
    trace.generation(name="step1", ...)
    
    # Step 2
    trace.generation(name="step2", ...)
```

### Prompt Management

```python
# Use versioned prompts
prompt = llm_logger.langfuse.get_prompt("pharma_analysis_v2")
messages = prompt.compile(drug_name="Aspirin")
```

### Scoring & Feedback

```python
# Add quality scores
trace.score(
    name="quality",
    value=0.95,
    comment="Excellent response"
)
```

## Resources

- **Langfuse Docs**: https://langfuse.com/docs
- **Python SDK**: https://langfuse.com/docs/sdk/python
- **Tracing Guide**: https://langfuse.com/docs/tracing
- **Dashboard Guide**: https://langfuse.com/docs/analytics

## Example Queries in Langfuse

### Cost by Context Type
```sql
SELECT 
  metadata->>'context_type' as context,
  SUM(cost) as total_cost,
  COUNT(*) as call_count
FROM traces
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY context
```

### Average Latency by Model
```sql
SELECT 
  model,
  AVG(latency_ms) as avg_latency,
  COUNT(*) as calls
FROM generations
GROUP BY model
```

## Summary

✅ **Easy to Enable**: Just set environment variables  
✅ **Zero Impact When Disabled**: Works without Langfuse  
✅ **Full Observability**: Track every LLM interaction  
✅ **Cost Monitoring**: Know exactly what you're spending  
✅ **Production Ready**: Async, retries, error handling  
✅ **Privacy Aware**: You control what gets logged  

**Langfuse integration gives you production-grade LLM observability with minimal code changes!**
