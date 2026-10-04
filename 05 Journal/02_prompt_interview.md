# Langchain

- Parallel RAG Data Retrieval -> from langchain_core.runnables import RunnableParallel, RunnablePassthrough -> This pattern is used to run independent tasks at the same time. In a RAG application, you often want to fetch relevant documents from a vector store and preserve the user's original question simultaneously before passing both to the prompt.
  - 
- LangChain Expression Language (LCEL) is a declarative way to compose and chain together modular AI components like prompts, models, and output parsers using the native pipe operator (|)
