## Langchain

#### General points

- Parallel RAG Data Retrieval -> from langchain_core.runnables import RunnableParallel, RunnablePassthrough -> This pattern is used to run independent tasks at the same time. In a RAG application, you often want to fetch relevant documents from a vector store and preserve the user's original question simultaneously before passing both to the prompt.
  
- LangChain Expression Language (LCEL) is a declarative way to compose and chain together modular AI components like prompts, models, and output parsers using the native pipe operator (|)

#### Text Splitters 
- LangChain offers several splitters for different content types. In your LangChain 0.3.x setup, the dedicated package is already listed in requirements, so prefer imports from langchain_text_splitters:

CharacterTextSplitter: splits on a chosen separator, such as paragraphs or newlines. Simple, but may produce chunks that exceed the size limit.
TokenTextSplitter: splits by model tokens, useful when you need to stay within an LLM’s token budget.
MarkdownHeaderTextSplitter: splits Markdown by headings and can preserve heading metadata.
HTMLHeaderTextSplitter: splits HTML by heading structure.
RecursiveJsonSplitter: splits nested JSON while trying to preserve its structure.
RecursiveCharacterTextSplitter.from_language(...): adapts recursive separators for languages such as Python, JavaScript, or Markdown.


## Langgraph
