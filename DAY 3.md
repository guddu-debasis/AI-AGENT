
# LangChain Agent Notes (Beginner Friendly)

# 1. Invoking an Agent

`agent.invoke()` is the simplest way to interact with an agent. It sends your input, waits for the agent to finish all reasoning and tool calls, and then returns the final response.

## How it works

```text
User Message
      │
      ▼
Agent receives input
      │
      ▼
LLM thinks
      │
      ▼
(Optional) Calls tools
      │
      ▼
Gets tool results
      │
      ▼
Generates final answer
      │
      ▼
Returns response
```

## Example

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.utils.uuid import uuid7

agent = create_agent(
    model="google_genai:gemini-3.5-flash",
    tools=[],
    checkpointer=InMemorySaver(),
)

config = {
    "configurable": {
        "thread_id": str(uuid7())
    }
}

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What's the weather in San Francisco?"
            }
        ]
    },
    config=config
)
```

### What happens?
- The user message is added to the agent state.
- The LLM decides whether it needs tools.
- If tools exist, they are executed.
- The agent creates a final response.
- `invoke()` returns only the final output.

---

# 2. Conversation History using `thread_id`

A `thread_id` identifies a conversation.

If you reuse the same `thread_id`, the agent remembers previous messages.

```python
config = {
    "configurable": {
        "thread_id": str(uuid7())
    }
}
```

### First request

```python
agent.invoke(
    {
        "messages":[
            {
                "role":"user",
                "content":"What's the weather in San Francisco?"
            }
        ]
    },
    config=config
)
```

### Second request

```python
agent.invoke(
    {
        "messages":[
            {
                "role":"user",
                "content":"What about tomorrow?"
            }
        ]
    },
    config=config
)
```

Because the same `thread_id` is used, the agent understands that "tomorrow" refers to San Francisco.

> A checkpointer (like `InMemorySaver`) is required for conversation history when running locally.

---

# 3. Runtime Context

`context` is additional information available during a run.

Unlike `thread_id`, it is **not** conversation history.

Example:

```python
from dataclasses import dataclass

@dataclass
class Context:
    user_id: str

agent = create_agent(
    model="google_genai:gemini-3.5-flash",
    tools=[],
    context_schema=Context,
    checkpointer=InMemorySaver(),
)

agent.invoke(
    {
        "messages":[
            {
                "role":"user",
                "content":"What's my balance?"
            }
        ]
    },
    config={"configurable":{"thread_id":str(uuid7())}},
    context=Context(user_id="user-123")
)
```

The tool can access `runtime.context.user_id` to identify the user.

---

# 4. Streaming

`invoke()` waits until everything finishes.

`stream_events()` shows progress while the agent is working.

```python
from langchain.messages import AIMessage, HumanMessage

stream = agent.stream_events(
    {
        "messages":[
            {
                "role":"user",
                "content":"Search AI news and summarize it"
            }
        ]
    },
    version="v3"
)

for snapshot in stream.values:
    latest_message = snapshot["messages"][-1]

    if latest_message.content:
        if isinstance(latest_message, HumanMessage):
            print("User:", latest_message.content)

        elif isinstance(latest_message, AIMessage):
            print("Agent:", latest_message.content)

    elif latest_message.tool_calls:
        print("Calling tools:",
              [tc["name"] for tc in latest_message.tool_calls])
```

### Output Flow

```text
User:
Search AI news

↓

Calling tools:
Search

↓

Agent:
Reading articles...

↓

Agent:
Summarizing...

↓

Final Answer
```

---

# invoke() vs stream_events()

| Feature | invoke() | stream_events() |
|----------|----------|-----------------|
| Returns | Final response | Intermediate updates + final response |
| Shows tool calls | ❌ | ✅ |
| Shows progress | ❌ | ✅ |
| Best for | Simple apps | Interactive apps |

---

# 5. Configuring the Agent Harness

The **Agent Harness** is the environment around an agent that provides additional capabilities.

Instead of only generating text, an agent can:
- Use tools
- Remember information
- Save files
- Retry failures
- Apply safety checks
- Ask for human approval

These capabilities are added using **middleware**.

---

# 6. Middleware

Middleware is a reusable component that adds one capability to an agent.

```python
agent = create_agent(
    model="google_genai:gemini-3.5-flash",
    tools=[search],
    middleware=[
        FilesystemMiddleware(...)
    ]
)
```

Benefits:
- Modular
- Easy to reuse
- Easy to combine

---

# 7. Middleware Categories

## Execution Environment
Provides a workspace.

Examples:
- Filesystem
- Python interpreter
- Shell execution
- Sandboxes

## Context Management

Helps manage long conversations.

Examples:
- Summaries
- Memory
- Prompt caching
- Skills

## Planning and Delegation

Allows large tasks to be split into smaller ones or delegated to subagents.

## Fault Tolerance

Improves reliability.

Examples:
- Retry failed API calls
- Fallback models
- Tool call limits

## Guardrails

Protects the application.

Examples:
- Detect PII
- Block unsafe content
- Apply safety policies

## Steering

Allows human approval before important actions.

---

# 8. FilesystemMiddleware

Allows the agent to read and write files.

Example from the documentation:

```python
from langchain.agents import create_agent
from deepagents.backends import StateBackend
from deepagents.middleware import FilesystemMiddleware

agent = create_agent(
    model="google_genai:gemini-3.5-flash",
    tools=[search],
    middleware=[
        FilesystemMiddleware(
            backend=StateBackend()
        )
    ]
)
```

## What each part does

`FilesystemMiddleware`
- Gives the agent a virtual file system.

`StateBackend`
- Stores files inside the agent state.
- Makes files available across conversation turns.

Flow:

```text
User
  │
  ▼
Search documents
  │
  ▼
Search Tool
  │
  ▼
FilesystemMiddleware
  │
  ▼
Save report.txt
  │
  ▼
Agent responds
```

---

# Quick Revision

- `invoke()` → wait and get the final response.
- `stream_events()` → receive updates while the agent works.
- `thread_id` → preserves conversation history.
- `context` → passes runtime data to tools.
- Middleware → extends the agent with new capabilities.
- `FilesystemMiddleware` → enables file operations.
- `StateBackend` → stores those files in the agent state.
