from src.tools import TOOLS
from src.prompts import SYSTEM_PROMPT
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
import operator, os, sys
from dotenv import load_dotenv
load_dotenv()

# Force UTF-8 output on Windows to avoid UnicodeEncodeError from cp1252
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

def safe_print(*args, **kwargs):
    """print() that never crashes on unencodable characters."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = " ".join(str(a) for a in args) + "\n"
        sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
        sys.stdout.buffer.flush()

class AgentState(TypedDict):
    question: str
    messages: Annotated[list, operator.add]
    final_answer: str
    action: str
    tool_input: str
    
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
import time


def extract_text(content) -> str:
    if isinstance(content, list):
        return content[0].get('text', '') if isinstance(content[0], dict) else str(content[0])
    return content


llm = ChatGroq(
    model='openai/gpt-oss-120b',
    api_key=os.getenv("GROQ_API_KEY")
)

def call_llm(prompt, retries=3):
    for attempt in range(retries):
        try:
            return llm.invoke(prompt)
        except Exception as e:
            if attempt < retries - 1:
                safe_print(f"LLM Error, retrying in 10s... ({e})")
                time.sleep(10)
            else:
                raise

def agent_node(state: AgentState):
  import re

  messages = [SystemMessage(content=SYSTEM_PROMPT)] + state['messages']

  response = call_llm(messages)
  content = response.content
  reply = extract_text(content).strip()

  # Check if Action: ANSWER is present anywhere in the reply
  answer_match = re.search(r'(?:\*\*|\b)Action(?:\*\*|\b)?\s*:\s*ANSWER\b', reply, re.IGNORECASE)
  if answer_match:
      after_answer = reply[answer_match.end():].strip()
      input_match = re.search(r'(?:\*\*|\b)Input(?:\*\*|\b)?\s*:\s*', after_answer, re.IGNORECASE)
      if input_match:
          final_ans = after_answer[input_match.end():].strip()
      else:
          final_ans = after_answer
      action = "ANSWER"
      inp = final_ans
  else:
      # Check for tool actions: retrieve or github_search
      tool_match = re.search(r'(?:\*\*|\b)Action(?:\*\*|\b)?\s*:\s*(retrieve|github_search)\b', reply, re.IGNORECASE)
      if tool_match:
          action = tool_match.group(1).lower()
          after_action = reply[tool_match.end():].strip()
          input_match = re.search(r'(?:\*\*|\b)Input(?:\*\*|\b)?\s*:\s*', after_action, re.IGNORECASE)
          if input_match:
              rest = after_action[input_match.end():].strip()
              query_lines = []
              for line in rest.splitlines():
                  if re.match(r'(?:\*\*|\b)?(Thought|Action|Observation)(?:\*\*|\b)?\s*:', line.strip(), re.IGNORECASE):
                      break
                  query_lines.append(line)
              inp = " ".join(query_lines).strip()
          else:
              inp = after_action.splitlines()[0].strip() if after_action else ""
      else:
          # Direct answer without Action: tag
          clean_ans = reply
          for prefix in ["**Answer:**", "Answer:", "**Final Answer:**", "Final Answer:"]:
              if clean_ans.startswith(prefix):
                  clean_ans = clean_ans[len(prefix):].strip()
                  break
          action = "ANSWER"
          inp = clean_ans

  safe_print(f"[AGENT] Action: {action} | Tool input: {inp[:100]}", flush=True)

  return {
      'messages' : [AIMessage(content=reply)],
      'action' : action,
      'tool_input' : inp,
      'final_answer' : inp if action == "ANSWER" else ""
  }


def tool_node(state: AgentState):
  action = state.get("action", "").strip()
  query = state.get("tool_input", "").strip()

  safe_print(f"[TOOL] Executing {action} with query: {query}", flush=True)

  # Dispatch to the right tool
  if action in TOOLS:
      observation = TOOLS[action](query)
  else:
      observation = f"Unknown tool: {action}"

  safe_print(f"[TOOL] {action} returned {len(observation)} chars", flush=True)

  return {
      "messages": [HumanMessage(content=f"Observation: {observation[:2000]}")]
  }

def should_continue(state: AgentState):
    if state.get("action") == "ANSWER" or state.get("final_answer"):
        return "end"
    return "tool"


graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)
graph.add_node("tool", tool_node)

graph.set_entry_point("agent")

graph.add_conditional_edges(
    "agent",
    should_continue,
    {"end": END, "tool": "tool"}
)

graph.add_edge("tool", "agent")

app = graph.compile()

def run_langgraph_agent(question: str):
    safe_print(f"\n[AGENT] Starting processing question: '{question}'", flush=True)
    result = app.invoke({
        "question": question,
        "messages": [HumanMessage(content=question)],
        "final_answer": "",
        "action": "",
        "tool_input": ""
    })
    final = result.get("final_answer", "")
    safe_print(f"[AGENT] Completed. Final answer length: {len(final)} chars\n", flush=True)
    return final