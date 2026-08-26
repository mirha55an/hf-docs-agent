from src.tools import TOOLS
from src.prompts import SYSTEM_PROMPT
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
import operator, os
from dotenv import load_dotenv
load_dotenv()

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
                print(f"LLM Error, retrying in 10s... ({e})")
                time.sleep(10)
            else:
                raise

def agent_node(state: AgentState):

  messages = [SystemMessage(content=SYSTEM_PROMPT)] + state['messages']

  response = call_llm(messages)
  content = response.content
  reply = extract_text(content)

  reply = reply.strip()

  lines = reply.split("\n")
  action_line = next((l for l in lines if l.startswith("Action:")), "")
  action = action_line.replace("Action:", "").strip()

  input_line_idx = next((i for i, l in enumerate(lines) if l.startswith("Input:")), None)

  if input_line_idx is not None:
    # First line of Input: plus all subsequent lines
    first_line = lines[input_line_idx].replace("Input:", "").strip()
    remaining = lines[input_line_idx + 1:]
    inp = "\n".join([first_line] + remaining).strip()
  else:
    inp = ""

  return {
      'messages' : [AIMessage(content=reply)],
      'action' : action,
      'tool_input' : inp,
      'final_answer' : inp if action == "ANSWER" else ""
  }


def tool_node(state: AgentState):
  content = state["messages"][-1].content
  last_message = extract_text(content)
  lines = last_message.split('\n')
  action_line = next((l for l in lines if l.startswith("Action:")), "")
  input_line = next((l for l in lines if l.startswith("Input:")), "Input: ")

  action = action_line.replace("Action:", "").strip()
  query = input_line.replace("Input:", "").strip()

  # Dispatch to the right tool
  if action in TOOLS:
      observation = TOOLS[action](query)
  else:
      observation = f"Unknown tool: {action}"

  return {
      "messages": [HumanMessage(content=f"Observation: {observation[:2000]}")]
  }

def should_continue(state: AgentState):
    last_content = state["messages"][-1].content
    last_message = extract_text(last_content)
    if "Action: ANSWER" in last_message:
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
    result = app.invoke({
        "question": question,
        "messages": [HumanMessage(content=question)],
        "final_answer": "",
        "action": "",
        "tool_input": ""
    })
    return result["final_answer"]