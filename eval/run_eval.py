from tqdm import tqdm
import time, os
from src.agent import run_langgraph_agent
from src.retrieval import retrieve
from src.tools import TOOLS
from eval.eval_dataset import eval_questions
from eval.eval_dataset import ground_truths
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from datasets import Dataset
from dotenv import load_dotenv
load_dotenv()

agent_answers = []
retrieved_contexts = []

for question in tqdm(eval_questions):

  captured_contexts = []
  original_retrieve = retrieve

  def retrieve_and_capture(query):
    result = original_retrieve(query)
    captured_contexts.append(result)
    return result

  TOOLS['retrieve'] = retrieve_and_capture

  try:
    answer = run_langgraph_agent(question)
    agent_answers.append(answer if answer else "No answer generated")
    retrieved_contexts.append(captured_contexts if captured_contexts else ["No contexts retrieved"])

  except Exception as e:
    print(f"Failed on {question} - {e}")
    agent_answers.append("Error.")
    retrieved_contexts.append("Error.")

  TOOLS['retrieve'] = original_retrieve

  time.sleep(2)

print(f"Collect {len(agent_answers)} answers.")

ragas_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite', api_key=os.getenv("GEMINI_API_KEY")))

ragas_embeddings = LangchainEmbeddingsWrapper(GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-2', google_api_key=os.getenv("GEMINI_API_KEY")))

eval_dataset = Dataset.from_dict({
    "question": eval_questions,
    "ground_truth": ground_truths,
    "response": agent_answers,
    "retrieved_contexts": retrieved_contexts
})

results = evaluate(
    dataset=eval_dataset,
    llm=ragas_llm,
    metrics=[faithfulness, answer_relevancy, context_precision],
    embeddings=ragas_embeddings,
)

print(results)
results_df = results.to_pandas()
print(results_df[["user_input", "faithfulness", "answer_relevancy", "context_precision"]])
