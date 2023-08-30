from langchain.evaluation.loading import load_evaluator
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import (AIMessage,
  HumanMessage,
  SystemMessage,
  BaseMessage)
from langchain.prompts.chat import (
  SystemMessagePromptTemplate,
  HumanMessagePromptTemplate,
  )
import os

from dotenv import load_dotenv

load_dotenv()


## MODEL
BASE_URL = os.environ["BASE_URL"]
DEPLOYMENT_NAME = os.environ["DEPLOYMENT_NAME"]
API_KEY = os.environ["API_KEY"]

model1 = AzureChatOpenAI(
openai_api_base=BASE_URL,
openai_api_version="2023-05-15",
deployment_name=DEPLOYMENT_NAME,
openai_api_key=API_KEY,
openai_api_type="azure",
temperature = 0)
###


examples = []
question = ""
answer = ""
Q = False
with open("examples.txt", "r") as f:
 for line in f:
   if line.lower() == "question\n":
       Q = True
       if question != "":
           examples.append({"query":question,"answer":anwer})
       question = ""
       answer = ""
       continue
   if line.lower() == "answer\n":
       Q = False
       continue
   if Q:
       question += line
   else:
       answer += line
 f.close()
examples.append({"query":question,"answer":answer})
for dictionary in examples:
 dictionary["context"] = "You are a Large Language Model that can use a database and evaluate this question in Correctness and compare it with the Reference Answer. The answer can be CORRECT, PARTIALLY CORRECT and INCORRECT, when compared with the reference."


def results_model(model,examples): # FUNCTION TO RETAIN THE ANSWERS OF A GIVEN MODEL TO THE QUESTIONS
  results = []
  for example in examples:
   results.append({"result":model( messages = [HumanMessage(content = example["query"]) ] ).content})
  return results

evaluator_context = load_evaluator("context_qa",llm = model1)
evaluator = load_evaluator("qa",llm = model1)  ## GPT-3.5-turbo evaluators


results = resuls_model(model1,examples)
print(  evaluator_context.evaluate(examples,results)  )
## WILL OUTPUT CHATGPT EVALUATION OF ITSELF


