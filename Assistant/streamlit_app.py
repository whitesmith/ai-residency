__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import sqlite3

from langchain.tools import BraveSearch, tool
from langchain.utilities import WikipediaAPIWrapper
from langchain.agents import load_tools, initialize_agent, Tool,LLMSingleActionAgent, AgentOutputParser, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import  BaseChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage, AgentAction, AgentFinish
from langchain.chat_models import AzureChatOpenAI, ChatAnthropic
from langchain.evaluation.loading import load_evaluator
from langchain.document_loaders import RecursiveUrlLoader, UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import os
import re
from typing import List, Union
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings

from streamlit_chat import message
import streamlit as st
import streamlit_authenticator as stauth
from time import time



load_dotenv()


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

model2 = ChatAnthropic(model = "claude-2", temperature = 0.2, max_tokens_to_sample = 10000)

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = BASE_URL
os.environ["OPENAI_API_KEY"] = API_KEY
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
embeddings = OpenAIEmbeddings(deployment="summersmith-2023-embeddings")



search = BraveSearch.from_api_key(api_key=os.environ["BRAVE_API_KEY"], search_kwargs={"count": 3})

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Useful for when you need to search about Current Events and World's current State or aren't familiarized with something. After using this tool you must write relevant findings in the Notes section."
        "You shall use this tool to confirm any information about which you are uncertain."
        "It takes as input a string, that is a query for search. It is called as follows:'''\n"
        "Action: Search\n"
        "Action Input: <Insert Query Here>\n"
        "Observation: <Results here>\n'''"
        "Notes: <Add any relevant information from the results to your notes>",
    )
]

tools += [
    Tool(
    name = "Wikipedia",
    func = WikipediaAPIWrapper().run,
    description = "This is a compendium of usefull historic, technical and trivia knowledge. You shall use this tool to learn about such topics in the following manner:\n"
    "Action: Wikipedia\n"
    "Action Input: <Insert topic here>")
]

def search_URL(url: str, query:str):
    loader = UnstructuredURLLoader([url])
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap  = 0
    )
    docs_split = text_splitter.split_documents(loader.load() )
    if len(docs_split) == 0:
        return "Can't extract information from this URL."
    db = Chroma.from_documents(docs_split[:3],embeddings)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm = model1, retriever = retriever,chain_type = "stuff")
    query = "Please provide a complete answer with examples, if applicable, to the following:\n" + query
    answer = qa.run(query)
    
    return answer

@tool("Search_Website")
def parsed_search_URL(url_query: str) -> str:
    """
    When you use the Search tool you may encounter some useful links that may contain useful extra information about a given website. 
    If you think some website may contain more valuable information than previously available use this tool to "ask questions" to these websites/links. It takes a string as an input (a question to the website)
    and returns a response from an AI that has read the website. The input is given in the following format, respecting the spacing exactly:
    '''
    URL: <url here>\nQUERY: <query here>
    '''
    Be sure to only use this tool one time per pair (website,query). 
    
    The query talks to an assistant that knows the website very well and therefore needs to be a well structured sentence or question!
    """
    ## STRING SHOULD BE FORMATTED AS "URL: <url here>\nQUERY: <query here>"
    url_ind = url_query.find("URL: ") + 5
    query_ind = url_query.find("\nQUERY:")
    URL = url_query[url_ind:query_ind]
    QUERY = url_query[query_ind+7:]
    return search_URL(URL,QUERY)
    
tools += [parsed_search_URL]



prompt_sistema = """You are an helpful and skilfull agent (referred to as Assistant) designed to answer the user's tasks to the best of your hability. When you don't know something you try to find out. Your skills include but are not limited to: coding, summarizing, writing, conversation. 
You are provided with a set of usefull tools (each has an input) that may help you in your tasks. Your purpose is to provide a complete Final Answer. Here are the tools:\n
{tools}
Use exactly the following format, and keep track of any relevant information using the Notes section, below the Observation section, as explained below:

'''
Question: the input question you must answer

Thought: you should always think about what to do

Pre-Act: given the information until now, what tools would a normal human being use? No tools may be an acceptable answer. If no tool is needed be sure to use the "Talk" tool!

Action: the action/tool to take, should only be one of these [{tool_names}]

Action Input: the input to the tool chosen in the "Action" step above

Observation: the result of the action/tool

Notes: Relevant Information to keep track of. This list will grow as your thought process grows. 
...
(this format Thought/Pre-Act/Action/Action Input/Observation/Notes can repeat N times)
...

Thought: I now know the final answer.
Final Answer:  In your final answer you should do your best to summarize the content Search, Search_Website and Wikipedia tools explicitly, if needed. You can also include anything you deduce from your tools and is not explicitly in the output. The more complete your answer is the better. 
'''

The Final Answer is the most important piece of this structure! It is the only part of your thought process that the user sees! It should include examples when applicable.
Consider the following example and notice how it explains the code and includes an examples in the final answer:
'''
User: How can I measure the length of a Python list?
AI:
Question: 'How can I measure the length of a Python list?'
Pre-Act: A normal person would search forums such as StackOverflow for programming related questions!
Action: Search
Action Input: how to measure length python list stackoverflow
Observation: <results of the search tool here>
Notes: My research tells me that the len function in Python can measure the length of Python lists.
Thought: I now know the final answer.
Final Answer: You can use the len function in Python to get the length of a python list. Here's an example:

```python
l = [1,2,3] #Define the list
print(len(l)) # > 3
```

'''

You should always strive to include clear examples in your answers. Coding related questions, in particular, benefit a lot from concrete examples.
If you feel you are stuck in a loop try another approach!

Here is a context of what has happened in the conversation so far:

{chat_history}

Be sure to use the above structure Thought / Pre-Act / Action / Action Input / Observation / Notes. Everything you write should be put in one of these sections! Final rule: Don't give up easily!
Begin!

Question: {input}
{agent_scratchpad}

"""


# Set up a prompt template
class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    
    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]
    

prompt = CustomPromptTemplate(
    template=prompt_sistema,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps", "chat_history"] )
@tool
def print_parse_erro_handler(output):
    """
    When you return a response in the wrong format this function will be called.
    """

    return (f"There was a parsing error, with the output\n{output}\nBe sure that the text belongs to one of the sections in the proposed structure!\n. The parsing error function has a message for me:\n\n You should only state the following, respecting the structure:\n"+
        " '''Here is the starting question:\"<rewrite the original starting question>\". Do I know waht to answer, given the above information, and don't need to keep on thinking?<Single word answer yes/no>'''."+
        "If the answer is yes then state exactly the following: '''\nFinal Answer: <Write the full final answer in a self contained way as well as any additional information relevant to the topic>'''."+
        "If the answer is no I then say the following '''\nNotes:<The notes you've been retaining>\nThought:<think about what to do next>''', and maintaing the original \"Thought/Pre-Act/Action/Action Input/Observation/Notes\" structure until you reach the Final Answer.\n\n"
        "I shall now do what the error message recommends.\n")

tools += [print_parse_erro_handler]

@tool
def Talk(text):
    """
    ONLY USE THIS WHEN USER QUERY REQUIRES NO TASKS.
    Every time you want to answer the Human without resorting to the other tools you need to use this one. This tool is only used in conversation responses or for accepting the user's information or when you have concluded
    that you won't need to perform tasks. It takes as input the response you want to give the human and includes it in the conversation as your response. 

    Example of tool usage:<
    User: Hello!

    Action: Talk

    Action Input: Hello! How may I assist you?

    Observation:  The Talk function has been called! In order to respect the structure, I have to rewrite the following, without the quotes:
        '''Notes:No need to retain relevant information.
            Thought: I now know the final answer!
            Final Answer: Hello! How may I assist you?'''
             I will now follow the above template:
    
    Notes: No need to retain relevant information.
    Thought: I now know the final answer!
    Final Answer: Hello! How may I assist you?

    >
    """
    return "The Talk function has been called! In order to answer the user's qustion and respect the structure, I have to rewrite the following, without the quotes:\n\t'''Notes:No need to retain relevant information.\n\tThought: I now know the final answer!\n\tFinal Answer: "+text+"'''\n\t I will rewrite the above text verbatim now:\n" 

tools += [Talk]

class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            #raise ValueError(f"Could not parse LLM output: `{llm_output}`")
            return AgentAction(tool="print_parse_erro_handler",tool_input=llm_output,log=llm_output)
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
    

output_parser = CustomOutputParser()



st.session_state.messages = st.session_state.get("messages",[])

msgs = StreamlitChatMessageHistory(key="messages")

st.session_state["messages"] = st.session_state.get("messages",[])


llm_chain = LLMChain(llm=model1, prompt=prompt)
memory = ConversationBufferMemory(memory_key="chat_history" , chat_memory=msgs , return_messages = True)


def PerfectPrompt(question: str = "",  model = model1, memory = memory):
    

    roles = {1:"AI: ",0:"Human: "}
    chat_history = memory.load_memory_variables({})['chat_history']

    history = ""

    for i,interaction in enumerate(chat_history):
        history += roles[i%2]+interaction.content+"\n"
    
    print("hello",history)
    
    
    prompt = f""" 
    You are a perfectionist and knowledgeable assistant that will always verify their answers are clear and have a nice structured and readable format.

    Here's the conversation so far:

    {history}

    Rewrite the last AI message in a way that is more complete and more organised and better suited to answer the Human. Prioritise bullet points and examples. 
    It is also important to abide by any formatting instructions that the Human asks! Please refrain from changing the intention or information of the original message!

    Present the text in the following format:

    'AI: <Improved AI response here>'

    Begin! 

    """
    return model.predict(prompt)[3:]

tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain, 
    output_parser=output_parser,
    stop=["\nObservation:"], 
    allowed_tools=tool_names
)
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory,
    handle_parsing_errors= True#"There was a parsing error, I should only state the following, respecting the structure:\n"
    # " '''Here is the starting question:\"<rewrite the original starting question>\". Do I know the answer given the above information, or should I keep on thinking?<Single word answer yes/no>'''."
    # "If the answer is yes then state exactly the following: '''\nFinal Answer: <Write the full final answer in a self contained way as well as any additional information relevant to the topic>'''."
    # "If the answer is no I then say the following '''\nThought:<think about what to do next>''', and maintaing the original structure until you reach the Final Answer. I will remember to avoid doing the same task twice.",
)


def main():
    # query = '0'
    # while True:
    #     query = input(">")
    #     if query == 'fim':
    #         break
    #     print(agent_executor.run(query))
    # st.set_page_config(
    #     page_title="An Initial Attempt of a Chat",
    #     page_icon="🤖"
    # )
    st.header("Your Assistant")


    for i,msg in enumerate(msgs.messages):
        message(msg.content, is_user=(msg.type!="ai"),key = i)
        print(msg.type)

    with st.sidebar:
        user_input = st.text_input("Your message: ", key="user_input")

            # handle user input
    st.session_state["time_prompt"] = time() - st.session_state.get("prompt_instant",0)
    if (st.sidebar.button("Prompt!") and st.session_state.time_prompt > 4):
        print("O prompt foi aceite com",st.session_state.time_prompt,"segundos.")
        print("")
        st.session_state["prompt_instant"] = time()
        message(user_input, is_user =True, key = len(msgs.messages) )

        with st.spinner("Thinking..."):
            response = agent_executor.run(user_input)
            response = PerfectPrompt()
            msgs.messages[-1] = AIMessage(content = response)
            print(response)
        message(response, is_user=False, key = len(msgs.messages)-1)
    
    
    return 0







if __name__ == '__main__':

    names = ['Name']
    usernames = ['Username']
    passwords = ['Password']
    
    st.set_page_config(
            page_title="An Initial Attempt of a Chat",
            page_icon="🤖"
        )
    
    hashed_passwords = stauth.Hasher(passwords).generate()
        
    authenticator = stauth.Authenticate({ "usernames" : {usernames[i]:{"name":names[i], "password":hashed_passwords[i]} for i in range(len(usernames))} },
        'some_cookie_name','some_signature_key',cookie_expiry_days=30)
    
    
    name, authentication_status, username = authenticator.login('Login', 'main')
    
    
    
    # Use the hasher module from the package to convert the plain text passwords to hashed passwords
    
    # Create a login widget using the stauth.login function
    
    
    # Implement user privileges using the stauth.authorize function to check if the authenticated user has the necessary privileges to access certain parts of the app
    if authentication_status:
        print(authentication_status,name, username)
        authenticator.logout("Logout", "sidebar")
        main()# print(name,authentication_status,username)# Allow access to admin-only parts of the app
    elif authentication_status == False:
        st.error('Username/password is incorrect')
        # Deny access to admin-only parts of the app
    elif authentication_status == None:
        st.warning("Enter the usermame and password")
        st.session_state.messages = []
