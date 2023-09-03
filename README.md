# ai-residency

The Part1_Initial_prompt_test.py is one of the first scripts I mention in the AI-Residency first blog post! It explores some basic capabilities of LangChain and how Prompting Architectures can be built!
\
The Evaluation Folder contains the remaining relevant files mentioned in the first part!
\
The Assistant folder contains the streamlit_app.py, the requirements.txt for it and an example.env! This python file contains all code related to building the AI assistant (GPT-3.5-turbo based) above the main function.
From the main function onwards, the code aims to incorparate the calling of the LLM assistant into the streamlit application. 

# Notes on usage

The streamlit app requires a .env file that contains all the relevant keys! You must also set a name, username and password in streamlit_app.py code in order to be able to log in!
\
To use the application you must enter the following in the terminal:

  streamlit run streamlit_app.py
