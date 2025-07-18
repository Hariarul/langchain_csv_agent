from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_experimental.tools.python.tool import PythonREPLTool
import os
from langchain_experimental.agents.agent_toolkits import create_csv_agent


load_dotenv()

if __name__ == "__main__":
    print("Starting agent...")
    print("Working directory:", os.getcwd())

 
    instructions = """You are an agent designed to write and execute Python code to answer questions.
You have access to a Python REPL, which you can use to execute code.
If you get an error, debug and try again.
Only use the output of your code to answer.
Even if you know the answer, you should run code to verify it.
If the question can't be answered with code, respond with "I don't know".
"""

    
    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)

    
    tools = [PythonREPLTool()]

   
    agent = create_react_agent(
        prompt=prompt,
        llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
        tools=tools
    )

    
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    
    task_input = {
        "input": """generate and save in current working directory 15 QR codes
                    that point to www.udemy.com/course/langchain.
                    You have the `qrcode` package already installed."""
    }


    csv_agent = create_csv_agent(llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0), path="episode_info.csv", 
                                 verbose=True,
                                 allow_dangerous_code=True)
    
    csv_agent.invoke(
        input={"input":"tell me how many columns are in the dataset?"}
    )


    csv_agent.invoke(
        input={
            "input": "print the seasons by ascending order of the number of episodes they have"
        }
    )


    # Run the agent
    # result = agent_executor.invoke(task_input)
    # print("\nAgent finished. Result:")
    # print(result)
