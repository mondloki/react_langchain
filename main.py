from dotenv import load_dotenv
from langchain.agents import tool
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain import hub
from langchain_core.tools.render import render_text_description
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.output_parsers.react_single_input import ReActSingleInputOutputParser


from langchain.agents import (
    create_react_agent,
    AgentExecutor, 
    )


load_dotenv()

@tool
def get_text_length(text: str) -> int:
    "Returns the length of the text by characters"
    text = text.strip("\n").strip()

    return len(text)


template = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:
"""
tools = [Tool(
            name="Calculate the length of text in characters",
            func=get_text_length,
            description="useful to calculate the length of text in characters",
        )]

# prompt = PromptTemplate(template=template).partial(
#     tools=render_text_description(tools=tools),
#     tool_names = ", ".join([tool.name for tool in tools])
# )

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
    # other params...
)

# agent = {"input": lambda x: x["input"]} | prompt | llm | ReActSingleInputOutputParser()

# res = agent.invoke(input={"input" : "What is the length of cat in characters?"},
#                    handle_parsing_errors=True)
react_prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
agent_executor = AgentExecutor(agent=agent, 
                                tools=tools, 
                                verbose=True, 
                                handle_parsing_errors=True,
                                return_intermediate_steps=False,
                                early_stopping_method="generate"
                                )

result = agent_executor.invoke(
    input={"input" : "What is the length of cat in characters?"}
)

print("*"*100)
print(result["output"])
print("*"*100)

