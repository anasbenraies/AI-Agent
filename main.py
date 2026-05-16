from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent , AgentExecutor
from tools import search_tool, wikipedia_tool , write_to_file_tool,translate_tool
load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",disable_streaming=True)
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# NEW: memory object — memory_key must match the placeholder name in the prompt
memory = ConversationBufferMemory(
    memory_key="chat_history",   # matches "{chat_history}" in prompt
    return_messages=True         # returns HumanMessage/AIMessage objects, not raw strings
)

tools=[search_tool, write_to_file_tool, wikipedia_tool, translate_tool] #  wikipedia_tool
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools, 
    prompt=prompt
)

agent_executor = AgentExecutor(agent=agent, tools=tools,verbose= True, memory=memory )


# NEW: conversation loop so memory actually accumulates across turns
print("Research Assistant ready. Type 'exit' to quit.\n")
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break

    raw_response = agent_executor.invoke({"query": query})

    try:
        structured_response = parser.parse(raw_response["output"])
        print(structured_response)
    except Exception as e:
        # fallback: print raw if parsing fails mid-conversation
        print("Raw response:", raw_response["output"])
        print("Parse error:", e)