pip install langgraph langsmith langchain_groq langchain_community
pip install arxiv wikipedia

#to use the state for the state graph
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

#work with tools -- importing tools other than LLM that needs to integrated 
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun

from langchain_groq import ChatGroq

## Arxiv and wikipedia tools
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=300)
arxiv_tool=ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=300)
wiki_tool=WikipediaQueryRun(api_wrapper=api_wrapper)

wiki_tool.invoke("who is Donald Trump")
arxiv_tool.invoke("what is ADHD")
tools=[wiki_tool]
#Define State Class
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

graph_builder= StateGraph(State)
# insert your personal groq_api_key here
groq_api_key=''
#Lama mode is called here
llm=ChatGroq(groq_api_key=groq_api_key,model_name="llama3-8b-8192")
llm_with_tools=llm.bind_tools(tools=tools)
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

#libraries required to add tools to the langgraph
from langgraph.prebuilt import ToolNode,tools_condition

tool_node=ToolNode(tools=tools)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools","chatbot")
graph_builder.add_edge("chatbot", END)

#compiling the graph
graph = graph_builder.compile()

from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

#Sample Run 
user_input="Hi There, who is Andrew Ng"
events=graph.stream(
{"messages": [("user", user_input)]},stream_mode="values")
for event in events:
    event["messages"][-1].pretty_print()
