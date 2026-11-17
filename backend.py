from langgraph.graph import StateGraph, START,END
from  langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv()


#getting the list of free models available
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# for model in client.models.list().data:
#     print(model.id)




from langgraph.graph.message import add_messages
class ChatState(TypedDict):
  messages: Annotated[list[BaseMessage], add_messages]


llm = ChatGroq(model="openai/gpt-oss-20b")


def chat_node(state: ChatState):
  #take user query from state
  messages = state['messages']
  #send to llm
  response = llm.invoke(messages)
  #store response to the state
  return {'messages':[response]}


checkpointer = MemorySaver()

graph = StateGraph(ChatState)

#add nodes
graph.add_node('chat_node', chat_node)

#adding edges
graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

#compile graph
chatbot = graph.compile(checkpointer=checkpointer)




  