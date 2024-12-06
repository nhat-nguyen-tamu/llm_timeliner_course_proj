from ModelGraph import AgentGraph
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import streamlit as st

def test_graph_call():
    graph = AgentGraph(st=st, model_name="TestGPT")
    # graph.call("what is 8 * 10, 9^5, and 3195 + 98? What's the sum of all 3 solutions?")
    # graph.call("Can you send an email to my professor telling him I will be late to class tomorrow?")
    # graph.call("put on my calendar to meet with my professor tomorrow at 10 AM")
    # graph.call("hello!")
    
    # graph.call("What major events happened in the US between 2000-2010?")
    # graph.call("What is the current valuation of N-Vidia?")
    # graph.call("What are the current news of Tesla from October to now?")
    # graph.call("What's going on in south korea in the last 2 weeks?")
    graph.call("What is the timeline of Mamba LLM?")

test_graph_call()