from ModelGraph import AgentGraph
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import streamlit as st

def test_graph_call():
    graph = AgentGraph(st=st, model_name="TestGPT")
    # graph.call("what is 8 * 10, 9^5, and 3195 + 98? What's the sum of all 3 solutions?")
    # graph.call("Can you send an email to my professor telling him I will be late to class tomorrow?")
    # graph.call("put on my calendar to meet with my professor tomorrow at 10 AM")
    # graph.call("hello!")
    
    # graph.call("What is the current valuation of NVidia?")

test_graph_call()