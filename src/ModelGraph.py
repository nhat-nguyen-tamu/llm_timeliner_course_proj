import os
import asyncio
from typing import Annotated, Dict, Optional, TypedDict
from typing_extensions import TypedDict
import uuid

import streamlit as st

from langchain.agents import AgentExecutor
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages, AnyMessage
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda

from Assistant import Assistant
import Tools

import datetime

def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)

def should_continue(state: MessagesState):
    # print("should continue?", state)
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

class AgentGraph():
    def __init__(self, st=None, model_name="PersonalGPT", event_callback=None, stream_callback=None):
        self.model_name = model_name
        self.event_callback = event_callback
        self.stream_callback = stream_callback
        self.st = st
        self.build_state_graph()

    def build_state_graph(self):
        workflow = StateGraph(MessagesState)

        self.assistant = Assistant(st=self.st)
        workflow.add_node("assistant", self.assistant.__call__)
        workflow.add_node("tools", self.assistant.tools.tools_fallback)
        
        workflow.add_edge(START, "assistant")
        workflow.add_conditional_edges("assistant", should_continue, ["tools", END])
        workflow.add_edge("tools", "assistant")

        self.memory = MemorySaver()

        self.config = {
            "configurable": {
                "thread_id": str(uuid.uuid4()),
            }
        }

        self.graph = workflow.compile(checkpointer=self.memory)
        
        # self.graph = workflow.compile() # no memory for now

    def call(self, user_input):
        _printed = set()
   
        state = {"messages": [
            SystemMessage(content=self.assistant.system_prompt), 
            HumanMessage(content=user_input),
        ]}

        # print("--->", "CALL INIT", user_input)
        self.message_index = 0 # where are you on the list of messages
        self.aborted = False

        for event in self.graph.stream(
            state,
            stream_mode="values",
            config=self.config
        ):
            if self.aborted:
                break

            _print_event(event, _printed)
            self.handle_event(event)

    def handle_event(self, event):
        if not self.event_callback:
            return

        # Example: Extract messages from the event and send via callback
        messages = event.get("messages")
        if messages:
            while self.message_index < len(messages):
                msg = messages[self.message_index]
                if isinstance(msg, HumanMessage):
                    # User message
                    self.event_callback({"user": msg.content})
                elif isinstance(msg, AIMessage):
                    # Assistant message
                    if msg.content:
                        self.event_callback({"assistant": msg.content})

                    if msg.tool_calls:
                        self.event_callback({"tool_call": msg.tool_calls})
                    
                elif isinstance(msg, ToolMessage):
                    # Tool message
                    self.event_callback({"tool_response": f"---> {msg.name}: {msg.content}"})

                self.message_index += 1

    def abort(self):
        self.aborted = True