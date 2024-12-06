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

from DotDict import DotDict
from Agents import Researcher, Questioner, Builder
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

class ST_Proxy():
    # we make this to get rid of the
    # "Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode."
    # error

    def __init__(self):
        self.reset_state()

    def reset_state(self):
        self.session_state = DotDict(self.new_state())
        self.secrets = DotDict({
            'OPENAI_API_KEY': st.secrets["OPENAI_API_KEY"],
            'ANTHROPIC_API_KEY': st.secrets["ANTHROPIC_API_KEY"],
        })

    def new_state(self):
        return {
            'questions': [],
            'answered_questions': [],
            'notes': [],
            'llm_state': {"messages": []},

            # this is always true until the LLM sets it to false, which shuts off the research loop
            'researching': True,

            'input_tokens': 0,
            'output_tokens': 0,
            
            'wikipedia_deep_calls': 0,
            'wikipedia_shallow_calls': 0,
            'DDGS_calls': 0,
            'arxiv_calls': 0,

            'call_failures': 0,

            'web_call_cache_hits': 0,
        }

class AgentGraph():
    def __init__(self, st=None, model_name="PersonalGPT", event_callback=None, stream_callback=None, max_questions=5, max_notes=5, recursion_depth=100):
        self.model_name = model_name
        self.event_callback = event_callback
        self.stream_callback = stream_callback
        self.real_st = st
        self.st = ST_Proxy()

        self.max_questions = max_questions
        self.max_notes = max_notes
        self.recursion_depth = recursion_depth

        self.reset_state()
        self.build_state_graph()

    def reset_state(self):
        self.st.reset_state()
        self.prompt = "<System> There is no prompt. Please alert the user."

    def build_state_graph(self):
        workflow = StateGraph(MessagesState)

        # agents
        self.questioner = Questioner(st=self.st)
        self.researcher = Researcher(st=self.st)
        self.builder = Builder(st=self.st)  # Final output processor

        # nodes
        workflow.add_node("questioner", self.questioner.__call__)
        workflow.add_node("researcher", self.researcher.__call__)
        workflow.add_node("builder", self.builder.__call__)
        
        workflow.add_node("questioner_tools", self.questioner.tools.tools_fallback)
        workflow.add_node("researcher_tools", self.researcher.tools.tools_fallback)

        # this node doesn't do anything by itself
        # it is used to build conditional connections that handle the flow of the graph
        workflow.add_node("dequeuer", lambda state: state) 

        # edges
        workflow.add_edge(START, "questioner")
        
        workflow.add_conditional_edges(
            "questioner",
            lambda state: self.should_continue_questioner(state=state),
            ["questioner_tools", "builder", "questioner"]
        )

        workflow.add_conditional_edges(
            "questioner_tools",
            lambda state: self.should_continue_questioner_tools(state=state),
            ["dequeuer", "questioner"]
        )
        
        workflow.add_conditional_edges(
            "dequeuer",
            lambda state: self.should_continue_dequeuer(state=state),
            ["researcher", "questioner"]
        )
        
        workflow.add_conditional_edges(
            "researcher",
            lambda state: self.should_continue_researcher(state=state),
            ["researcher_tools", "dequeuer"]
        )
        workflow.add_edge("researcher_tools", "researcher")
        
        workflow.add_edge("builder", END)

        # self.memory = MemorySaver()

        self.config = {
            "configurable": {
                "thread_id": str(uuid.uuid4()),
            },
            "recursion_limit": self.recursion_depth,
        }

        self.graph = workflow.compile()# checkpointer=self.memory)
        # self.graph = workflow.compile() # no memory

    def call(self, user_input):
        self.reset_state()
   
        _printed = set()

        self.prompt = user_input
        self.st.session_state.llm_state = {"messages": []}

        # print("--->", "CALL INIT", user_input)
        self.message_index = 0 # where are you on the list of messages
        self.aborted = False

        self.load_system_prompt(self.questioner)

        for event in self.graph.stream(
            self.st.session_state.llm_state,
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

    def reset_memory(self, state: MessagesState) -> MessagesState:
        """Reset the message history for questioner, keeping only system message"""
        # Get the system message
        system_message = next((msg for msg in state["messages"] if isinstance(msg, SystemMessage)), None)
        
        # Create new state with just system message
        new_state = state.copy()
        new_state["messages"] = [system_message] if system_message else []
        return new_state

    def load_system_prompt(self, agent):
        """Wipes memory and loads the system prompt for a given agent"""

        state = self.st.session_state.llm_state
        state = self.reset_memory(state)
        
        # Update messages in place
        state["messages"] = [
            SystemMessage(content=agent.system_prompt),
            HumanMessage(content=self.prompt),
            agent.get_state(),
        ]

        print("---------------->>> EDGE EVENT: system prompt reset", agent.name)

        self.st.session_state.llm_state = state

    def should_continue_dequeuer(self, state: MessagesState):
        """Determines if dequeuer should continue processing questions or return to questioner"""
        if not self.st.session_state.questions:  # No more questions in queue
            self.load_system_prompt(self.questioner)
            return "questioner"  # Always go back to questioner when done
        
        self.load_system_prompt(self.researcher)
        return "researcher" # Go to researcher if there is stuff in the question queue

    def should_continue_researcher(self, state: MessagesState):
        """Determines if researcher should continue processing questions or return to dequeuer"""
        if state["messages"][-1].tool_calls:
            return "researcher_tools"
        else:
            # if the LLM doesn't call any more tools, assume its response is the answer to the question
            answer = state["messages"][-1].content
            question = self.st.session_state.questions.pop(0)
            self.st.session_state.answered_questions.append(f"{question} -> {answer}")
            return "dequeuer"

    def should_continue_questioner(self, state: MessagesState):
        """Go to builder if LLM has no more questions, go to tools if LLM wants to add questions for research"""
        last_message = state["messages"][-1]
        
        if not last_message.tool_calls or len(self.st.session_state.answered_questions) >= self.max_questions or len(self.st.session_state.notes) >= self.max_notes:
            self.load_system_prompt(self.builder)
            return "builder"  # Continue if no tool was called
        else:
            print("---------------->>> EDGE EVENT: moving to tools")
            return "questioner_tools"  # Handle tool execution

    def should_continue_questioner_tools(self, state: MessagesState):
        """Go to dequeuer if LLM has questions, go back to questioner if there's no questions or issues"""
        last_message = state["messages"][-1]
        
        if self.st.session_state.questions: # make sure the LLM doesn't ask bad questions that get rejected
            return "dequeuer"  # Handle tool execution
        else:
            print("---------------->>> EDGE EVENT: questions failed, asking questioner again")
            return "questioner"  # Handle tool execution