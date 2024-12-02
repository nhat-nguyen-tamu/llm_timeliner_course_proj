from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph.message import add_messages, AnyMessage
from langchain.agents import initialize_agent, Tool

from typing import Annotated, Dict, Optional, TypedDict
from typing_extensions import TypedDict

import streamlit as st

import datetime

from Tools import Tools

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages] = []

class Assistant:
    def __init__(self, st=None, stream_callback=None):
        self.stream_callback = stream_callback
        self.st = st
        self.system_prompt = (
            "You are a news researcher. "
            "Your job is to perform research to create a timeline of important points for a given topic. "
            "You MUST use tools and end every message with an emoji."
            f"Today is {datetime.datetime.today().strftime('%m-%d-%Y')} (MM-DD-YY)."
        )

        # llm = ChatOllama(model="llama3.2")

        # CONFIGURABLE BLOCK: you can disable the llm above and use chatgpt's llm here if you have the API key, it's much smarter and faster
        api_key = st.secrets["OPENAI_API_KEY"]
        llm = ChatOpenAI(model="gpt-4o", api_key=api_key)  # Replace with your API key

        self.tools = Tools(st=self.st, assistant=llm)
        self.runnable = self.tools.get_assistant()

    def convert_tool_messages(self, messages: list[AnyMessage]) -> list[AnyMessage]:
        """
        Converts all ToolMessage objects in the messages list to AIMessage objects.
        """

        converted_messages = []
        for msg in messages:
            if isinstance(msg, AIMessage):
                if msg.content: # remove any entries that use pure tool call with no content
                    ai_message = AIMessage(content=msg.content) # remove tool calls since it might break the LLM
                    converted_messages.append(ai_message)
                else:
                    action_text = "Tool calls:\n"
                    tool_text_list = []
                    for tool in msg.tool_calls:
                        if tool['name'] == "no_tool_call":
                            tool_text_list = []
                            msg.response_metadata['no_tool_used'] = True
                            break

                        tool_text_list.append(f"'name': {str(tool['name'])}, 'args': {str(tool['args'])}")

                    if tool_text_list:
                        action_text += '\n'.join(tool_text_list)

                        ai_message = AIMessage(content=action_text) # remove tool calls since it might break the LLM
                        converted_messages.append(ai_message)
            elif isinstance(msg, ToolMessage):
                # Extract relevant information from ToolMessage
                tool_name = msg.name
                tool_response = msg.content
                tool_status = msg.status

                # Create an AIMessage with the tool's response
                ai_message_content = f"Ran tool {tool_name}. Tool output:\nStatus: {tool_status}\nContent: {tool_response}"
                ai_message = AIMessage(content=ai_message_content)
                converted_messages.append(ai_message)
            else:
                # Retain the original message if it's not a ToolMessage
                converted_messages.append(msg)
        return converted_messages

    def __call__(self, state: State, config: RunnableConfig):
        # print("--->", "ASSISTANT CALL")
        while True:
            # print("--->", "START INVOKING", state['messages'])
            #state['messages'] = self.convert_tool_messages(state['messages'])

            invoke_input = state['messages'] # use this for well behaved models like chatgpt
            # invoke_input = self.convert_tool_messages(state['messages']) # we have to do this for ollama models because there is a bug where seeing ToolMessage will confuse it
            
            result = self.runnable.invoke(invoke_input)


            # print("--->", "INVOKING DONE")
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            # print("--->", "LLM RESPONSE:", result.content)
            # print("--->", "LLM TOOL CALLS:", result.tool_calls)

            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                # print("--->", "MODIFYING STATE", state)
                messages = state["messages"] + [HumanMessage(content="Provide a nonempty response.")]
                state = {**state, "messages": messages}
            else:
                # print("--->", "ELSE STATEMENT")
                break

        if self.stream_callback and result.content:
            self.stream_callback(result.content)

        # print("--->", "RETURNING RESULT", result)
        return {"messages": result}