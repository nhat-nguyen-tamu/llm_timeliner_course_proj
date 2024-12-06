from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
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
    def __init__(self, st=None, stream_callback=None, tool_set="questioner"):
        self.stream_callback = stream_callback
        self.st = st

        # CAUTION: if you plan on enable ollama, you MUST enable the helper for it, search within the project 'convert_tool_messages'
        # llm = ChatOllama(model="llama3.2")

        # CONFIGURABLE BLOCK: you can disable whichever LLM if you have the API key

        # you can use 'gpt-4o' if you are rich lol
        api_key = st.secrets["OPENAI_API_KEY"]
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)  # Replace with your API key

        # you can use 'claude-3-5-sonnet-latest' if you are rich lol
        # api_key = st.secrets["ANTHROPIC_API_KEY"]
        # llm = ChatAnthropic(model="claude-3-5-haiku-latest", api_key=api_key)  # Replace with your API key

        self.llm = llm
        self.tools = Tools(st=self.st, assistant=llm, tool_set=tool_set)
        self.runnable = self.tools.get_assistant()

    def convert_messages_for_llm(self, messages, provider="anthropic"):
        """
        Converts message format between OpenAI and Anthropic styles.
        
        Args:
            messages: List of messages to convert
            provider: "anthropic" or "openai"
        """
        converted = []
        
        for msg in messages:
            if provider == "anthropic":
                if isinstance(msg, SystemMessage):
                    converted.append(msg)
                elif isinstance(msg, HumanMessage):
                    converted.append(msg)
                elif isinstance(msg, AIMessage):
                    # For Anthropic, combine tool calls and content into a single message
                    content = []
                    if msg.content:
                        content.append({"type": "text", "text": msg.content})
                    
                    if msg.tool_calls:
                        for tool in msg.tool_calls:
                            content.append({
                                "type": "tool_use",
                                "tool_use": {
                                    "recipient": tool["name"],
                                    "tool_name": tool["name"],
                                    "tool_args": tool["args"],
                                    "id": tool["id"]
                                }
                            })
                    
                    converted.append(AIMessage(content=content))
                
                elif isinstance(msg, ToolMessage):
                    # Convert tool messages to Anthropic's tool_result block format
                    converted.append(AIMessage(content=[{
                        "type": "tool_result",
                        "tool_result": {
                            "tool_name": msg.name,
                            "tool_call_id": msg.tool_call_id,
                            "result": msg.content
                        }
                    }]))
            
            else:  # OpenAI format
                if isinstance(msg, (SystemMessage, HumanMessage)):
                    converted.append(msg)
                elif isinstance(msg, AIMessage):
                    # For OpenAI, separate tool calls and content
                    if isinstance(msg.content, list):  # Convert from Anthropic format
                        content = ""
                        tool_calls = []
                        
                        for block in msg.content:
                            if block["type"] == "text":
                                content = block["text"]
                            elif block["type"] == "tool_use":
                                tool_use = block["tool_use"]
                                tool_calls.append({
                                    "id": tool_use["id"],
                                    "name": tool_use["tool_name"],
                                    "args": tool_use["tool_args"]
                                })
                        
                        converted.append(AIMessage(content=content, tool_calls=tool_calls))
                    else:
                        converted.append(msg)
                
                elif isinstance(msg, ToolMessage):
                    converted.append(msg)
    
        return converted

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
    
    def clean_messages(self, state):
        # Clean messages before sending to API
        cleaned_messages = []
        for msg in state['messages']:
            if isinstance(msg.content, list):
                continue # anthropic produces a list intead of a string for msg.content

            if isinstance(msg, AIMessage):
                # Trim trailing whitespace from content
                content = msg.content.rstrip() if msg.content else ''
                cleaned_msg = AIMessage(
                    content=content,
                    tool_calls=msg.tool_calls if hasattr(msg, 'tool_calls') else ''
                )
                cleaned_messages.append(cleaned_msg)
            elif isinstance(msg, HumanMessage):
                cleaned_messages.append(HumanMessage(content=msg.content.rstrip()))
            elif isinstance(msg, SystemMessage):
                cleaned_messages.append(SystemMessage(content=msg.content.rstrip()))
            elif isinstance(msg, ToolMessage):
                cleaned_msg = ToolMessage(
                    content=msg.content.rstrip(),
                    tool_call_id=msg.tool_call_id,
                    name=msg.name,
                    status=msg.status
                )
                cleaned_messages.append(cleaned_msg)
            else:
                cleaned_messages.append(msg)
        
        state['messages'] = cleaned_messages

    def __call__(self, state: State, config: RunnableConfig):
        state['messages'] = self.st.session_state.llm_state['messages']
        state = self.st.session_state.llm_state

        self.clean_messages(state)

        # print("--->", "ASSISTANT CALL")
        while True:
            # print("--->", "START INVOKING", state['messages'])
            #state['messages'] = self.convert_tool_messages(state['messages'])

            # anthropic has a special structure that's unique to itself, we need a converter
            # provider = "anthropic" if isinstance(self.llm, ChatAnthropic) else "openai"
            # invoke_input = self.convert_messages_for_llm(state['messages'], provider)

            invoke_input = state['messages'] # use this for well behaved models like chatgpt
            # invoke_input = self.convert_tool_messages(state['messages']) # we have to do this for ollama models because there is a bug where seeing ToolMessage will confuse it

            result = self.runnable.invoke(invoke_input)

            # print("--->", "INVOKING DONE")
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            # print("--->", "LLM RESPONSE:", result.content)
            # print("--->", "LLM TOOL CALLS:", result.tool_calls)

            try:
                self.st.session_state.input_tokens += result.usage_metadata['input_tokens']
                self.st.session_state.output_tokens += result.usage_metadata['output_tokens']
            except Exception as e:
                print(f"An error occurred: {str(e)}")
            
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

        state['messages'] += [result]
        # print("--->", "RETURNING RESULT", result)
        return self.st.session_state.llm_state