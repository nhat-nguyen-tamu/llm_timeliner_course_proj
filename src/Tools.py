from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode
import streamlit as st
import functools
import datetime
from duckduckgo_search import DDGS

# from GoogleAPIHelper import GoogleAPIHelper
# google_api = GoogleAPIHelper()

class Tools:
    def __init__(self, st=None, assistant=None):
        self.st = st
        self.tools = self.get_tools()
        self.tools_fallback = self.create_tool_node_with_fallback(self.tools)
        self.assistant = assistant.bind_tools(self.tools)

    def get_assistant(self):
        return self.assistant
    
    def create_tool_node_with_fallback(self, tools) -> ToolNode:
        def handle_tool_error(state) -> dict:
            error = state.get("error")
            tool_calls = state["messages"][-1].tool_calls
            return {
                "messages": [
                    ToolMessage(
                        content=f"Error: {repr(error)}\n please fix your mistakes.",
                        tool_call_id=tc["id"],
                    )
                    for tc in tool_calls
                ]
            }
        
        return ToolNode(tools).with_fallbacks(
            [RunnableLambda(handle_tool_error)], exception_key="error"
        )

    def get_tools(self):
        def action_request(func): # ideally some tools should ask us for confirmation before submitting but I'm out of time to code this
            @functools.wraps(func) # we need this to preserve the docstrings for each tool when we wrap it
            def wrapper(*args, **kwargs):
                # Generate the confirmation message
                params = ', '.join([str(arg) for arg in args] + [f"{k}={v}" for k, v in kwargs.items()])
                confirmation_message = f"Do you want to execute '{func.__name__}' with parameters: {params}? Reply 'yes' or 'no'."
                
                print("WRAPPER SESSION STATE", st.session_state)
                st.session_state.confirmations.append({
                    'args': args,
                    'kwargs': kwargs,
                    'message': confirmation_message,
                    'function': func,
                })

                if st.session_state.get('user_confirmed', False):
                    return func(*args, **kwargs)
                else:
                    return "Tool successfully ran. User must approve of request."
            return wrapper

        @tool
        def duck_duck_go(search_term: str) -> str:
            '''Search online via. Do not search private information online. Information may be incorrect.'''
            private_information_blacklist = st.secrets["BLACKLIST_SEARCH_TERMS"]

            try:
                for term in private_information_blacklist:
                    if term in search_term.lower():
                        return f"There was an error executing the search: You cannot search private information online ({term})"


                result = DDGS().text(search_term, max_results=5)

                # print("DUCK DUCK GO RESULTS:", result)
                
                return str(result)
            except Exception as e:
                return f"There was an error executing the search: {str(e)}"

        tools = [duck_duck_go] # , no_tool_call]
        return tools



