from datetime import datetime
from typing import Annotated, Dict, Optional, TypedDict
from typing_extensions import TypedDict
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from Assistant import State
from Assistant import Assistant

def get_date():
    return f"{datetime.today().strftime('%m-%d-%Y')} (MM-DD-YY)"

def get_time():
    return f"{datetime.now().strftime('%I:%M %p')}"

def list_to_readable(items):
    return '\n'.join(f"{i + 1}. {item}" for i, item in enumerate(items))

def get_state_questioner(self):
    state = (
        "[Questioner] :\n\n"
        # f"Questions answered: {len(self.st.session_state.answered_questions)}. Notes: {len(self.st.session_state.notes)} "
        f"Questions:\n{list_to_readable(self.st.session_state.questions)}\n\n"
        f"Answered Questions:\n{list_to_readable(self.st.session_state.answered_questions)}\n\n"
        f"Notes:\n{list_to_readable(self.st.session_state.notes)}\n\n"
        # f"Previous Action and Thought: {list_to_readable(self.st.session_state.questions)}"
        
    )

    return HumanMessage(content=f"{state}")

def get_state_researcher(self):
    state = (
        "[Researcher] Self-Reflect:\n\n"
        f"Research Question:\n{self.st.session_state.questions[0]}\n\n"
        f"Answered Questions:\n{list_to_readable(self.st.session_state.answered_questions)}\n\n"
        f"Notes:\n{list_to_readable(self.st.session_state.notes)}\n\n"
        # f"Previous Action and Thought: {list_to_readable(self.st.session_state.questions)}"
    )

    return HumanMessage(content=f"{state}")

def get_state_builder(self):
    state = (
        "[Timeline Builder] Self-Reflect:\n\n"
        f"Answered Questions:\n{list_to_readable(self.st.session_state.answered_questions)}\n\n"
        f"Notes:\n{list_to_readable(self.st.session_state.notes)}\n\n"
    )

    return HumanMessage(content=f"{state}")

# This function is to jerry rig the inability for langgraph to reset its message state
# we maintain our own state and have full control of it
def copy_tool_output_over(source_list, target_list):
    """
    Copies the most recent tool responses from source_list to target_list.
    Handles duplicate tool calls by considering their position in the message chain.
    
    Args:
        source_list: List containing all messages including tool responses
        target_list: List containing messages that need corresponding tool responses
    """
    # Find all tool calls in the target list and track which ones already have responses
    pending_tool_calls = []
    existing_responses = {}  # Maps (tool_call_id, call_position) to response count
    
    # First pass: collect existing tool responses and their positions
    call_position = 0
    for msg in target_list:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tool_call in msg.tool_calls:
                key = (tool_call['id'], call_position)
                call_position += 1
                existing_responses[key] = 0
                
        if isinstance(msg, ToolMessage):
            # Find the corresponding call position
            for (call_id, pos), count in existing_responses.items():
                if call_id == msg.tool_call_id:
                    existing_responses[(call_id, pos)] = count + 1
                    break
    
    # Second pass: collect pending tool calls that need more responses
    call_position = 0
    for msg in target_list:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tool_call in msg.tool_calls:
                key = (tool_call['id'], call_position)
                # If this call has fewer responses than it should, add it to pending
                if existing_responses[key] < 1:  # Adjust if tool calls can have multiple responses
                    pending_tool_calls.append((tool_call, call_position))
                call_position += 1
    
    if not pending_tool_calls:
        return  # No new tool calls to process
    
    # Find matching tool responses in source list, starting from the end
    tool_responses = []
    source_responses_used = set()  # Track which responses we've already used
    
    for msg in reversed(source_list):
        if isinstance(msg, ToolMessage):
            # Generate a unique identifier for this response
            response_id = (msg.tool_call_id, msg.content, id(msg))
            
            if response_id not in source_responses_used:
                # Check if this tool response matches any pending tool call
                for (tool_call, pos) in pending_tool_calls[:]:  # Use slice to allow removal
                    if (tool_call['id'] == msg.tool_call_id and 
                        tool_call['name'] == msg.name):
                        tool_responses.append(msg)
                        source_responses_used.add(response_id)
                        pending_tool_calls.remove((tool_call, pos))
                        break
        
        if not pending_tool_calls:
            break  # Found all needed tool responses
    
    # Append only the new tool responses to target list in correct order
    target_list.extend(reversed(tool_responses))

class Questioner(Assistant):
    def __init__(self, st=None, stream_callback=None):
        self.name = "questioner"
        super().__init__(st, stream_callback, tool_set=self.name)

        self.system_prompt = (
            "You are a timeline researcher. "
            "Your will call ask_questions to produce concise questions appropriate to the user's prompt that can be researched online. "
            "If no more questions are needed, you will NOT call ask_questions, and will simply output 'done'. "
            "You may produce up to 3 questions. "
            "You MUST ask questions that forward the understanding of the user prompt (ex. try asking, what happened in year/month/date X?). "
            "You MUST produce diverse questions (ex, don't make multiple questions around the same date) unless a certain date is extremely important or queried by the user. "
            f"Today is {get_date()}. The time is {get_time()} "
        )

    def get_state(self):
        return get_state_questioner(self)

    def __call__(self, state: State, config: RunnableConfig):
        # tool results appear in the state here for some reason, we need it to show up in the local llm log
        copy_tool_output_over(state['messages'], self.st.session_state.llm_state['messages'])
        
        state = self.st.session_state.llm_state
        
        print("!!! questions", "\n\t --> " + "\n\t --> ".join(self.st.session_state.questions), "\n")
        print("!!! answered_questions", "\n\t --> " + "\n\t --> ".join(self.st.session_state.answered_questions), "\n")
        print("!!! notes", "\n\t --> " + "\n\t --> ".join(self.st.session_state.notes), "\n")

        return super().__call__(state, config)


class Researcher(Assistant):
    def __init__(self, st=None, stream_callback=None):
        self.name = "researcher"
        super().__init__(st, stream_callback, tool_set=self.name)

        self.system_prompt = (
            "You are a timeline researcher. "
            "You will be assigned a research question. "
            "Your job is to search online to answer your assigned question. You may search up to 3 times. "
            "After searching, you will log dates found into your notes to place it in persistent memory. "
            # "You MUST ONLY log extremely important dates, since note logging is expensive. "
            "Your log MUST follow this format -> (MM-DD-YYYY): <note>. "
            "Do NOT log identical events: two notes cannot talk about the same event. "
            "After logging notes, you will output a concise response under under 50 words. "
            "If you do not know the answer, write 'I could not find anything on this'. "
            f"Today is {get_date()}. The time is {get_time()}"
        )

    def get_state(self):
        return get_state_researcher(self)

    def __call__(self, state: State, config: RunnableConfig):
        # tool results appear in the state here for some reason, we need it to show up in the local llm log
        copy_tool_output_over(state['messages'], self.st.session_state.llm_state['messages'])

        state = self.st.session_state.llm_state
        
        return super().__call__(state, config)

class Builder(Assistant):
    def __init__(self, st=None, stream_callback=None):
        self.name = "builder"
        super().__init__(st, stream_callback, tool_set=self.name)

        self.system_prompt = (
            "You are a timeline builder. "
            "You will take the provided notes and questions/answers and builder an event timeline. "
            "You will build a list of events in chronological order. "
            "Your output MUST follow this format -> (MM-DD-YYYY): <event>. "
            "Your output will be displayed as markdown text. "
            "Use both the notes and questions and answers to build this timeline. "
        )

    def get_state(self):
        return get_state_builder(self)

    def __call__(self, state: State, config: RunnableConfig):
        # tool results appear in the state here for some reason, we need it to show up in the local llm log
        copy_tool_output_over(state['messages'], self.st.session_state.llm_state['messages'])

        state = self.st.session_state.llm_state
        
        return super().__call__(state, config)