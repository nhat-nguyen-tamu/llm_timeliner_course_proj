from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode
import streamlit as st
import functools
import datetime
from duckduckgo_search import DDGS
import wikipedia
import arxiv

# from GoogleAPIHelper import GoogleAPIHelper
# google_api = GoogleAPIHelper()

class Tools:
    def __init__(self, st=None, assistant=None, tool_set="questioner"):
        self.st = st
        self.tools = self.get_tools(tool_set)
        if self.tools:
            # caches reduce the chances of a rate limit error
            self.ddg_cache = {}
            self.wiki_shallow_cache = {}
            self.wiki_deep_cache = {}
            self.arxiv_cache = {}

            self.tools_fallback = self.create_tool_node_with_fallback(self.tools)
            self.assistant = assistant.bind_tools(self.tools)
        else:
            self.assistant = assistant

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

    def get_tools(self, tool_set="questioner"):
        def action_request(func): # ideally some tools should ask us for confirmation before submitting but I'm out of time to code this
            @functools.wraps(func) # we need this to preserve the docstrings for each tool when we wrap it
            def wrapper(*args, **kwargs):
                # Generate the confirmation message
                params = ', '.join([str(arg) for arg in args] + [f"{k}={v}" for k, v in kwargs.items()])
                confirmation_message = f"Do you want to execute '{func.__name__}' with parameters: {params}? Reply 'yes' or 'no'."
                
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
        def ask_question(questions: list[str]) -> str:
            '''Write questions into the state. These questions will be used by the researcher to follow up with search results. Checks for duplicate and similar questions.'''
            
            if not self.st.session_state.researching:
                return "Cannot ask any more questions - research phase has ended"

            if not isinstance(questions, list):
                return "Error: questions must be provided as a list of strings"

            error_messages = []
            success_count = 0
            
            for question in questions:
                if not question or not isinstance(question, str):
                    error_messages.append(f"Skipped invalid question: {question}")
                    continue
                    
                # Clean and normalize the question
                cleaned_question = " ".join(question.strip().split())
                
                # Check if question is already in questions list
                if cleaned_question in self.st.session_state.questions:
                    error_messages.append(f"Question already exists: '{cleaned_question}'")
                    continue
                    
                # Check if question is a substring of or contains any answered questions
                substring_match = False
                for answered in self.st.session_state.answered_questions:
                    if cleaned_question.lower() in answered.lower() or answered.lower() in cleaned_question.lower():
                        error_messages.append(f"Question '{cleaned_question}' is too similar to previously answered question: '{answered}'")
                        substring_match = True
                        break
                        
                if substring_match:
                    continue
                    
                # Add the question if it passes all checks
                self.st.session_state.questions.append(cleaned_question)
                success_count += 1
            
            # Construct response message
            response_parts = []
            if success_count > 0:
                response_parts.append(f"Successfully added {success_count} question{'s' if success_count != 1 else ''}")
            if error_messages:
                response_parts.append("\nWarnings:\n- " + "\n- ".join(error_messages))
                
            return " ".join(response_parts) if response_parts else "No valid questions were provided"

        @tool
        def duck_duck_go(search_term: str) -> str:
            '''Search online. Useful for initial search or informal searching. Do not search private information online. Information may be incorrect.'''
            private_information_blacklist = st.secrets["BLACKLIST_SEARCH_TERMS"]

            self.st.session_state.DDGS_calls += 1

            # Check cache first
            if search_term in self.ddg_cache:
                self.st.session_state.web_call_cache_hits += 1
                return self.ddg_cache[search_term]

            try:
                for term in private_information_blacklist:
                    if term in search_term.lower():
                        return f"There was an error executing the search: You cannot search private information online ({term})"

                result = DDGS().text(search_term, max_results=10)
                result_str = str(result)
                
                # Cache the result
                self.ddg_cache[search_term] = result_str
                
                return result_str
            except Exception as e:
                return f"There was an error executing the search: {str(e)}"
        
        @tool
        def arxiv_search(search_term: str) -> str:
            '''Search on arxiv. Returns abstracts of the top 5 most recent academic papers matching the search term.'''
            private_information_blacklist = st.secrets["BLACKLIST_SEARCH_TERMS"]

            self.st.session_state.arxiv_calls += 1

            # Check cache first
            if search_term in self.arxiv_cache:
                self.st.session_state.web_call_cache_hits += 1
                return self.arxiv_cache[search_term]

            try:
                # Check for blacklisted terms
                for term in private_information_blacklist:
                    if term in search_term.lower():
                        return f"There was an error executing the search: You cannot search private information online ({term})"

                # Create client and search
                client = arxiv.Client()
                search = arxiv.Search(
                    query=search_term,
                    max_results=5,
                    sort_by=arxiv.SortCriterion.SubmittedDate
                )

                # Fetch results
                results = []
                for result in client.results(search):
                    paper_info = {
                        'title': result.title,
                        'authors': ', '.join(author.name for author in result.authors),
                        'abstract': result.summary,
                        'url': result.entry_id,
                        'published': result.published.strftime('%Y-%m-%d')
                    }
                    results.append(paper_info)

                # Format results as a string
                if not results:
                    result_str = "No papers found matching the search term."
                else:
                    result_str = "Top 5 most recent papers:\n\n"
                    for i, paper in enumerate(results, 1):
                        result_str += f"{i}. Title: {paper['title']}\n"
                        result_str += f"   Authors: {paper['authors']}\n"
                        result_str += f"   Published: {paper['published']} (YYYY-MM-DD)\n"
                        result_str += f"   URL: {paper['url']}\n"
                        result_str += f"   Abstract: {paper['abstract']}\n\n"

                # Cache the result
                self.arxiv_cache[search_term] = result_str
                
                return result_str
            except Exception as e:
                return f"There was an error executing the search: {str(e)}"

        @tool
        def wikipedia_shallow(search_term: str) -> str:
            '''Shallow wikipedia search, only provides a summary. Useful for quick referencing and low token usage. For a deeper search, use wikipedia_deep.'''
            private_information_blacklist = st.secrets["BLACKLIST_SEARCH_TERMS"]

            self.st.session_state.wikipedia_shallow_calls += 1

            # Check cache first
            if search_term in self.wiki_shallow_cache:
                self.st.session_state.web_call_cache_hits += 1
                return self.wiki_shallow_cache[search_term]

            try:
                for term in private_information_blacklist:
                    if term in search_term.lower():
                        return f"There was an error executing the search: You cannot search private information online ({term})"

                result = wikipedia.summary(search_term)
                result_str = str(result)
                
                # Cache the result
                self.wiki_shallow_cache[search_term] = result_str
                
                return result_str
            except Exception as e:
                return f"There was an error executing the search: {str(e)}"

        @tool
        def wikipedia_deep(search_term: str) -> str:
            '''Provides the full wikipedia text on requested content. This is VERY expensive. It is recommended that wikipedia_shallow is called first.'''
            private_information_blacklist = st.secrets["BLACKLIST_SEARCH_TERMS"]

            self.st.session_state.wikipedia_deep_calls += 1

            # Check cache first
            if search_term in self.wiki_deep_cache:
                self.st.session_state.web_call_cache_hits += 1
                return self.wiki_deep_cache[search_term]

            try:
                for term in private_information_blacklist:
                    if term in search_term.lower():
                        return f"There was an error executing the search: You cannot search private information online ({term})"

                result = wikipedia.page(search_term).content
                result_str = str(result)
                
                # Cache the result
                self.wiki_deep_cache[search_term] = result_str
                
                return result_str
            except Exception as e:
                return f"There was an error executing the search: {str(e)}"
            
        @tool
        def log_timeline(day: int, month: int, year: int, title: str, description: str) -> str:
            '''Log information learned from online sources, this log will be used to build a timeline. Avoid appending duplicates.'''

        @tool
        def take_notes(notes: list[str]) -> str:
            '''Write notes into storage. These notes will be used to generate a timeline. Avoids duplicate notes and validates input.'''
            
            if not self.st.session_state.researching:
                return "Cannot take notes - research phase has ended"

            if not isinstance(notes, list):
                return "Error: notes must be provided as a list of strings"
                
            error_messages = []
            success_count = 0
            
            for note in notes:
                # Basic input validation
                if not note or not isinstance(note, str):
                    error_messages.append(f"Skipped invalid note entry: {note}")
                    continue
                    
                # Remove excess whitespace and normalize
                cleaned_note = " ".join(note.strip().split())
                
                # Check for duplicates (case insensitive)
                if any(existing_note.lower() == cleaned_note.lower() 
                    for existing_note in self.st.session_state.notes):
                    error_messages.append(f"Note already exists: '{cleaned_note}'")
                    continue
                    
                # Add the note if it passes all checks
                self.st.session_state.notes.append(cleaned_note)
                success_count += 1
            
            # Construct response message
            response_parts = []
            if success_count > 0:
                response_parts.append(f"Successfully added {success_count} note{'s' if success_count != 1 else ''}")
            if error_messages:
                response_parts.append("\nWarnings:\n- " + "\n- ".join(error_messages))
                
            return " ".join(response_parts) if response_parts else "No valid notes were provided"
    
        @tool
        def done() -> str:
            '''Once no more questions are needed, done() should be called to end research. Do not call this twice.'''

            self.st.session_state.researching = False
            if not self.st.session_state.researching:
                return "Cannot call this twice"
            
            return "Research has ended"

        if tool_set == "questioner":
            tools = [ask_question]
        elif tool_set == "researcher":
            tools = [duck_duck_go, take_notes, wikipedia_shallow, wikipedia_deep, arxiv_search]
        elif tool_set == "builder":
            tools = []

        return tools



