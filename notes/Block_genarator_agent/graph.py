from langgraph.graph import StateGraph,START,END
from pydantic import BaseModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt,Command

from state import BlogState
from agents import get_llm,researcher_agent,writer_agent,editor_agent

MAX_REVISION =3

#### Nodes and Edges

##Defining nodes

def researcher_node(state:BlogState):
    """ Researcher Agent generates (or revises) the research outline """
    llm= get_llm
    research_data = researcher_agent(
        llm =llm,
        topic=state.topic,
        audience = state.audience,
        feedback = state.research_feedback
    )

    state.research= research_data
    state.research_feedback=""

    return state


def human_review_research_node(state:BlogState):
    """ pause and ask human to approve the research or send the feedback  """


def writer_node(state:BlogState):
    pass


def editor_node(state:BlogState):
    pass



