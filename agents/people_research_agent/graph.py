from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
from typing import Any
from langchain_openai import ChatOpenAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from pydantic import BaseModel, Field
import asyncio
from typing import cast, Any, Literal
import json
from tavily import AsyncTavilyClient


from agents.people_research_agent.state import OverallState, InputState, OutputState
from agents.people_research_agent.configuration import Configuration
from agents.people_research_agent.utils import deduplicate_and_format_sources,format_all_notes

from agents.people_research_agent.prompts import QUERY_WRITER_PROMPT,INFO_PROMPT,EXTRACTION_PROMPT,REFLECTION_PROMPT

rate_limiter = InMemoryRateLimiter(
    requests_per_second=4,
    check_every_n_seconds=0.1,
    max_bucket_size=10,  # Controls the maximum burst size.
)

model = ChatOpenAI(model="gpt-4o-mini", rate_limiter=rate_limiter)


#Search 
tavily_async_client = AsyncTavilyClient()

class Queries(BaseModel):
    queries: list[str] = Field(
        description="List of search queries.",
    )
    
class ReflectionOutput(BaseModel):
    is_satisfactory: bool = Field(
        description="True if all required fields are well populated, False otherwise"
    )
    missing_fields: list[str] = Field(
        description="List of field names that are missing or incomplete"
    )
    
    search_queries: list[str] = Field(
        description="If is_satisfactory is false, provide 1-3 targeted search queries to help find the mission information"
    )
    reasoning: str = Field(description="Brief explanation of the assessment")

def generate_queries(state: OverallState, config: RunnableConfig) -> dict[str, Any]:
    """Generate search queries based on the user input and extraction schema"""
    
    #Gen configuration
    configuration = Configuration.from_runnable_config(config)
    max_search_queries = configuration.max_search_queries
    
    #Generate search queries
    structured_llm = model.with_structured_output(Queries)
    
    # Format system instructions
    person_str = f"Email: {state.person['email']}"
    if "name" in state.person:
        person_str += f" Name: {state.person['name']}"
    if "linkedin" in state.person:
        person_str += f" LinkedIn URL: {state.person['linkedin']}"
    if "role" in state.person:
        person_str += f" Role: {state.person['role']}"
    if "company" in state.person:
        person_str += f" Company: {state.person['company']}"
    
    query_instructions = QUERY_WRITER_PROMPT.format(
        person=person_str,
        info=json.dumps(state.extraction_schema, indent=2),
        user_notes=state.user_notes,
        max_search_queries=max_search_queries,
    )
    
    #Generate queries
    results = cast(
        Queries,
        structured_llm.invoke(
            [
                {"role": "system", "content": query_instructions},
                {
                    "role": "user",
                    "content": "",
                },
            ]
        )
    )
    
    #Queries
    query_list = [query for query in results.queries]
    
    return {"search_queries": query_list}


async def research_person(state: OverallState, config: RunnableConfig) -> dict[str, Any]:
    """Execute a multi-step web search and information extraction process.

    This function performs the following steps:
    1. Executes concurrent web searches using the Tavily API
    2. Deduplicates and formats the search results
    """
    
    #get configuration
    configuration = Configuration.from_runnable_config(config)
    max_search_results = configuration.max_search_results
    
    #Web search
    search_tasks = []
    for query in state.search_queries:
        search_tasks.append(
            tavily_async_client.search(
                query,
                days=360,
                max_results=max_search_results,
                include_raw_content=True,
                topic="general"
            )
        ) 
        
    #Execute all search concurrently
    search_docs = await asyncio.gather(*search_tasks)
    
    # Deduplicate and format sources
    source_str = deduplicate_and_format_sources(
        search_docs, max_tokens_per_source=1000, include_raw_content=True
    )
    
    #Generate structured notes relevant to the extraction schema
    system_instruction = INFO_PROMPT.format(
        info = json.dumps(state.extraction_schema, indent=2),
        content = source_str,
        people = state.person,
        user_notes = state.user_notes,
    )
    
    response = await model.ainvoke(system_instruction)
    
    return {"completed_notes": [str(response.content)]}



def gather_notes_extract_schema(state: OverallState) -> dict[str, Any]:
    """Gather notes from the web search and extract the schema fields."""
    
    #From all notes
    notes = format_all_notes(state.completed_notes)
    
    #Extract schema fields
    system_prompt = EXTRACTION_PROMPT.format(
        schema = json.dumps(state.extraction_schema, indent=2),
        notes = notes,
    )
    
    structured_llm = model.with_structured_output(state.extraction_schema)
    
    result = structured_llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "Produce a structured output fro these notes",
            }
        ]
    )
    
    return {"info": result}


def reflection(state: OverallState) -> dict[str, Any]:
    """Reflect on the extracted information and generate search queries to find mission information"""
    
    structured_llm = model.with_structured_output(ReflectionOutput)
    
    #Format reflection prompt
    system_prompt = REFLECTION_PROMPT.format(
        schema = json.dumps(state.extraction_schema, indent=2),
        info=state.info
    )
    
    #Invoke
    result = cast(
        ReflectionOutput,
        structured_llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": "Produce a structured reflection output.",
                },
            ]
        )
    )
    
    if result.is_satisfactory:
        return {"is_satisfactory": result.is_satisfactory}
    else:
        return {
            "is_satisfactory": result.is_satisfactory,
            "search_queries": result.search_queries,
            "reflection_steps_taken": state.reflection_steps_taken + 1,
        }

def route_from_reflection(
    state: OverallState, config: RunnableConfig
) -> Literal[END, "research_person"]:  # type: ignore
    """Route the graph based on the reflection output."""
    # Get configuration
    configurable = Configuration.from_runnable_config(config)

    # If we have satisfactory results, end the process
    if state.is_satisfactory:
        return END

    # If results aren't satisfactory but we haven't hit max steps, continue research
    if state.reflection_steps_taken <= configurable.max_reflection_steps:
        return "research_person"

    # If we've exceeded max steps, end even if not satisfactory
    return END

workflow = StateGraph(OverallState, input=InputState, output=OutputState, config_schema=Configuration)


workflow.add_node("generate_queries",generate_queries)
workflow.add_node("research_person",research_person)
workflow.add_node("gather_notes_extract_schema",gather_notes_extract_schema)
workflow.add_node("reflection",reflection)

workflow.set_entry_point("generate_queries")
workflow.add_edge("generate_queries", "research_person")
workflow.add_edge("research_person", "gather_notes_extract_schema")
workflow.add_edge("gather_notes_extract_schema", "reflection")
workflow.add_conditional_edges("reflection", route_from_reflection)

graph = workflow.compile()

