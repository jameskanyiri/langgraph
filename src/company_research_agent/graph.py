import json
import asyncio
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from pydantic import BaseModel,Field
from typing import cast, Any, Literal
from langgraph.graph import StateGraph, END
from tavily import AsyncTavilyClient


from src.company_research_agent.state import OverallState, InputState, OutputState
from src.company_research_agent.configuration import Configuration
from src.company_research_agent.utils import deduplicate_and_format_sources,format_all_notes
from src.company_research_agent.prompts import QUERY_WRITER_PROMPT,INFO_PROMPT,EXTRACTION_PROMPT,REFLECTION_PROMPT


#rate limiter
# Create a rate limiter
rate_limiter = InMemoryRateLimiter(
    requests_per_second=4,  # 4 request every 10 seconds
    check_every_n_seconds=0.1,  # Check token availability every 100ms
    max_bucket_size=10  # Maximum burst size
)

# Initialize ChatOpenAI with the rate limiter
model = ChatOpenAI(
    model="gpt-4o",
    rate_limiter=rate_limiter,
    temperature=0
)

tavily_async_client = AsyncTavilyClient()


class Queries(BaseModel):
    queries: list[str] = Field(
        description="List of search queries"
    )
    
class ReflectionOutput(BaseModel):
    is_satisfactory: bool = Field(
        description="True if all required fields are well populated, False otherwise"
    )
    missing_fields: list[str] = Field(
        description="List of field names that are missing or incomplete"
    )
    search_queries: list[str] = Field(
        description="If is_satisfactory is False, provide 1-3 targeted search queries to find the missing information"
    )
    reasoning: str = Field(description="Brief explanation of the assessment")


def generate_queries(state: OverallState, config: RunnableConfig):
    """Generate search queries based on user input and the extraction schema"""
    
    #Get configuration
    configuration  = Configuration.from_runnable_config(config)
    max_search_queries = configuration.max_search_queries
    
    #Generate search queries
    structured_llm = model.with_structured_output(Queries)
    
    #Format system instructions
    query_instructions = QUERY_WRITER_PROMPT.format(
        company = state.company,
        info = json.dumps(state.extraction_schema, indent=2),
        user_notes = state.user_notes,
        max_search_queries = max_search_queries
    )
    
    #Generate queries
    results = cast(
        Queries,
        structured_llm.invoke(
            [
                {"role": "system", "content": query_instructions},
                {
                    "role": "user",
                    "content": "Please generate a list of search queries related to the schema that you want to populate."
                }
            ]
        )
    )
    
    #Queries
    query_list = [query for query in results.queries]
    
    return {"search_queries": query_list}


async def research_company(state: OverallState, config: RunnableConfig)->dict[str, Any]:
    """Execute a multi-step web search and information extraction process
    
    This function performs the following steps:
    1. Extract concurrent web search queries using the Tavily API
    2. Deduplicate and format the search results
    """
    
    #Get configuration
    configuration = Configuration.from_runnable_config(config)
    max_search_results = configuration.max_search_results
    
    #Search tasks
    search_tasks = []
    for query in state.search_queries:
        search_tasks.append(
            tavily_async_client.search(
                query,
                max_results=max_search_results,
                include_raw_content=True,
                topic="general",
                search_depth="advanced"
            )
        )
        
    #Execute all search concurrently
    search_docs = await asyncio.gather(*search_tasks)
    
    #Deduplicate and format sources
    source_str = deduplicate_and_format_sources(
        search_docs,
        max_tokens_per_source=1000,
        include_raw_content=True
    )
    
    #Generate structured notes relevant to the extraction schema
    system_instruction = INFO_PROMPT.format(
        company = state.company,
        info = json.dumps(state.extraction_schema, indent=2),
        content = source_str,
        user_notes = state.user_notes
    )
    
    result = await model.ainvoke(system_instruction)
    
    return {"completed_notes": [str(result.content)]}


def gather_notes_extract_schema(state: OverallState) -> dict[str, Any]:
    """Gather notes from the web search and extract the schema fields"""
    
    #Format all nodes
    notes = format_all_notes(state.completed_notes)
    
    #Extract schema fields
    system_prompt = EXTRACTION_PROMPT.format(
        schema = json.dumps(state.extraction_schema, indent=2),
        notes = notes
    )
    
    structured_llm = model.with_structured_output(state.extraction_schema)
    
    
    result = structured_llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "Please help me extract the relevant information from the gathered notes."
            }
        ]
    ) 
    
    return {"info": result}


def reflection(state: OverallState) -> dict[str, Any]:
    """Reflection on the extracted information and generate search queries to find mission information"""
    
    structured_llm = model.with_structured_output(ReflectionOutput)
    
    #Format reflection prompt
    system_prompt = REFLECTION_PROMPT.format(
        schema = json.dumps(state.extraction_schema, indent=2),
        info = state.info
    )
    
    # Invoke
    result = cast(
        ReflectionOutput,
        structured_llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Produce a structured reflection output."},
            ]
        ),
    )

    if result.is_satisfactory:
        return {"is_satisfactory": result.is_satisfactory}
    else:
        return {
            "is_satisfactory": result.is_satisfactory,
            "search_queries": result.search_queries,
            "reflection_steps_taken": state.reflection_steps_taken + 1,
        }


def route_from_reflection(state: OverallState, config: RunnableConfig) -> Literal[END, "research_company"]:
    """Route based on the reflection output"""
    
    configurable = Configuration.from_runnable_config(config)
    
    #if we have satisfactory result then end the process
    if state.is_satisfactory:
        return END
    
    #if result aren't satisfactory but we haven't hit max steps, continue research
    if state.reflection_steps_taken <= configurable.max_reflection_steps:
        return "research_company"
    
    #If we've exceeded the steps, end even if not satisfactory
    return END
    
    
    


workflow = StateGraph(OverallState, input=InputState, output=OutputState, config_schema=Configuration)
workflow.add_node("generate_queries", generate_queries)
workflow.add_node("research_company", research_company)
workflow.add_node("gather_notes_extract_schema", gather_notes_extract_schema)
workflow.add_node("reflection", reflection)

workflow.set_entry_point("generate_queries")

workflow.add_edge("generate_queries", "research_company")
workflow.add_edge("research_company", "gather_notes_extract_schema")
workflow.add_edge("gather_notes_extract_schema", "reflection")
workflow.add_conditional_edges("reflection", route_from_reflection)


graph = workflow.compile()