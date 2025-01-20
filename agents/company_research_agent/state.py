from dataclasses import dataclass, field
from typing import Any, Optional,Annotated
import operator

DEFAULT_EXTRACTION_SCHEMA = {
    "title": "CompanyInfo",
    "description": "Comprehensive information about a company",
    "type": "object",
    "properties": {
        "company_name": {
            "type": "string",
            "description": "Official registered name of the company",
        },
        "founding_info": {
            "type": "object",
            "properties": {
                "founding_year": {
                    "type": "integer",
                    "description": "Year the company was founded"
                },
                "founding_location": {
                    "type": "string",
                    "description": "City and country where the company was founded"
                },
                "founder_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Names of the founding team members"
                }
            }
        },
        "company_details": {
            "type": "object",
            "properties": {
                "industry": {
                    "type": "string",
                    "description": "Primary industry sector"
                },
                "company_type": {
                    "type": "string",
                    "description": "Legal structure (e.g., LLC, Corporation, etc.)"
                },
                "employee_count": {
                    "type": "integer",
                    "description": "Current number of employees"
                },
                "headquarters": {
                    "type": "string",
                    "description": "Location of company headquarters"
                }
            }
        },
        "business_info": {
            "type": "object",
            "properties": {
                "product_description": {
                    "type": "string",
                    "description": "Detailed description of the company's main products or services"
                },
                "business_model": {
                    "type": "string",
                    "description": "Description of how the company generates revenue"
                },
                "target_market": {
                    "type": "string",
                    "description": "Description of the company's target customer segments"
                },
                "competitors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of main competitors"
                }
            }
        },
        "financial_info": {
            "type": "object",
            "properties": {
                "funding_summary": {
                    "type": "string",
                    "description": "Summary of the company's funding history"
                },
                "funding_rounds": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "date": {"type": "string"},
                            "round_type": {"type": "string"},
                            "amount": {"type": "string"},
                            "lead_investors": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        }
                    },
                    "description": "Detailed information about each funding round"
                },
                "revenue_range": {
                    "type": "string",
                    "description": "Estimated annual revenue range"
                },
                "public_status": {
                    "type": "string",
                    "description": "Whether the company is public or private"
                }
            }
        },
        "online_presence": {
            "type": "object",
            "properties": {
                "website": {
                    "type": "string",
                    "description": "Company's official website"
                },
                "social_media": {
                    "type": "object",
                    "properties": {
                        "linkedin": {"type": "string"},
                        "twitter": {"type": "string"},
                        "facebook": {"type": "string"}
                    },
                    "description": "Social media profiles"
                }
            }
        },
        "additional_info": {
            "type": "object",
            "properties": {
                "recent_news": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "date": {"type": "string"},
                            "title": {"type": "string"},
                            "summary": {"type": "string"}
                        }
                    },
                    "description": "Recent significant news about the company"
                },
                "awards_recognition": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Notable awards or recognition received"
                }
            }
        }
    },
    "required": ["company_name", "founding_info", "company_details", "business_info"]
}

@dataclass(kw_only=True)
class InputState:
    """Input state define the interface between the graph and the user (User input)."""
    
    company: str
    "Company to research on."
    
    extraction_schema: dict[str, Any] = field(
        default_factory=lambda: DEFAULT_EXTRACTION_SCHEMA
    )
    "The json schema defines the information the agent is tasked with filling out."
    
    user_notes: Optional[dict[str, Any]] = field(default=None)
    "Any notes from the user to start the research process."

@dataclass
class OverallState:
    "Input state defines the interface between"
    company: str
    "Company to research provided by the user."
    
    extraction_schema : dict[str, Any] = field(
        default_factory=lambda: DEFAULT_EXTRACTION_SCHEMA
    ) 
    "The json schema defines the information the agent is tasked with filling out."
    
    user_notes: str = field(default=None)
    "Any notes from the user to start the research process."
    
    search_queries: list = field(default=None)
    "List of generated search quires to find relevant information"
    
    completed_notes: Annotated[list, operator.add] = field(default_factory=list)
    "Noted from completed research related to the schema."
    
    info: dict[str, Any] = field(default=None)
    """ 
    A dictionary containing the extracted and processed information based on the users query and the graph execution
    This is the primary output of the enrichment process.
    """
    
    is_satisfactory: bool = field(default=None)
    "True if all required fields are well populated, False otherwise"

    reflection_steps_taken: int = field(default=0)
    "Number of times the reflection node has been executed"


@dataclass(kw_only=True)
class OutputState:
    """The response object for the end user.

    This class defines the structure of the output that will be provided
    to the user after the graph's execution is complete.
    """

    info: dict[str, Any]
    """
    A dictionary containing the extracted and processed information
    based on the user's query and the graph's execution.
    This is the primary output of the enrichment process.
    """