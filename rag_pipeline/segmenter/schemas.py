from typing import List, Literal
from pydantic import BaseModel, Field

class Chunk(BaseModel):
    id: str = ""
    title : str = ""
    source : str = ""
    cleaned_text : str = ""
    summary: str = ""
    high_level_summary: list[str] = None
    keywords: List[str] = None  
    questions: List[str] = None   

class Document(BaseModel):
    id: str = ""
    name: str = ""
    source : str = ""
    page_count: int  = ""  
    category: Literal["SOP","Manual","Untitled"] = "Untitled"
    process_name: str = ""
    chunks : List[Chunk] = None

class Output(BaseModel):
    summary : str = Field(...,description="Summary of the text chunk")
    high_level_summary: list[str] = Field(..., description="Short, dense fact strings about the asset in this chunk - tag, name/type, key facts. A list, not a paragraph.")
    questions : list[str] = Field(...,description="Set of questions that the chunk can answer.")
    keywords : list[str] = Field(..., description="Keywords from text")    

class OutputBasic(BaseModel):
    summary: str = Field(..., description="Concise, information-dense summary of the chunk")
    questions: list[str] = Field(..., description="5-7 diverse questions answerable strictly from the chunk")
