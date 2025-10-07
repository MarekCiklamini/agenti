from google import genai
import os
from dotenv import load_dotenv

from typing import Annotated, Any, List
from typing_extensions import TypedDict

from langchain_openai import AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from visualizer import visualize
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import START, END


load_dotenv()


llm = AzureChatOpenAI(
    api_version="2025-01-01-preview",
    azure_endpoint=os.environ.get("ENDPOINT"),
    azure_ad_token=os.environ.get("AZURE_OPENAI_API_KEY")
)

# ---------------------------
# Mars Terraforming Graph (Analogous to Joke Graph)
# ---------------------------

import requests
import json
from typing_extensions import TypedDict

# Mars Terraforming State (analogous to Joke State)
class MarsState(TypedDict):
    terraforming_plan: str  # analogous to 'theme'
    atmospheric_data: str   # analogous to 'jokeText'
    resource_data: str      # analogous to 'storyText'
    habitat_data: str       # analogous to 'memeText'

# MCP Client helper function
def call_mcp_tool(tool_name: str, arguments: dict = None):
    """Call MCP server tool - synchronous version for LangGraph"""
    if arguments is None:
        arguments = {}
    
    # Use session from previous successful connection
    session_id = "fb34b771636243d5b7e1e64be6f4c3f9"  # From successful test
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "mcp-session-id": session_id
    }
    
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments
        },
        "id": 1
    }
    
    try:
        response = requests.post("http://localhost:8001/mcp", 
                               json=payload, 
                               headers=headers,
                               timeout=10)
        
        # Parse SSE response
        lines = response.text.strip().split('\n')
        for line in lines:
            if line.startswith('data: '):
                data = json.loads(line[6:])
                if "result" in data and "content" in data["result"]:
                    content = data["result"]["content"][0]["text"]
                    return json.loads(content)
        
        return {"error": "No valid response"}
        
    except Exception as e:
        return {"error": str(e)}

# Atmospheric Analysis Node (analogous to JokeNode)
def AtmosphericNode(state: MarsState):
    """Analyze atmospheric conditions for terraforming plan"""
    
    # Search existing knowledge about atmospheric conversion
    search_result = call_mcp_tool("fetch_from_qdb", {
        "collection_name": "TerraformingMars",
        "query": f"atmospheric conversion techniques for {state['terraforming_plan']}",
        "tenant": "mars_project",
        "top_k": 3,
        "score_threshold": 0.1
    })
    
    # Generate atmospheric analysis using LLM + MCP data
    mcp_data = ""
    if "results" in search_result and search_result["results"]:
        mcp_data = "\n".join([r.get("text", "") for r in search_result["results"]])
    
    messages = [
        SystemMessage("You are an atmospheric engineer specializing in Mars terraforming!"),
        HumanMessage(f"""
        Terraforming Plan: {state['terraforming_plan']}
        
        Existing Knowledge from Database:
        {mcp_data}
        
        Provide a detailed atmospheric conversion analysis including:
        1. Current atmospheric composition 
        2. Required greenhouse gas releases
        3. Timeline for atmospheric thickening
        4. Pressure and temperature targets
        """),
    ]
    response = llm.invoke(messages)
    return {"atmospheric_data": response}

# Resource Assessment Node (analogous to StoryNode) 
def ResourceNode(state: MarsState):
    """Assess resource availability for terraforming plan"""
    
    # Search for resource-related knowledge
    search_result = call_mcp_tool("fetch_from_qdb", {
        "collection_name": "TerraformingMars", 
        "query": f"water extraction and resource mining for {state['terraforming_plan']}",
        "tenant": "mars_project",
        "top_k": 3,
        "score_threshold": 0.1
    })
    
    mcp_data = ""
    if "results" in search_result and search_result["results"]:
        mcp_data = "\n".join([r.get("text", "") for r in search_result["results"]])
    
    messages = [
        SystemMessage("You are a planetary resource specialist!"),
        HumanMessage(f"""
        Terraforming Plan: {state['terraforming_plan']}
        Atmospheric Analysis: {state['atmospheric_data'].content if hasattr(state['atmospheric_data'], 'content') else str(state['atmospheric_data'])}
        
        Existing Knowledge from Database:
        {mcp_data}
        
        Provide a comprehensive resource assessment including:
        1. Water ice locations and extraction methods
        2. Mineral resources for construction
        3. Energy requirements and sources
        4. Supply chain logistics
        """),
    ]
    response = llm.invoke(messages)
    return {"resource_data": response}

# Habitat Planning Node (analogous to MemeNode)
def HabitatNode(state: MarsState):
    """Plan habitat construction based on atmospheric and resource analysis"""
    
    # Search for habitat construction knowledge
    search_result = call_mcp_tool("fetch_from_qdb", {
        "collection_name": "TerraformingMars",
        "query": f"habitat construction and life support for {state['terraforming_plan']}",
        "tenant": "mars_project", 
        "top_k": 3,
        "score_threshold": 0.1
    })
    
    mcp_data = ""
    if "results" in search_result and search_result["results"]:
        mcp_data = "\n".join([r.get("text", "") for r in search_result["results"]])
    
    messages = [
        SystemMessage("You are a Mars habitat architect and life support engineer!"),
        HumanMessage(f"""
        Terraforming Plan: {state['terraforming_plan']}
        Atmospheric Analysis: {state['atmospheric_data'].content if hasattr(state['atmospheric_data'], 'content') else str(state['atmospheric_data'])}
        Resource Assessment: {state['resource_data'].content if hasattr(state['resource_data'], 'content') else str(state['resource_data'])}
        
        Existing Knowledge from Database:
        {mcp_data}
        
        Design comprehensive habitat infrastructure including:
        1. Radiation shielding using available materials
        2. Pressurized dome specifications
        3. Life support system integration
        4. Agricultural facilities for food production
        5. Integration with atmospheric conversion timeline
        """),
    ]
    response = llm.invoke(messages)
    
    # Store the complete terraforming plan in MCP database
    plan_document = {
        "id": "550e8400-e29b-41d4-a716-446655440099",
        "text": f"Complete terraforming plan: {response.content}",
        "title": f"Habitat Plan for {state['terraforming_plan']}",
        "doc_id": f"habitat_plan_{state['terraforming_plan'].lower().replace(' ', '_')}",
        "tenant": "mars_project",
        "lang": "en",
        "category": "habitat_planning"
    }
    
    # Add to MCP database
    add_result = call_mcp_tool("add_documents", {
        "collection_name": "TerraformingMars",
        "documents": [plan_document]
    })
    
    return {"habitat_data": response}

# Build the Mars Terraforming Graph
print("\n🔴 Building Mars Terraforming Graph...")
mars_builder = StateGraph(MarsState)
mars_builder.add_node("atmospheric", AtmosphericNode)
mars_builder.add_node("resources", ResourceNode)  
mars_builder.add_node("habitat", HabitatNode)

# Create the same linear flow as the Joke graph
mars_builder.add_edge(START, "atmospheric")
mars_builder.add_edge("atmospheric", "resources")
mars_builder.add_edge("resources", "habitat")
mars_builder.add_edge("habitat", END)

# Compile the Mars graph
mars_graph = mars_builder.compile()

# Visualize the Mars terraforming graph
visualize(mars_graph, "mars_terraforming_graph.png")

# ---------------------------
# Run the Mars Terraforming Graph
# ---------------------------
print("\n🚀 Running Mars Terraforming Analysis...")

terraforming_plan = "Olympus Mons Base Establishment"
initial_mars_state = {
    "atmospheric_data": "", 
    "resource_data": "", 
    "habitat_data": "", 
    "terraforming_plan": terraforming_plan
}

try:
    mars_result = mars_graph.invoke(initial_mars_state)
    
    print("\n" + "="*60)
    print("🌍 MARS TERRAFORMING COMPLETE ANALYSIS")
    print("="*60)
    print(f"📋 Plan: {terraforming_plan}")
    print("\n🌫️ ATMOSPHERIC ANALYSIS:")
    print(mars_result["atmospheric_data"].content)
    print("\n💧 RESOURCE ASSESSMENT:")
    print(mars_result["resource_data"].content)
    print("\n🏗️ HABITAT INFRASTRUCTURE:")
    print(mars_result["habitat_data"].content)
    
    print("\n✅ Terraforming analysis complete and stored in MCP database!")
    
except Exception as e:
    print(f"❌ Mars terraforming analysis failed: {e}")
    print("Make sure your MCP server is running on localhost:8001")
