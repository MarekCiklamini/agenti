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
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import START, END


load_dotenv()

# llms
# llm_openai = AzureChatOpenAI(
llm = AzureChatOpenAI(
    api_version="2025-01-01-preview",
    azure_endpoint=os.environ.get("ENDPOINT"),
    azure_ad_token=os.environ.get("AZURE_OPENAI_API_KEY")
)

# Tool
search = TavilySearchResults(max_results=2)
llm_with_search = llm.bind_tools([search])  # Bind the tool to the model

# ---------------------------
# Mars Terraforming Graph
# ---------------------------

import requests
import json
import uuid
from typing_extensions import TypedDict

# Mars Terraforming State (analogous to Joke State)
class MarsState(TypedDict):
    terraforming_plan: str  
    atmospheric_data: str   
    resource_data: str      
    habitat_data: str      

# Global session ID for MCP
_mcp_session_id = None

def initialize_mcp_session():
    """Initialize MCP session and return session ID"""
    global _mcp_session_id
    
    print("🔌 Initializing MCP session...")
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream"
    }
    
    # Step 1: Initialize protocol
    init_payload = {
        "jsonrpc": "2.0",
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "Mars-Terraforming-LangGraph",
                "version": "1.0.0"
            }
        },
        "id": 1
    }
    
    try:
        print("📡 Connecting to MCP server on localhost:8001...")
        response = requests.post("http://localhost:8001/mcp", 
                               json=init_payload, 
                               headers=headers,
                               timeout=10)
        
        # Get session ID from headers
        session_id = response.headers.get('mcp-session-id')
        if session_id:
            _mcp_session_id = session_id
            headers['mcp-session-id'] = session_id
            
            # Send initialized notification
            notify_payload = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {}
            }
            
            requests.post("http://localhost:8001/mcp", 
                         json=notify_payload, 
                         headers=headers,
                         timeout=5)
            
            print(f"✅ MCP session initialized: {session_id[:8]}...")
            return session_id
        else:
            print("❌ No session ID received from server")
            return None
            
    except Exception as e:
        print(f"❌ Session initialization failed: {e}")
        print("💡 Make sure your MCP server is running: python mcp-server-mars.py")
        return None

# MCP Client helper function
def call_mcp_tool(tool_name: str, arguments: dict = None):
    """Call MCP server tool - synchronous version for LangGraph"""
    global _mcp_session_id
    
    if arguments is None:
        arguments = {}
    
    # Initialize session if not already done
    if not _mcp_session_id:
        _mcp_session_id = initialize_mcp_session()
        if not _mcp_session_id:
            return {"error": "Failed to initialize MCP session"}
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "mcp-session-id": _mcp_session_id
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
        print(f"🔧 MCP Tool '{tool_name}' error: {e}")
        return {"error": str(e)}

# Atmospheric Analysis Node
def AtmosphericNode(state: MarsState):
    """Analyze atmospheric conditions for terraforming plan"""
    
    try:
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
            SystemMessage("You are an atmospheric engineer specializing in Mars terraforming! Use web search to get latest scientific data if needed."),
            HumanMessage(f"""
            Terraforming Plan: {state['terraforming_plan']}
            
            Existing Knowledge from Database:
            {mcp_data}
            
            Provide a detailed atmospheric conversion analysis including:
            1. Current atmospheric composition 
            2. Required greenhouse gas releases
            3. Timeline for atmospheric thickening
            4. Pressure and temperature targets
            
            Use web search for latest Mars atmospheric research if helpful.
            """),
        ]
        
        print("🔍 AtmosphericNode: Calling LLM with search...")
        response = llm_with_search.invoke(messages)
        
        # Handle tool calls if present
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"🔍 AtmosphericNode: {len(response.tool_calls)} tool call(s) made")
            
            # Execute tool calls and create proper ToolMessage responses
            tool_messages = [response]  # Start with the original AI message containing tool_calls
            
            for tool_call in response.tool_calls:
                if tool_call['name'] == 'tavily_search_results_json':
                    try:
                        search_tool = TavilySearchResults(max_results=2)
                        tool_result = search_tool.invoke(tool_call['args']['query'])
                        print(f"✅ AtmosphericNode: Web search completed")
                        
                        # Create proper ToolMessage with tool_call_id
                        tool_messages.append(ToolMessage(
                            content=str(tool_result),
                            tool_call_id=tool_call['id']
                        ))
                    except Exception as search_error:
                        print(f"❌ AtmosphericNode: Search failed: {search_error}")
                        tool_messages.append(ToolMessage(
                            content=f"Search failed: {search_error}",
                            tool_call_id=tool_call['id']
                        ))
            
            # Get final response with proper tool message format
            if len(tool_messages) > 1:  # Has tool responses
                final_messages = messages + tool_messages
                final_response = llm.invoke(final_messages)
                return {"atmospheric_data": final_response}
        
        # If no tool calls or no content, use regular LLM
        if not response.content:
            print("⚠️ AtmosphericNode: No content from llm_with_search, using regular LLM")
            response = llm.invoke(messages)
        
        return {"atmospheric_data": response}
        
    except Exception as e:
        print(f"❌ AtmosphericNode error: {e}")
        # Fallback to regular LLM without search
        try:
            messages = [
                SystemMessage("You are an atmospheric engineer specializing in Mars terraforming!"),
                HumanMessage(f"Terraforming Plan: {state['terraforming_plan']}\n\nProvide atmospheric conversion analysis.")
            ]
            response = llm.invoke(messages)
            return {"atmospheric_data": response}
        except Exception as fallback_error:
            print(f"❌ AtmosphericNode fallback error: {fallback_error}")
            return {"atmospheric_data": AIMessage(content="Error in atmospheric analysis")}

# Resource Assessment Node 
def ResourceNode(state: MarsState):
    """Assess resource availability for terraforming plan"""
    
    try:
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
            SystemMessage("You are a planetary resource specialist! Use web search to find latest Mars resource data if needed."),
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
            
            Use web search for latest Mars resource surveys and NASA data if helpful.
            """),
        ]
        
        print("🔍 ResourceNode: Calling LLM with search...")
        response = llm_with_search.invoke(messages)
        
        # Handle tool calls if present
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"🔍 ResourceNode: {len(response.tool_calls)} tool call(s) made")
            
            # Execute tool calls and create proper ToolMessage responses
            tool_messages = [response]  # Start with the original AI message containing tool_calls
            
            for tool_call in response.tool_calls:
                if tool_call['name'] == 'tavily_search_results_json':
                    try:
                        search_tool = TavilySearchResults(max_results=2)
                        tool_result = search_tool.invoke(tool_call['args']['query'])
                        print(f"✅ ResourceNode: Web search completed")
                        
                        # Create proper ToolMessage with tool_call_id
                        tool_messages.append(ToolMessage(
                            content=str(tool_result),
                            tool_call_id=tool_call['id']
                        ))
                    except Exception as search_error:
                        print(f"❌ ResourceNode: Search failed: {search_error}")
                        tool_messages.append(ToolMessage(
                            content=f"Search failed: {search_error}",
                            tool_call_id=tool_call['id']
                        ))
            
            # Get final response with proper tool message format
            if len(tool_messages) > 1:  # Has tool responses
                final_messages = messages + tool_messages
                final_response = llm.invoke(final_messages)
                return {"resource_data": final_response}
        
        # If no tool calls or no content, use regular LLM
        if not response.content:
            print("⚠️ ResourceNode: No content from llm_with_search, using regular LLM")
            response = llm.invoke(messages)
        
        return {"resource_data": response}
        
    except Exception as e:
        print(f"❌ ResourceNode error: {e}")
        # Fallback to regular LLM without search
        try:
            messages = [
                SystemMessage("You are a planetary resource specialist!"),
                HumanMessage(f"Terraforming Plan: {state['terraforming_plan']}\n\nProvide resource assessment analysis.")
            ]
            response = llm.invoke(messages)
            return {"resource_data": response}
        except Exception as fallback_error:
            print(f"❌ ResourceNode fallback error: {fallback_error}")
            return {"resource_data": AIMessage(content="Error in resource analysis")}

# Habitat Planning Node
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
    
    try:
        messages = [
            SystemMessage("You are a Mars habitat architect and life support engineer! Use web search to find latest habitat technologies if needed."),
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
            
            Use web search for latest Mars habitat designs and space agriculture if helpful.
            """),
        ]
        
        print("🔍 HabitatNode: Calling LLM with search...")
        response = llm_with_search.invoke(messages)
        
        # Handle tool calls if present
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"🔍 HabitatNode: {len(response.tool_calls)} tool call(s) made")
            
            # Execute tool calls and create proper ToolMessage responses
            tool_messages = [response]  # Start with the original AI message containing tool_calls
            
            for tool_call in response.tool_calls:
                if tool_call['name'] == 'tavily_search_results_json':
                    try:
                        search_tool = TavilySearchResults(max_results=2)
                        tool_result = search_tool.invoke(tool_call['args']['query'])
                        print(f"✅ HabitatNode: Web search completed")
                        
                        # Create proper ToolMessage with tool_call_id
                        tool_messages.append(ToolMessage(
                            content=str(tool_result),
                            tool_call_id=tool_call['id']
                        ))
                    except Exception as search_error:
                        print(f"❌ HabitatNode: Search failed: {search_error}")
                        tool_messages.append(ToolMessage(
                            content=f"Search failed: {search_error}",
                            tool_call_id=tool_call['id']
                        ))
            
            # Get final response with proper tool message format
            if len(tool_messages) > 1:  # Has tool responses
                final_messages = messages + tool_messages
                final_response = llm.invoke(final_messages)
                response = final_response
        
        # If no tool calls or no content, use regular LLM
        if not response.content:
            print("⚠️ HabitatNode: No content from llm_with_search, using regular LLM")
            response = llm.invoke(messages)
        
        # Store the complete terraforming plan in MCP database
        try:
            plan_document = {
                "id": str(uuid.uuid4()),  # Generate unique random UUID
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
            print("✅ HabitatNode: Plan stored in MCP database")
            
        except Exception as storage_error:
            print(f"⚠️ HabitatNode: Failed to store in database: {storage_error}")
        
        return {"habitat_data": response}
        
    except Exception as e:
        print(f"❌ HabitatNode error: {e}")
        # Fallback to regular LLM without search
        try:
            messages = [
                SystemMessage("You are a Mars habitat architect and life support engineer!"),
                HumanMessage(f"Terraforming Plan: {state['terraforming_plan']}\n\nDesign habitat infrastructure.")
            ]
            response = llm.invoke(messages)
            return {"habitat_data": response}
        except Exception as fallback_error:
            print(f"❌ HabitatNode fallback error: {fallback_error}")
            return {"habitat_data": AIMessage(content="Error in habitat analysis")}

# Initialize MCP Connection
def initialize_mars_mcp():
    """Initialize connection to Mars MCP server"""
    # First establish MCP session
    session_id = initialize_mcp_session()
    if not session_id:
        return False
    
    # Then initialize Qdrant connection
    print("🔌 Initializing Qdrant connection...")
    init_result = call_mcp_tool("initialize_qdrant")
    if "error" in init_result:
        print(f"❌ Qdrant initialization failed: {init_result['error']}")
        return False
    else:
        print(f"✅ Qdrant connected: {init_result.get('total_collections', 0)} collections available")
        return True

# Build the Mars Terraforming Graph
print("\n🔴 Building Mars Terraforming Graph...")
mars_builder = StateGraph(MarsState)
mars_builder.add_node("atmospheric", AtmosphericNode)
mars_builder.add_node("resources", ResourceNode)  
mars_builder.add_node("habitat", HabitatNode)

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
print("\n🚀 Initializing Mars Terraforming System...")

# Initialize MCP connection first
if not initialize_mars_mcp():
    print("❌ Failed to initialize MCP connection. Exiting...")
    exit(1)

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
    print(f"❌ Error details: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()
