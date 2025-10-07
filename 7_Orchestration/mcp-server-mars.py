# Mars Terraforming MCP Server - Minimal Version
# Contains only the tools needed for the Mars Terraforming LangGraph system

from fastmcp import FastMCP
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import PlainTextResponse
from datetime import datetime
import os
import sys
import logging
from typing import Dict, List, Any, Optional

# Qdrant imports
try:
    from qdrant_client import QdrantClient, models
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    print("WARNING: Qdrant client not installed. Install with: pip install qdrant-client", file=sys.stderr)

# Embedding function
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    print("WARNING: SentenceTransformers not installed. Install with: pip install sentence-transformers", file=sys.stderr)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize the MCP server
mcp = FastMCP("Mars Terraforming MCP Server")

# Global Qdrant client
qdrant_client = None

# ============= Helper Functions =============

def get_qdrant_client() -> Optional[QdrantClient]:
    """Get or create Qdrant client instance"""
    global qdrant_client
    
    if not QDRANT_AVAILABLE:
        return None
    
    if qdrant_client is None:
        qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
        qdrant_key = os.environ.get("QDRANT_KEY", "")
        
        try:
            if qdrant_key:
                qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
            else:
                qdrant_client = QdrantClient(url=qdrant_url)
            
            logger.info(f"Connected to Qdrant at {qdrant_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            return None
    
    return qdrant_client

def generate_embedding(text: str) -> List[float]:
    """Generate embedding for text using SentenceTransformers"""
    if not EMBEDDING_AVAILABLE:
        # Fallback: return random embedding for demo purposes
        import random
        return [random.random() for _ in range(384)]
    
    try:
        embedding = EMBEDDING_MODEL.encode(text).tolist()
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return [0.0] * 384

def build_filter(tenant: Optional[str] = None, **kwargs) -> Optional[Filter]:
    """Build Qdrant filter based on conditions"""
    conditions = []
    
    if tenant:
        conditions.append(FieldCondition(key="tenant", match=MatchValue(value=tenant)))
    
    for key, value in kwargs.items():
        if value is not None:
            conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
    
    return Filter(must=conditions) if conditions else None



@mcp.tool
def initialize_qdrant() -> Dict[str, Any]:
    """
    Initialize connection to Qdrant vector database
    
    Returns:
        Connection status and server information
    """
    if not QDRANT_AVAILABLE:
        return {"error": "Qdrant client not available. Please install qdrant-client."}
    
    client = get_qdrant_client()
    if not client:
        return {"error": "Failed to connect to Qdrant server"}
    
    try:
        # Test connection and get collections
        collections = client.get_collections()
        collection_list = [col.name for col in collections.collections]
        
        # Create TerraformingMars collection if it doesn't exist
        if "TerraformingMars" not in collection_list:
            try:
                client.create_collection(
                    collection_name="TerraformingMars",
                    vectors_config=models.VectorParams(
                        size=384,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info("Created TerraformingMars collection")
                collections = client.get_collections()
                collection_list = [col.name for col in collections.collections]
            except Exception as create_error:
                logger.warning(f"Could not create TerraformingMars collection: {create_error}")
        
        return {
            "status": "connected",
            "collections": collection_list,
            "total_collections": len(collection_list),
            "connection_url": os.environ.get("QDRANT_URL", "http://localhost:6333"),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        return {"error": f"Failed to initialize Qdrant: {str(e)}"}

@mcp.tool
def fetch_from_qdb(
    collection_name: str,
    query: str,
    tenant: Optional[str] = None,
    top_k: int = 5,
    score_threshold: float = 0.0
) -> Dict[str, Any]:
    """
    Fetch/search documents from Qdrant vector database for Mars terraforming knowledge
    
    Args:
        collection_name: Name of the collection to search
        query: Text query to search for
        tenant: Optional tenant filter (e.g., 'mars_project')
        top_k: Number of results to return
        score_threshold: Minimum similarity score threshold
    
    Returns:
        Search results with documents and metadata
    """
    if not QDRANT_AVAILABLE:
        return {"error": "Qdrant client not available"}
    
    client = get_qdrant_client()
    if not client:
        return {"error": "Failed to connect to Qdrant"}
    
    try:
        # Generate query embedding
        query_embedding = generate_embedding(query)
        
        # Build filter
        query_filter = build_filter(tenant=tenant)
        
        # Search the collection
        hits = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            query_filter=query_filter,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=False
        )
        
        # Format results
        results = []
        for hit in hits:
            result = {
                "id": str(hit.id),
                "score": hit.score,
                "text": hit.payload.get("text", ""),
                "title": hit.payload.get("title", ""),
                "doc_id": hit.payload.get("doc_id", ""),
                "category": hit.payload.get("category", ""),
                "tenant": hit.payload.get("tenant", "")
            }
            results.append(result)
        
        return {
            "query": query,
            "collection": collection_name,
            "results": results,
            "total_found": len(results),
            "search_params": {
                "top_k": top_k,
                "tenant": tenant,
                "score_threshold": score_threshold
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return {"error": f"Search failed: {str(e)}"}

@mcp.tool
def add_documents(
    collection_name: str,
    documents: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Add Mars terraforming documents to Qdrant collection
    
    Args:
        collection_name: Name of the collection (typically 'TerraformingMars')
        documents: List of documents with text, title, and metadata
    
    Returns:
        Upload status and statistics
    """
    if not QDRANT_AVAILABLE:
        return {"error": "Qdrant client not available"}
    
    client = get_qdrant_client()
    if not client:
        return {"error": "Failed to connect to Qdrant"}
    
    try:
        points = []
        for doc in documents:
            text = doc.get("text", "")
            if not text:
                continue
            
            # Generate embedding
            embedding = generate_embedding(text)
            
            # Create point with terraforming-specific payload
            point = models.PointStruct(
                id=doc.get("id", len(points)),
                vector=embedding,
                payload={
                    "text": text,
                    "title": doc.get("title", ""),
                    "doc_id": doc.get("doc_id", ""),
                    "tenant": doc.get("tenant", "mars_project"),
                    "lang": doc.get("lang", "en"),
                    "category": doc.get("category", "terraforming"),
                    "created_at": datetime.utcnow().isoformat()
                }
            )
            points.append(point)
        
        # Upload all points
        if points:
            client.upsert(
                collection_name=collection_name,
                points=points
            )
        
        return {
            "status": "success",
            "collection": collection_name,
            "documents_processed": len(documents),
            "points_uploaded": len(points),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to add documents: {e}")
        return {"error": f"Failed to add documents: {str(e)}"}

# ============= Health Check =============

@mcp.custom_route("/", methods=["GET"])
@mcp.custom_route("/healthcheck", methods=["GET"])
async def health_check():
    """Health check endpoint"""
    client = get_qdrant_client()
    if client:
        return PlainTextResponse("Mars Terraforming MCP Server is healthy")
    else:
        return PlainTextResponse("Mars Terraforming MCP Server - Qdrant connection failed", status_code=503)

# ============= Main =============

if __name__ == "__main__":
    import uvicorn
    
    # Check command line arguments for transport type
    if len(sys.argv) > 1 and sys.argv[1] == "--stdio":
        # STDIO transport for LLM integration
        mcp.run(transport="stdio")
    else:
        # HTTP transport for browser access
        custom_middleware = [
            Middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["GET", "POST", "OPTIONS"],
                allow_headers=["*"],
                expose_headers=["mcp-session-id", "*"]
            )
        ]
        
        http_app = mcp.http_app(
            path="/mcp",
            middleware=custom_middleware
        )
        
        print("🚀 Starting Mars Terraforming MCP Server...")
        print("🔗 Server available at: http://localhost:8001/mcp")
        print("💊 Health check at: http://localhost:8001/healthcheck")
        print("🌍 CORS enabled for browser access")
        print("")
        print("🔧 Available tools:")
        print("   • initialize_qdrant - Connect to vector database")
        print("   • fetch_from_qdb - Search terraforming knowledge")
        print("   • add_documents - Store terraforming plans")
        
        if not QDRANT_AVAILABLE:
            print("\n❌ WARNING: Qdrant client not available!")
            print("   Install with: pip install qdrant-client")
        
        if not EMBEDDING_AVAILABLE:
            print("\n❌ WARNING: SentenceTransformers not available!")
            print("   Install with: pip install sentence-transformers")
        
        print("\n⏹️  Press Ctrl+C to stop the server")
        
        try:
            uvicorn.run(http_app, host="127.0.0.1", port=8001)
        except KeyboardInterrupt:
            print("\n🛑 Mars Terraforming MCP Server stopped.")
