#!/usr/bin/env python3
"""
Flask API Server for Mini RAG GUI (Qdrant Version)

This server provides REST API endpoints to connect the web GUI
to the Mini RAG system using Qdrant as the vector database.

Usage:
    python rag_server_qdrant.py

The server will start on http://localhost:5000
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import sys
import asyncio
from pathlib import Path
import time
import json

# Import the RAG system
try:
    from rag_qdrant import (
        VectorIndex, 
        ingest_documents, 
        answer_query_async,
        META_FILE,
        INDEX_DIR,
        config,
        DocChunk
    )
except ImportError as e:
    print(f"Error importing RAG system: {e}")
    print("Make sure mini_rag_optimized.py is in the same directory")
    sys.exit(1)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global index instance
index_instance = None

def ensure_index_dir():
    """Ensure index directory exists"""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

def get_index():
    """Get or load the vector index"""
    global index_instance
    
    if index_instance is None:
        # Initialize index (it will connect to Qdrant)
        index_instance = VectorIndex()
        
        # Load ensures Qdrant collection exists
        index_instance.load(INDEX_DIR)
        
        # Load chunks metadata from local file if available
        if META_FILE.exists():
            try:
                with open(META_FILE, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                chunks_data = meta.get('chunks', [])
                index_instance.chunks = [DocChunk(**c) for c in chunks_data]
                print(f"Loaded {len(index_instance.chunks)} chunks metadata from local file")
            except Exception as e:
                print(f"Warning: Could not load metadata: {e}")
                index_instance.chunks = []
        else:
            # No local metadata, initialize empty
            index_instance.chunks = []
            print("No local metadata found. Qdrant collection may contain data.")
    
    return index_instance

def save_metadata(chunks):
    """Save chunks metadata to local file"""
    ensure_index_dir()
    
    meta = {
        'chunks': [
            {
                'doc_id': c.doc_id,
                'source': c.source,
                'text': c.text,
                'checksum': c.checksum,
                'chunk_index': c.chunk_index,
                'total_chunks': c.total_chunks,
                'metadata': c.metadata
            }
            for c in chunks
        ],
        'config': {
            'embed_model': config.EMBED_MODEL,
            'chunk_size': config.CHUNK_SIZE,
            'chunk_overlap': config.CHUNK_OVERLAP,
            'dimension': None,  # Will be set when index is built
            'device': config.EMBED_DEVICE
        }
    }
    
    with open(META_FILE, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

@app.route('/')
def home():
    """Serve the GUI"""
    return send_file('rag_gui.html')

@app.route('/api/ask', methods=['POST'])
def api_ask():
    """
    Answer a question using RAG
    
    Request body:
        {
            "question": "your question here",
            "top_k": 5  // optional
        }
    """
    try:
        data = request.json
        question = data.get('question')
        top_k = data.get('top_k', config.TOP_K)
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        # Get index
        try:
            index = get_index()
        except Exception as e:
            return jsonify({
                'error': 'Index not available. Please ingest documents first.',
                'details': str(e)
            }), 404
        
        # Answer query
        result = asyncio.run(
            answer_query_async(question, index, top_k=top_k, verbose=False)
        )
        
        return jsonify(result)
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        app.logger.error(f"Error in ask endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/search', methods=['POST'])
def api_search():
    """
    Search for relevant chunks without LLM
    
    Request body:
        {
            "query": "search terms",
            "top_k": 5  // optional
        }
    """
    try:
        data = request.json
        query = data.get('query')
        top_k = data.get('top_k', config.TOP_K)
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Get index
        try:
            index = get_index()
        except Exception as e:
            return jsonify({
                'error': 'Index not available. Please ingest documents first.',
                'details': str(e)
            }), 404
        
        # Search
        results = index.search(query, top_k=top_k)
        
        # Format results
        formatted_results = []
        for r in results:
            formatted_results.append({
                'rank': r.rank,
                'similarity': r.similarity,
                'doc_id': r.chunk.doc_id,
                'source': r.chunk.source,
                'text': r.chunk.text
            })
        
        return jsonify({
            'query': query,
            'results': formatted_results
        })
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        app.logger.error(f"Error in search endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ingest', methods=['POST'])
def api_ingest():
    """
    Ingest documents from a folder
    
    Request body:
        {
            "folder": "./data",
            "force": false  // optional
        }
    """
    try:
        data = request.json
        folder = data.get('folder')
        force = data.get('force', False)
        
        if not folder:
            return jsonify({'error': 'Folder path is required'}), 400
        
        folder_path = Path(folder)
        if not folder_path.exists():
            return jsonify({'error': f'Folder not found: {folder}'}), 404
        
        # Ingest documents (this will upload to Qdrant and save metadata locally)
        start_time = time.time()
        index = ingest_documents(folder_path, force_rebuild=force, show_progress=False)
        elapsed = time.time() - start_time
        
        # Save metadata to local file
        save_metadata(index.chunks)
        
        # Update global index
        global index_instance
        index_instance = index
        
        return jsonify({
            'success': True,
            'total_chunks': len(index.chunks),
            'total_files': len(set(c.source for c in index.chunks)),
            'elapsed_time': elapsed,
            'storage': 'Qdrant + Local Metadata'
        })
    
    except Exception as e:
        app.logger.error(f"Error in ingest endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/info', methods=['GET'])
def api_info():
    """
    Get index information
    """
    try:
        # Try to get from loaded index first
        if index_instance and hasattr(index_instance, 'chunks') and index_instance.chunks:
            chunks = index_instance.chunks
            sources = set(c.source for c in chunks)
            total_chars = sum(len(c.text) for c in chunks)
            avg_chunk_size = total_chars / len(chunks) if chunks else 0
            
            return jsonify({
                'total_chunks': len(chunks),
                'unique_sources': len(sources),
                'total_characters': total_chars,
                'avg_chunk_size': avg_chunk_size,
                'config': {
                    'embed_model': config.EMBED_MODEL,
                    'chunk_size': config.CHUNK_SIZE,
                    'chunk_overlap': config.CHUNK_OVERLAP,
                    'device': config.EMBED_DEVICE,
                    'llama_cpp_url': config.LLAMA_CPP_URL
                },
                'storage': 'Qdrant',
                'qdrant_url': 'http://172.18.2.251:30986/'
            })
        
        # Fall back to metadata file if index not loaded
        if META_FILE.exists():
            with open(META_FILE, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            
            chunks = meta.get('chunks', [])
            sources = set(c.get('source') for c in chunks)
            total_chars = sum(len(c.get('text', '')) for c in chunks)
            avg_chunk_size = total_chars / len(chunks) if chunks else 0
            
            return jsonify({
                'total_chunks': len(chunks),
                'unique_sources': len(sources),
                'total_characters': total_chars,
                'avg_chunk_size': avg_chunk_size,
                'config': meta.get('config', {}),
                'storage': 'Qdrant + Local Metadata',
                'qdrant_url': 'http://172.18.2.251:30986/'
            })
        
        # No data available
        return jsonify({
            'error': 'No index found',
            'message': 'Please ingest documents first',
            'storage': 'Qdrant',
            'qdrant_url': 'http://172.18.2.251:30986/'
        }), 404
    
    except Exception as e:
        app.logger.error(f"Error in info endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    """
    Get or update configuration
    """
    if request.method == 'GET':
        return jsonify({
            'embed_model': config.EMBED_MODEL,
            'chunk_size': config.CHUNK_SIZE,
            'chunk_overlap': config.CHUNK_OVERLAP,
            'llama_cpp_url': config.LLAMA_CPP_URL,
            'llama_cpp_model': config.LLAMA_CPP_MODEL,
            'llama_cpp_timeout': config.LLAMA_CPP_TIMEOUT,
            'top_k': config.TOP_K,
            'similarity_threshold': config.SIM_THRESHOLD,
            'qdrant_url': 'http://172.18.2.251:30986/',
            'device': config.EMBED_DEVICE
        })
    
    else:  # POST
        try:
            data = request.json
            
            # Update config (note: this only affects the current session)
            if 'embed_model' in data:
                config.EMBED_MODEL = data['embed_model']
            if 'chunk_size' in data:
                config.CHUNK_SIZE = int(data['chunk_size'])
            if 'chunk_overlap' in data:
                config.CHUNK_OVERLAP = int(data['chunk_overlap'])
            if 'llama_cpp_url' in data:
                config.LLAMA_CPP_URL = data['llama_cpp_url']
            if 'llama_cpp_model' in data:
                config.LLAMA_CPP_MODEL = data['llama_cpp_model']
            if 'top_k' in data:
                config.TOP_K = int(data['top_k'])
            if 'similarity_threshold' in data:
                config.SIM_THRESHOLD = float(data['similarity_threshold'])
            
            return jsonify({'success': True, 'message': 'Configuration updated'})
        
        except Exception as e:
            return jsonify({'error': str(e)}), 400

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    # Check Qdrant connection
    qdrant_available = False
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url="http://172.18.2.251:30986/")
        collections = client.get_collections()
        qdrant_available = True
        collection_exists = any(c.name == "mini_rag_collection" for c in collections.collections)
    except Exception as e:
        qdrant_available = False
        collection_exists = False
    
    return jsonify({
        'status': 'healthy' if qdrant_available else 'degraded',
        'qdrant_available': qdrant_available,
        'qdrant_collection_exists': collection_exists,
        'index_loaded': index_instance is not None,
        'metadata_exists': META_FILE.exists(),
        'storage': 'Qdrant'
    })

@app.route('/api/qdrant/reset', methods=['POST'])
def reset_qdrant():
    """Delete Qdrant collection and start fresh"""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url="http://172.18.2.251:30986/")
        
        try:
            client.delete_collection("mini_rag_collection")
            
            # Also clear local metadata
            if META_FILE.exists():
                META_FILE.unlink()
            
            # Reset global index
            global index_instance
            index_instance = None
            
            return jsonify({
                'success': True,
                'message': 'Qdrant collection and metadata deleted successfully'
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Collection may not exist: {str(e)}'
            })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Mini RAG API Server (Qdrant)')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Ensure index directory exists
    ensure_index_dir()
    
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║              Mini RAG API Server (Qdrant)                 ║
╠═══════════════════════════════════════════════════════════╣
║  Server:        http://{args.host}:{args.port}                      ║
║  GUI:           http://localhost:{args.port}                   ║
║  Qdrant:        http://172.18.2.251:30986/                ║
║  Qdrant UI:     http://172.18.2.251:30986/dashboard       ║
╠═══════════════════════════════════════════════════════════╣
║  Storage:       Qdrant Vector Database                    ║
║  Metadata:      Local JSON file (./mini_index_optimized)  ║
╠═══════════════════════════════════════════════════════════╣
║  Endpoints:                                               ║
║    POST /api/ask          - Ask a question with RAG      ║
║    POST /api/search       - Search chunks without LLM    ║
║    POST /api/ingest       - Ingest documents             ║
║    GET  /api/info         - Get index information        ║
║    GET  /api/config       - Get configuration            ║
║    POST /api/config       - Update configuration         ║
║    GET  /api/health       - Health check                 ║
║    POST /api/qdrant/reset - Reset Qdrant collection      ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    app.run(host=args.host, port=args.port, debug=args.debug)