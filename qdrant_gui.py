#!/usr/bin/env python3
"""
Ders Seçim Chatbot — Flask API Server

REST API endpoints for the course selection chatbot.
All data stored in Qdrant. Semantic routing is automatic.

Usage:
    python qdrant_gui.py
    # Production: gunicorn -w 4 -b 0.0.0.0:5000 qdrant_gui:app
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import sys
import asyncio
import time
import logging

from config import config, load_config_from_qdrant, save_config_to_qdrant

try:
    from rag_qdrant import (
        VectorIndex,
        ingest_documents,
        answer_query_async,
    )
except ImportError as e:
    print(f"Error importing RAG system: {e}")
    sys.exit(1)

try:
    from router import route_and_answer, get_router
    ROUTER_AVAILABLE = True
except ImportError:
    ROUTER_AVAILABLE = False
    logging.warning("Semantic router not available — falling back to direct LLM")

app = Flask(__name__)
CORS(app)

# Global index instance
index_instance = None

logger = logging.getLogger(__name__)


def get_index() -> VectorIndex:
    """Get or create the vector index (connects to Qdrant)."""
    global index_instance

    if index_instance is None:
        index_instance = VectorIndex()
        index_instance.ensure_ready()
        logger.info("VectorIndex initialized, connected to Qdrant")

    return index_instance


# ========================= ENDPOINTS =========================

@app.route('/')
def home():
    """Serve the GUI."""
    return send_file('rag_gui.html')


@app.route('/api/ask', methods=['POST'])
def api_ask():
    """
    Answer a question — routing is automatic.
    If semantic router is available, it decides easy/hard/reject.
    Otherwise falls back to direct RAG pipeline.

    Request: {"question": "...", "top_k": 5}
    """
    try:
        data = request.json
        question = data.get('question')
        top_k = data.get('top_k', config.TOP_K)

        if not question:
            return jsonify({'error': 'Question is required'}), 400

        # Use semantic router if available (automatic easy/hard/reject)
        if ROUTER_AVAILABLE:
            result = asyncio.run(route_and_answer(question, verbose=True))
            return jsonify(result)
        else:
            # Fallback: direct RAG pipeline
            try:
                index = get_index()
            except Exception as e:
                return jsonify({
                    'error': 'Index not available. Please ingest documents first.',
                    'details': str(e)
                }), 404

            result = asyncio.run(
                answer_query_async(question, index, top_k=top_k, verbose=False)
            )
            return jsonify(result)

    except Exception as e:
        app.logger.error(f"Error in ask endpoint: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/search', methods=['POST'])
def api_search():
    """
    Search for relevant chunks without LLM.

    Request: {"query": "...", "top_k": 5}
    """
    try:
        data = request.json
        query = data.get('query')
        top_k = data.get('top_k', config.TOP_K)

        if not query:
            return jsonify({'error': 'Query is required'}), 400

        try:
            index = get_index()
        except Exception as e:
            return jsonify({
                'error': 'Index not available.',
                'details': str(e)
            }), 404

        results = index.search(query, top_k=top_k)

        return jsonify({
            'query': query,
            'results': [
                {
                    'rank': r.rank,
                    'similarity': r.similarity,
                    'doc_id': r.chunk.doc_id,
                    'source': r.chunk.source,
                    'text': r.chunk.text
                }
                for r in results
            ]
        })

    except Exception as e:
        app.logger.error(f"Error in search endpoint: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ingest', methods=['POST'])
def api_ingest():
    """
    Ingest documents from a folder into Qdrant.

    Request: {"folder": "./data", "force": false}
    """
    try:
        data = request.json
        folder = data.get('folder')
        force = data.get('force', False)

        if not folder:
            return jsonify({'error': 'Folder path is required'}), 400

        from pathlib import Path
        folder_path = Path(folder)
        if not folder_path.exists():
            return jsonify({'error': f'Folder not found: {folder}'}), 404

        start_time = time.time()
        index = ingest_documents(folder_path, force_rebuild=force, show_progress=False)
        elapsed = time.time() - start_time

        # Update global index
        global index_instance
        index_instance = index

        return jsonify({
            'success': True,
            'total_chunks': len(index.chunks),
            'total_files': len(set(c.source for c in index.chunks)),
            'elapsed_time': elapsed,
            'storage': 'Qdrant'
        })

    except Exception as e:
        app.logger.error(f"Error in ingest endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/info', methods=['GET'])
def api_info():
    """Get index information from Qdrant."""
    try:
        index = get_index()
        info = index.get_collection_info()

        return jsonify({
            'collection': config.QDRANT_COLLECTION,
            'qdrant_url': config.QDRANT_URL,
            'points_count': info['points_count'],
            'vectors_count': info['vectors_count'],
            'status': info['status'],
            'config': {
                'embed_model': config.EMBED_MODEL,
                'chunk_size': config.CHUNK_SIZE,
                'chunk_overlap': config.CHUNK_OVERLAP,
                'device': config.EMBED_DEVICE,
                'llama_cpp_url': config.LLAMA_CPP_URL,
            },
            'cache_stats': index.cache.stats(),
        })

    except Exception as e:
        app.logger.error(f"Error in info endpoint: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check — reports status of all 3 Qdrant collections."""
    from qdrant_client import QdrantClient

    qdrant_available = False
    collections_status = {}

    try:
        client = QdrantClient(url=config.QDRANT_URL)
        existing = [c.name for c in client.get_collections().collections]
        qdrant_available = True

        for coll in [config.QDRANT_COLLECTION, config.QDRANT_CONFIG_COLLECTION,
                     config.QDRANT_CACHE_COLLECTION]:
            collections_status[coll] = coll in existing

    except Exception as e:
        logger.warning(f"Qdrant health check failed: {e}")

    return jsonify({
        'status': 'healthy' if qdrant_available else 'degraded',
        'qdrant_available': qdrant_available,
        'collections': collections_status,
        'router_available': ROUTER_AVAILABLE,
        'storage': 'Qdrant (all data)',
    })


@app.route('/api/qdrant/reset', methods=['POST'])
def reset_qdrant():
    """Delete Qdrant collections and start fresh."""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=config.QDRANT_URL)

        deleted = []
        for coll in [config.QDRANT_COLLECTION, config.QDRANT_CONFIG_COLLECTION,
                     config.QDRANT_CACHE_COLLECTION]:
            try:
                client.delete_collection(coll)
                deleted.append(coll)
            except Exception:
                pass

        # Reset global index
        global index_instance
        index_instance = None

        return jsonify({
            'success': True,
            'deleted_collections': deleted,
            'message': 'All Qdrant collections deleted'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ========================= MAIN =========================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Ders Seçim Chatbot — API Server')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    print(f"""
╔═══════════════════════════════════════════════════════════╗
║          Ders Seçim Chatbot — API Server                  ║
╠═══════════════════════════════════════════════════════════╣
║  Server:     http://{args.host}:{args.port}                          ║
║  Qdrant:     {config.QDRANT_URL:<43}║
║  LLM:        {config.LLAMA_CPP_URL:<43}║
║  Router:     {'✓ Active' if ROUTER_AVAILABLE else '✗ Disabled':<43}║
╠═══════════════════════════════════════════════════════════╣
║  Storage:    Qdrant (all data — zero local files)         ║
╠═══════════════════════════════════════════════════════════╣
║  Endpoints:                                               ║
║    POST /api/ask     — Ask (auto-routed easy/hard/reject)║
║    POST /api/search  — Search chunks (no LLM)            ║
║    POST /api/ingest  — Ingest documents                  ║
║    GET  /api/info    — Collection info                   ║
║    GET  /api/health  — Health check                      ║
║    POST /api/qdrant/reset — Reset all collections        ║
╚═══════════════════════════════════════════════════════════╝
    """)

    app.run(host=args.host, port=args.port, debug=args.debug)