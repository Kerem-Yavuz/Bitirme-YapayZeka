#!/bin/bash
# Setup script for Ders Seçim Chatbot with Qdrant

set -e

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║       Ders Seçim Chatbot — Setup Script                   ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python
echo -n "Checking Python... "
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION"
else
    echo -e "${RED}✗${NC} Python 3 not found!"
    exit 1
fi

# Check if Qdrant is running
QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
echo -n "Checking Qdrant at $QDRANT_URL... "
if curl -s --connect-timeout 5 "$QDRANT_URL" &> /dev/null; then
    echo -e "${GREEN}✓${NC} Qdrant is running"
else
    echo -e "${RED}✗${NC} Qdrant not reachable at $QDRANT_URL"
    echo "Make sure the Qdrant server is running"
    echo "You can set a different URL with: export QDRANT_URL=http://your-server:6333"
fi

# Check llama.cpp server
LLAMA_CPP_URL="${LLAMA_CPP_URL:-http://localhost:8080}"
echo -n "Checking llama.cpp server at $LLAMA_CPP_URL... "
if curl -s --connect-timeout 5 "$LLAMA_CPP_URL/health" &> /dev/null; then
    echo -e "${GREEN}✓${NC} llama.cpp server is running"
    # Auto-detect model from server
    MODEL_NAME=$(curl -s --connect-timeout 5 "$LLAMA_CPP_URL/v1/models" 2>/dev/null | python3 -c "import sys,json; data=json.load(sys.stdin); print(data['data'][0]['id'] if data.get('data') else 'unknown')" 2>/dev/null || echo "auto-detect")
    echo -e "${GREEN}✓${NC} Model: $MODEL_NAME (auto-detected)"
else
    echo -e "${YELLOW}!${NC} llama.cpp server not reachable"
    echo "Make sure the server is running at: $LLAMA_CPP_URL"
    echo "You can set a different URL with: export LLAMA_CPP_URL=http://your-server:8080"
fi

# Install Python dependencies
echo -n "Installing Python dependencies... "
if pip3 install -q -r requirements.txt 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Dependencies installed"
else
    echo -e "${YELLOW}!${NC} Some dependencies may need manual installation"
fi

# Check for data folder
echo -n "Checking for data folder... "
if [ -d "./data" ]; then
    file_count=$(find ./data -type f \( -name "*.txt" -o -name "*.pdf" -o -name "*.md" \) | wc -l)
    echo -e "${GREEN}✓${NC} Found ./data with $file_count documents"
else
    echo -e "${YELLOW}!${NC} No ./data folder found"
    echo "Create a ./data folder and add your documents (txt, pdf, md)"
fi

# Setup complete
echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║                    Setup Complete!                        ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo ""
echo "1. Start the server:"
echo "   python3 qdrant_gui.py"
echo ""
echo "2. Open your browser:"
echo "   http://localhost:5000"
echo ""
echo "3. Interactive API docs:"
echo "   http://localhost:5000/docs"
echo ""
echo "4. Ingest your documents:"
echo "   curl -X POST http://localhost:5000/api/ingest \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"folder\": \"./data\"}'"
echo ""
