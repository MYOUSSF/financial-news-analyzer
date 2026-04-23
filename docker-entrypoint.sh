#!/bin/bash
# =============================================================================
# docker-entrypoint.sh — starts API + Dashboard inside the container
# =============================================================================
set -e

echo "=========================================="
echo " Financial News Analyzer"
echo "=========================================="

# 1. Initialize ChromaDB if not already done
echo "[1/3] Initializing vector database..."
python src/utils/init_db.py --path "${CHROMA_DB_PATH:-/app/data/chroma_db}" || {
    echo "WARNING: DB init failed — continuing anyway (will use empty store)"
}

# 2. Start FastAPI in the background
echo "[2/3] Starting FastAPI server on port 8000..."
uvicorn src.api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level "${LOG_LEVEL:-info}" \
    --workers 1 &
API_PID=$!

# 3. Start Streamlit in the foreground
echo "[3/3] Starting Streamlit dashboard on port 8501..."
streamlit run streamlit_app/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false &
DASH_PID=$!

echo ""
echo "Services running:"
echo "  API       → http://localhost:8000  (docs: /docs)"
echo "  Dashboard → http://localhost:8501"
echo ""

# Wait for either process to exit; if one dies, kill the other
wait -n $API_PID $DASH_PID
EXIT_CODE=$?

kill $API_PID $DASH_PID 2>/dev/null || true
exit $EXIT_CODE
