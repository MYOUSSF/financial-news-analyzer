# 📌 Quick Reference Card

## Essential Commands

### Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run streamlit_app/app.py

# Run API
uvicorn src.api.main:app --reload

# Run tests
pytest tests/ -v

# Run demo
python demo.py
```

### Docker
```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Python Usage
```python
from src.agents.research_agent import ResearchAgent
from langchain.llms import OpenAI

llm = OpenAI(temperature=0.3)
agent = ResearchAgent(llm=llm, tools=[])
result = agent.execute({"symbol": "AAPL", "days_back": 7})
```

## API Endpoints

```
POST   /api/analyze              - Full analysis
GET    /api/stocks/{symbol}/news - Get news
POST   /api/sentiment/analyze    - Sentiment analysis
POST   /api/risks/detect         - Risk assessment
POST   /api/search/semantic      - Semantic search
GET    /health                   - Health check
```

## File Locations

```
Source Code:        src/
Agents:            src/agents/
API:               src/api/main.py
Dashboard:         streamlit_app/app.py
Tests:             tests/
Documentation:     docs/
Configuration:     config/
Data:              data/
```

## Environment Variables

```env
OPENAI_API_KEY=your-key
NEWSAPI_KEY=your-key
ALPHA_VANTAGE_KEY=your-key
LOG_LEVEL=INFO
```

## Tech Stack

**AI/ML:** LangChain, OpenAI GPT-4, Transformers, ChromaDB  
**Backend:** FastAPI, Python, Pydantic  
**Frontend:** Streamlit, Plotly  
**DevOps:** Docker, docker-compose, pytest  
**Data:** NewsAPI, Alpha Vantage, World Bank  

## Project Structure

```
├── src/agents/          # AI agents
├── src/api/             # REST API
├── src/tools/           # LangChain tools
├── streamlit_app/       # Dashboard
├── tests/               # Tests
├── docs/                # Documentation
└── notebooks/           # Examples
```

## Quick Start

1. `cp .env.example .env` (add your API keys)
2. `pip install -r requirements.txt`
3. `streamlit run streamlit_app/app.py`
4. Open http://localhost:8501

## URLs

- Dashboard: http://localhost:8501
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Key Features

✅ Multi-agent AI system  
✅ Real-time data integration  
✅ Sentiment analysis  
✅ Risk assessment  
✅ REST API  
✅ Interactive dashboard  
✅ Docker deployment  
✅ Comprehensive tests  

## For Demos

1. Start API and dashboard
2. Enter stock symbol (e.g., AAPL)
3. Run analysis
4. Show results in tabs
5. Explain architecture
6. Show API docs

## Contact

GitHub: [your-repo-url]  
Email: your.email@example.com  
