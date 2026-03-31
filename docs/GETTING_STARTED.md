# Getting Started Guide

This guide will help you get the Financial News Analyzer up and running quickly.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.9 or higher**
- **pip** (Python package manager)
- **Git**
- **Virtual environment tool** (venv, conda, or virtualenv)

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/financial-news-analyzer.git
cd financial-news-analyzer
```

### 2. Create Virtual Environment

Using venv:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Using conda:
```bash
conda create -n finanalyzer python=3.11
conda activate finanalyzer
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up API Keys

#### Required API Keys

1. **OpenAI API Key** (or Anthropic)
   - Sign up at: https://platform.openai.com/
   - Get your API key from the dashboard
   - Free tier available with usage limits

2. **NewsAPI Key**
   - Sign up at: https://newsapi.org/
   - Free tier: 100 requests/day
   - Get your API key instantly

3. **Alpha Vantage Key**
   - Sign up at: https://www.alphavantage.co/support/#api-key
   - Free tier: 5 requests/minute, 500/day
   - Get your API key instantly

#### Configure Environment Variables

Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```env
# Required
OPENAI_API_KEY=sk-your-openai-key-here
NEWSAPI_KEY=your-newsapi-key-here
ALPHA_VANTAGE_KEY=your-alpha-vantage-key-here

# Optional
LOG_LEVEL=INFO
```

### 5. Initialize the Database

```bash
python src/utils/init_db.py
```

This will:
- Create the ChromaDB vector store
- Set up the cache directory
- Initialize configuration files

### 6. Run the Application

#### Option A: Streamlit Dashboard (Recommended for first-time users)

```bash
streamlit run streamlit_app/app.py
```

The dashboard will open in your browser at `http://localhost:8501`

#### Option B: FastAPI Backend

```bash
uvicorn src.api.main:app --reload
```

API will be available at:
- API: `http://localhost:8000`
- Documentation: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

#### Option C: Both (Recommended for full functionality)

Terminal 1:
```bash
uvicorn src.api.main:app --reload
```

Terminal 2:
```bash
streamlit run streamlit_app/app.py
```

## First Analysis

### Using the Dashboard

1. Open the Streamlit dashboard
2. Enter a stock symbol (e.g., `AAPL`)
3. Adjust the analysis period (default: 7 days)
4. Select analysis options:
   - ✓ Sentiment Analysis
   - ✓ Risk Assessment
   - ✓ Historical Trends
5. Click "🚀 Run Analysis"
6. View results in different tabs

### Using the API

Test the API with curl:

```bash
# Analyze a stock
curl -X POST "http://localhost:8000/api/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "days_back": 7,
    "include_sentiment": true,
    "include_risk": true
  }'

# Get recent news
curl "http://localhost:8000/api/stocks/AAPL/news?days=7"

# Check system health
curl "http://localhost:8000/health"
```

Or use Python:

```python
import requests

# Analyze stock
response = requests.post(
    "http://localhost:8000/api/analyze",
    json={
        "symbol": "AAPL",
        "days_back": 7,
        "include_sentiment": True,
        "include_risk": True
    }
)

result = response.json()
print(f"Sentiment: {result['sentiment']['overall_sentiment']}")
print(f"Risk Level: {result['risk']['risk_level']}")
```

### Using Python SDK

```python
from src.chains.analysis_chain import FinancialAnalysisChain

# Initialize
chain = FinancialAnalysisChain()

# Analyze
result = chain.analyze_stock(
    symbol="AAPL",
    days_back=7,
    include_sentiment=True
)

print(result)
```

## Docker Deployment

### Using Docker Compose (Easiest)

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

Access:
- Dashboard: `http://localhost:8501`
- API: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`

### Using Docker Only

```bash
# Build image
docker build -t financial-analyzer .

# Run container
docker run -d \
  -p 8000:8000 \
  -p 8501:8501 \
  --env-file .env \
  --name financial-analyzer \
  financial-analyzer

# View logs
docker logs -f financial-analyzer

# Stop container
docker stop financial-analyzer
```

## Troubleshooting

### Issue: ImportError or ModuleNotFoundError

**Solution**: Ensure all dependencies are installed
```bash
pip install -r requirements.txt --upgrade
```

### Issue: API Key Errors

**Solution**: Verify your `.env` file
```bash
# Check if file exists
cat .env

# Verify keys are not empty
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('OpenAI:', bool(os.getenv('OPENAI_API_KEY'))); print('NewsAPI:', bool(os.getenv('NEWSAPI_KEY')))"
```

### Issue: Port Already in Use

**Solution**: Change the port or kill the process
```bash
# Find process using port 8501
lsof -i :8501  # On Mac/Linux
netstat -ano | findstr :8501  # On Windows

# Kill process (replace PID)
kill -9 PID

# Or use different port
streamlit run streamlit_app/app.py --server.port 8502
```

### Issue: ChromaDB Errors

**Solution**: Clear and reinitialize database
```bash
rm -rf data/chroma_db
python src/utils/init_db.py
```

### Issue: Rate Limit Errors from APIs

**Solution**: 
1. Check your API quotas
2. Enable caching (default)
3. Reduce request frequency
4. Consider upgrading API plans

### Issue: Slow Performance

**Solutions**:
1. Enable Redis caching
2. Reduce `days_back` parameter
3. Limit number of articles analyzed
4. Use faster LLM models

## Configuration

### Logging

Adjust log level in `.env`:
```env
LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

View logs:
```bash
tail -f logs/app.log
```

### Agent Behavior

Edit `config/agents_config.yaml`:
```yaml
research_agent:
  max_iterations: 5
  temperature: 0.3

sentiment_agent:
  confidence_threshold: 0.75

risk_agent:
  alert_threshold: 0.7
```

### Cache Settings

Configure caching in `.env`:
```env
CACHE_ENABLED=True
CACHE_TTL=3600  # 1 hour in seconds
CACHE_TYPE=disk  # disk, memory, redis
```

## Next Steps

### Learn More
- Read the [Architecture Overview](docs/architecture.md)
- Check out [API Reference](docs/api_reference.md)
- Explore [Jupyter Notebooks](notebooks/)

### Customize
- Add new data sources
- Create custom agents
- Modify analysis logic
- Build custom dashboards

### Contribute
- Report issues on GitHub
- Submit pull requests
- Share feedback
- Add documentation

## Getting Help

- **Documentation**: Check the `docs/` folder
- **Examples**: See `notebooks/` for examples
- **Issues**: Open an issue on GitHub
- **Email**: your.email@example.com

## Quick Reference

### Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run dashboard
streamlit run streamlit_app/app.py

# Run API
uvicorn src.api.main:app --reload

# Run with Docker
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Format code
black src/ tests/

# Run linter
flake8 src/ tests/

# Type checking
mypy src/
```

### Environment Variables Quick Reference

```env
# LLM
OPENAI_API_KEY=your-key

# Data Sources
NEWSAPI_KEY=your-key
ALPHA_VANTAGE_KEY=your-key

# Database
CHROMA_DB_PATH=./data/chroma_db

# Application
LOG_LEVEL=INFO
CACHE_ENABLED=True
DEBUG=False
```

## Success!

You should now have a fully functional Financial News Analyzer!

Try analyzing some stocks and exploring the different features. If you encounter any issues, check the troubleshooting section or reach out for help.

Happy analyzing! 📊
