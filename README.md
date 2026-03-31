# 📊 Financial News Analysis & Investment Research Assistant

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1.0+-green.svg)](https://python.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready AI-powered financial research assistant that analyzes real-time market news, performs sentiment analysis, and generates actionable investment insights using LangChain's multi-agent architecture.

![Dashboard Preview](docs/images/dashboard_preview.png)

## 🌟 Key Features

### Multi-Agent Architecture
- **Research Agent**: Retrieves and analyzes financial news from multiple sources
- **Sentiment Agent**: Performs advanced sentiment analysis on market news
- **Risk Agent**: Identifies potential risks and red flags in market events
- **Summary Agent**: Synthesizes insights into comprehensive reports

### Real-Time Data Integration
- 📰 Live news from NewsAPI (50+ financial sources)
- 📈 Stock market data from Alpha Vantage & Yahoo Finance
- 🌍 Global economic indicators from World Bank API
- 📊 Historical market events with sentiment scores

### RAG (Retrieval-Augmented Generation)
- Vector database for historical news storage (ChromaDB)
- Semantic search over past market events
- Context-aware responses with relevant historical data

### Production Features
- REST API with FastAPI
- Interactive Streamlit dashboard
- Comprehensive logging and monitoring
- Docker containerization
- Unit and integration tests

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.9+
pip or conda
OpenAI API key (or use Ollama for local LLMs)
NewsAPI key (free tier available)
Alpha Vantage API key (free)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/financial-news-analyzer.git
cd financial-news-analyzer
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys
```

5. **Initialize the database**
```bash
python src/utils/init_db.py
```

6. **Run the Streamlit dashboard**
```bash
streamlit run streamlit_app/app.py
```

## 📖 Usage

### Dashboard Interface

The Streamlit dashboard provides an intuitive interface for:
- Real-time news monitoring for specific stocks/sectors
- Sentiment analysis with visualizations
- Historical trend analysis
- Investment report generation
- Risk alerts and notifications

### API Endpoints

Start the FastAPI server:
```bash
uvicorn src.api.main:app --reload
```

**Available Endpoints:**

```bash
POST /api/analyze/news
GET /api/stocks/{symbol}/sentiment
POST /api/reports/generate
GET /api/risks/detect
POST /api/search/semantic
```

### Python SDK

```python
from src.chains.analysis_chain import FinancialAnalysisChain

# Initialize the chain
chain = FinancialAnalysisChain()

# Analyze a stock
result = chain.analyze_stock(
    symbol="AAPL",
    days_back=7,
    include_sentiment=True
)

print(result['summary'])
print(result['sentiment_score'])
print(result['risk_factors'])
```

### Command Line Interface

```bash
# Analyze a specific stock
python -m src.cli analyze --symbol AAPL --days 7

# Generate investment report
python -m src.cli report --symbol TSLA --output pdf

# Monitor news in real-time
python -m src.cli monitor --symbols AAPL,GOOGL,MSFT

# Search historical events
python -m src.cli search --query "interest rate hikes" --limit 10
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Streamlit Dashboard                    │
│                    (User Interface)                      │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                    FastAPI Server                        │
│                 (REST API Layer)                         │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│              LangChain Multi-Agent System                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Research   │  │  Sentiment   │  │     Risk     │  │
│  │    Agent     │  │    Agent     │  │    Agent     │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                  │                  │          │
│         └──────────────────┼──────────────────┘          │
│                            │                             │
│                    ┌───────▼────────┐                    │
│                    │  Summary Agent │                    │
│                    └────────────────┘                    │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│              Data Layer & Vector Store                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
│  │ NewsAPI  │  │  Alpha   │  │  World   │  │ ChromaDB│ │
│  │          │  │ Vantage  │  │   Bank   │  │ (Vector)│ │
│  └──────────┘  └──────────┘  └──────────┘  └─────────┘ │
└─────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
financial-news-analyzer/
│
├── src/
│   ├── agents/                 # Multi-agent implementations
│   │   ├── __init__.py
│   │   ├── research_agent.py
│   │   ├── sentiment_agent.py
│   │   ├── risk_agent.py
│   │   └── summary_agent.py
│   │
│   ├── chains/                 # LangChain chains
│   │   ├── __init__.py
│   │   ├── analysis_chain.py
│   │   └── report_chain.py
│   │
│   ├── tools/                  # Custom LangChain tools
│   │   ├── __init__.py
│   │   ├── news_tool.py
│   │   ├── stock_tool.py
│   │   └── economic_tool.py
│   │
│   ├── utils/                  # Utility functions
│   │   ├── __init__.py
│   │   ├── api_clients.py
│   │   ├── vector_store.py
│   │   ├── logger.py
│   │   └── init_db.py
│   │
│   ├── api/                    # FastAPI application
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── routes/
│   │
│   └── cli.py                  # Command-line interface
│
├── streamlit_app/              # Streamlit dashboard
│   ├── app.py
│   ├── pages/
│   │   ├── 1_News_Monitor.py
│   │   ├── 2_Sentiment_Analysis.py
│   │   ├── 3_Historical_Trends.py
│   │   └── 4_Report_Generator.py
│   └── components/
│
├── data/
│   ├── raw/                    # Raw data cache
│   └── processed/              # Processed datasets
│
├── notebooks/                  # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_agent_development.ipynb
│   └── 03_evaluation.ipynb
│
├── tests/                      # Test suite
│   ├── test_agents.py
│   ├── test_chains.py
│   └── test_api.py
│
├── docs/                       # Documentation
│   ├── architecture.md
│   ├── api_reference.md
│   └── deployment.md
│
├── config/                     # Configuration files
│   ├── agents_config.yaml
│   └── logging_config.yaml
│
├── .env.example                # Environment variables template
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose setup
└── README.md                   # This file
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# LLM Provider (choose one)
OPENAI_API_KEY=your_openai_key
# OR
OLLAMA_BASE_URL=http://localhost:11434

# Data Sources
NEWSAPI_KEY=your_newsapi_key
ALPHA_VANTAGE_KEY=your_alpha_vantage_key
WORLD_BANK_API_KEY=optional

# Vector Database
CHROMA_DB_PATH=./data/chroma_db

# Application Settings
LOG_LEVEL=INFO
MAX_NEWS_ARTICLES=50
CACHE_EXPIRY_HOURS=24
```

### Agent Configuration

Customize agent behavior in `config/agents_config.yaml`:

```yaml
research_agent:
  max_iterations: 5
  sources: [newsapi, yahoo_finance]
  time_range_days: 7

sentiment_agent:
  model: gpt-4
  temperature: 0.3
  include_reasoning: true

risk_agent:
  alert_threshold: 0.7
  factors: [volatility, news_sentiment, volume]
```

## 📊 Sample Output

### Sentiment Analysis Report

```json
{
  "symbol": "AAPL",
  "analysis_date": "2024-03-31",
  "sentiment_score": 0.72,
  "sentiment_label": "Positive",
  "key_insights": [
    "Strong Q4 earnings beat expectations",
    "New product launches receiving positive reception",
    "Increased institutional buying activity"
  ],
  "risk_factors": [
    "Regulatory concerns in EU markets",
    "Supply chain pressures in Asia"
  ],
  "recommendation": "HOLD with positive outlook",
  "confidence": 0.85
}
```

### Historical Trend Visualization

The dashboard includes interactive charts showing:
- Sentiment trends over time
- Correlation between news sentiment and stock price
- Risk factor frequency analysis
- Comparative analysis across multiple stocks

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test module
pytest tests/test_agents.py -v
```

## 🐳 Docker Deployment

Build and run with Docker:

```bash
# Build image
docker build -t financial-analyzer .

# Run container
docker run -p 8501:8501 -p 8000:8000 --env-file .env financial-analyzer
```

Or use Docker Compose:

```bash
docker-compose up -d
```

## 📈 Performance Metrics

- **API Response Time**: < 2 seconds for news analysis
- **Vector Search**: < 100ms for semantic queries
- **Agent Decision Time**: 3-5 seconds for complete analysis
- **Dashboard Load Time**: < 1 second
- **Data Freshness**: Real-time to 15-minute delay

## 🛠️ Development

### Adding New Agents

1. Create agent class in `src/agents/`
2. Inherit from `BaseAgent`
3. Implement `execute()` method
4. Register in agent orchestrator

Example:
```python
from src.agents.base import BaseAgent

class CustomAgent(BaseAgent):
    def execute(self, input_data):
        # Your agent logic
        return result
```

### Adding New Data Sources

1. Create tool in `src/tools/`
2. Implement `_run()` method
3. Add to tool registry
4. Update configuration

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- LangChain team for the amazing framework
- NewsAPI for financial news data
- Alpha Vantage for market data
- ChromaDB for vector storage
- Streamlit for the dashboard framework

## 📧 Contact

Your Name - your.email@example.com

Project Link: [https://github.com/yourusername/financial-news-analyzer](https://github.com/yourusername/financial-news-analyzer)

## 🗺️ Roadmap

- [ ] Add support for cryptocurrency analysis
- [ ] Implement multi-language news support
- [ ] Add technical indicator integration
- [ ] Build mobile app interface
- [ ] Add backtesting capabilities
- [ ] Integrate with brokerage APIs
- [ ] Add email/SMS alert system
- [ ] Implement portfolio optimization

## 📚 Documentation

For detailed documentation, visit:
- [Architecture Overview](docs/architecture.md)
- [API Reference](docs/api_reference.md)
- [Deployment Guide](docs/deployment.md)
- [Tutorial Notebooks](notebooks/)

---

**⭐ If you find this project helpful, please consider giving it a star!**
