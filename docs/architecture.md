# Architecture Overview

## System Architecture

The Financial News Analyzer is built using a modular, microservices-inspired architecture with clear separation of concerns.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Presentation Layer                      │
│  ┌──────────────────────┐    ┌──────────────────────────┐  │
│  │  Streamlit Dashboard │    │   REST API (FastAPI)     │  │
│  │   (User Interface)   │    │  (Programmatic Access)   │  │
│  └──────────┬───────────┘    └──────────┬───────────────┘  │
└─────────────┼──────────────────────────┼───────────────────┘
              │                           │
              └───────────┬───────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    Application Layer                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │        LangChain Multi-Agent Orchestration           │  │
│  │                                                        │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐           │  │
│  │  │ Research │  │Sentiment │  │   Risk   │           │  │
│  │  │  Agent   │  │  Agent   │  │  Agent   │           │  │
│  │  └─────┬────┘  └─────┬────┘  └─────┬────┘           │  │
│  │        │             │             │                  │  │
│  │        └─────────────┼─────────────┘                  │  │
│  │                      │                                 │  │
│  │              ┌───────▼────────┐                       │  │
│  │              │ Summary Agent  │                       │  │
│  │              └────────────────┘                       │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                      Tools Layer                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  News    │  │  Stock   │  │ Economic │  │ Semantic │   │
│  │  Tool    │  │  Tool    │  │   Tool   │  │  Search  │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                      Data Layer                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ NewsAPI  │  │  Alpha   │  │  World   │  │ ChromaDB │   │
│  │          │  │ Vantage  │  │   Bank   │  │ (Vector) │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Presentation Layer

#### Streamlit Dashboard
- **Purpose**: Interactive web interface for end users
- **Features**:
  - Real-time news monitoring
  - Sentiment visualization
  - Risk assessment dashboard
  - Report generation
- **Technology**: Streamlit, Plotly, Pandas
- **Location**: `streamlit_app/`

#### REST API
- **Purpose**: Programmatic access to analysis capabilities
- **Features**:
  - RESTful endpoints
  - JSON request/response
  - OpenAPI documentation
  - Rate limiting
- **Technology**: FastAPI, Pydantic, Uvicorn
- **Location**: `src/api/`

### 2. Application Layer - Multi-Agent System

The core intelligence of the system is implemented as a multi-agent architecture using LangChain.

#### Research Agent
**Responsibility**: Information gathering and preliminary analysis

**Capabilities**:
- Fetch news from multiple sources
- Retrieve stock market data
- Identify key events and announcements
- Aggregate information from various APIs

**Implementation**:
- Uses LangChain's Agent framework
- Equipped with custom tools (news, stock data)
- Iterative reasoning with REACT pattern
- Maintains conversation memory

**File**: `src/agents/research_agent.py`

#### Sentiment Agent
**Responsibility**: Analyze sentiment from text and news

**Capabilities**:
- Multi-model sentiment analysis (transformer + LLM)
- Historical sentiment tracking
- Confidence scoring
- Insight extraction

**Models Used**:
- DistilBERT for fast sentiment classification
- GPT-4 for reasoning and context
- Custom aggregation logic

**File**: `src/agents/sentiment_agent.py`

#### Risk Agent
**Responsibility**: Identify and assess financial risks

**Capabilities**:
- Multi-category risk detection
  - Regulatory risks
  - Financial risks
  - Market risks
  - Operational risks
  - Volatility risks
- Risk scoring and severity assessment
- Alert generation
- Recommendation engine

**File**: `src/agents/risk_agent.py`

#### Summary Agent
**Responsibility**: Synthesize findings into actionable reports

**Capabilities**:
- Aggregate multi-agent outputs
- Generate coherent narratives
- Provide investment recommendations
- Create executive summaries

**File**: `src/agents/summary_agent.py`

### 3. Tools Layer

Custom LangChain tools that agents use to interact with external services.

#### News Tool
- **Data Source**: NewsAPI, RSS feeds
- **Functionality**: Search and retrieve financial news
- **Caching**: Implements TTL-based caching
- **File**: `src/tools/news_tool.py`

#### Stock Tool
- **Data Sources**: Alpha Vantage, Yahoo Finance
- **Functionality**: Stock prices, volume, technical indicators
- **File**: `src/tools/stock_tool.py`

#### Economic Tool
- **Data Source**: World Bank API
- **Functionality**: Economic indicators (GDP, inflation, etc.)
- **File**: `src/tools/economic_tool.py`

#### Semantic Search Tool
- **Backend**: ChromaDB vector store
- **Functionality**: Search historical news by semantic similarity
- **Embedding Model**: sentence-transformers
- **File**: `src/tools/search_tool.py`

### 4. Data Layer

#### External APIs
- **NewsAPI**: Real-time news aggregation
- **Alpha Vantage**: Stock market data
- **Yahoo Finance**: Historical prices
- **World Bank**: Economic indicators

#### Vector Database
- **Technology**: ChromaDB
- **Purpose**: Store embeddings of historical news
- **Features**:
  - Persistent storage
  - Fast similarity search
  - Metadata filtering
- **Location**: `data/chroma_db/`

#### Cache Layer
- **Technology**: DiskCache / Redis
- **Purpose**: Reduce API calls and improve response times
- **TTL**: Configurable per data source
- **Location**: `data/cache/`

## Data Flow

### Analysis Request Flow

```
1. User Request
   ↓
2. API/Dashboard receives request
   ↓
3. Request routed to Analysis Chain
   ↓
4. Research Agent activated
   ├─ Calls News Tool
   ├─ Calls Stock Tool
   └─ Returns findings
   ↓
5. Sentiment Agent processes findings
   ├─ Runs ML model
   ├─ Calls LLM for reasoning
   └─ Returns sentiment analysis
   ↓
6. Risk Agent analyzes data
   ├─ Identifies risk factors
   ├─ Calculates risk scores
   └─ Generates alerts
   ↓
7. Summary Agent synthesizes
   ├─ Combines all findings
   ├─ Generates recommendations
   └─ Creates report
   ↓
8. Response returned to user
```

## Key Design Patterns

### 1. Agent Pattern
Each agent is an independent, specialized component with:
- Single responsibility
- Clear interface (execute method)
- Tool access
- Memory management

### 2. Chain of Responsibility
Agents process data sequentially, with each adding value:
- Research → Sentiment → Risk → Summary
- Each agent can operate independently
- Results passed through pipeline

### 3. Observer Pattern
For real-time monitoring:
- Agents emit events
- Listeners can subscribe to specific events
- Used for alerts and notifications

### 4. Strategy Pattern
Different analysis strategies can be selected:
- Quick analysis vs. deep research
- Different risk models
- Customizable report formats

## Scalability Considerations

### Horizontal Scaling
- **API Layer**: Multiple FastAPI instances behind load balancer
- **Agents**: Can be deployed as separate microservices
- **Database**: Vector store can be sharded

### Caching Strategy
- **L1**: In-memory cache (LRU)
- **L2**: Redis distributed cache
- **L3**: Persistent disk cache

### Rate Limiting
- Per-user rate limits on API
- Respect API limits of data sources
- Queue system for batch processing

## Security

### Authentication
- API key-based authentication
- JWT tokens for session management
- Role-based access control (RBAC)

### Data Protection
- API keys stored in environment variables
- Secrets managed with proper encryption
- No sensitive data in logs

### Input Validation
- Pydantic models for request validation
- SQL injection prevention
- XSS protection in dashboard

## Monitoring & Observability

### Logging
- **Library**: Loguru
- **Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Destinations**: File, stdout, external logging service

### Metrics
- API response times
- Agent execution times
- Cache hit rates
- Error rates
- Data source availability

### Health Checks
- `/health` endpoint
- Service dependency checks
- Database connectivity
- API availability

## Deployment Options

### 1. Docker Container
- Single container with all services
- Suitable for development and small deployments

### 2. Docker Compose
- Multi-container setup
- Separate containers for API, Dashboard, Redis
- Suitable for production

### 3. Kubernetes
- Full orchestration
- Auto-scaling
- High availability
- Suitable for large-scale production

## Future Enhancements

### Planned Features
1. **Real-time WebSocket streaming**
2. **Multi-user collaboration**
3. **Custom alert rules**
4. **Portfolio management**
5. **Backtesting engine**
6. **Mobile app**
7. **Email/SMS notifications**
8. **Integration with brokerage APIs**

### Technical Debt
- Add comprehensive integration tests
- Implement circuit breakers for external APIs
- Add request queuing for rate-limited APIs
- Implement data validation layer
- Add performance profiling

## Dependencies

### Core
- Python 3.9+
- LangChain 0.1.20+
- OpenAI API / Anthropic API
- ChromaDB 0.4.24+

### Infrastructure
- Docker 20+
- Redis 7+ (optional)
- PostgreSQL 15+ (optional)

### Monitoring
- Prometheus (metrics)
- Grafana (visualization)
- Sentry (error tracking)
