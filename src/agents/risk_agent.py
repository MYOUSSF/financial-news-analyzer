"""
Risk Agent - Identifies and analyzes potential risks in financial markets.
"""
from typing import Any, Dict, List
from datetime import datetime
from loguru import logger

from langchain.prompts import PromptTemplate

from .base import BaseAgent


class RiskAgent(BaseAgent):
    """
    Agent responsible for identifying and analyzing financial risks.
    
    Analyzes news, market data, and sentiment to detect potential risk factors.
    """
    
    # Risk categories and their indicators
    RISK_CATEGORIES = {
        "volatility": {
            "indicators": ["price_swings", "volume_spikes", "beta"],
            "threshold": 0.7
        },
        "regulatory": {
            "indicators": ["investigations", "lawsuits", "compliance_issues"],
            "threshold": 0.6
        },
        "financial": {
            "indicators": ["debt_levels", "cash_flow", "profitability"],
            "threshold": 0.65
        },
        "market": {
            "indicators": ["sector_decline", "competitor_moves", "market_share"],
            "threshold": 0.6
        },
        "operational": {
            "indicators": ["supply_chain", "management_changes", "strikes"],
            "threshold": 0.65
        }
    }
    
    def __init__(self, llm: Any, tools: List[Any] = None, verbose: bool = False):
        """
        Initialize the Risk Agent.
        
        Args:
            llm: Language model to use
            tools: Optional list of tools
            verbose: Enable verbose logging
        """
        super().__init__(
            name="RiskAgent",
            description="Identifies and analyzes financial risks",
            llm=llm,
            tools=tools or [],
            verbose=verbose
        )
        
        # Risk analysis prompt
        self.risk_prompt = PromptTemplate(
            input_variables=["symbol", "news_data", "market_data", "sentiment"],
            template="""
            You are a financial risk analyst. Analyze the following data for {symbol} 
            to identify potential risks and concerns.
            
            Recent News:
            {news_data}
            
            Market Data:
            {market_data}
            
            Sentiment Analysis:
            {sentiment}
            
            Identify and categorize risks into:
            1. Regulatory risks (legal issues, compliance, investigations)
            2. Financial risks (debt, cash flow, profitability concerns)
            3. Market risks (competition, market share, sector trends)
            4. Operational risks (supply chain, management, operations)
            5. Volatility risks (price swings, trading volume anomalies)
            
            For each risk:
            - Describe the risk clearly
            - Assess severity (LOW, MEDIUM, HIGH, CRITICAL)
            - Estimate likelihood (0-1)
            - Suggest mitigation or monitoring strategies
            
            Provide a risk score (0-1) where 1 is highest risk.
            """
        )
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute risk analysis on provided data.
        
        Args:
            input_data: Dictionary containing:
                - symbol: Stock symbol
                - news_data: Recent news findings
                - market_data: Market performance data
                - sentiment: Sentiment analysis results
        
        Returns:
            Dictionary containing risk analysis results
        """
        try:
            symbol = input_data.get("symbol", "")
            news_data = input_data.get("news_data", "No news data available")
            market_data = input_data.get("market_data", "No market data available")
            sentiment = input_data.get("sentiment", {})
            
            logger.info(f"Analyzing risks for {symbol}")
            
            # Format prompt
            prompt = self.risk_prompt.format(
                symbol=symbol,
                news_data=self._format_news(news_data),
                market_data=self._format_market_data(market_data),
                sentiment=self._format_sentiment(sentiment)
            )
            
            # Get LLM risk analysis
            llm_analysis = self.llm.predict(prompt)
            
            # Extract structured risks
            risks = self._parse_risks(llm_analysis)
            
            # Calculate overall risk score
            overall_risk = self._calculate_risk_score(risks, sentiment)
            
            # Generate alerts if needed
            alerts = self._generate_alerts(risks, overall_risk)
            
            output = {
                "symbol": symbol,
                "analysis_date": datetime.now().isoformat(),
                "overall_risk_score": overall_risk["score"],
                "risk_level": overall_risk["level"],
                "identified_risks": risks,
                "alerts": alerts,
                "recommendations": self._generate_recommendations(risks, overall_risk),
                "llm_analysis": llm_analysis,
                "metadata": {
                    "agent": self.name,
                    "risk_categories_checked": list(self.RISK_CATEGORIES.keys())
                }
            }
            
            self._log_execution(input_data, output)
            
            logger.info(f"Risk analysis complete for {symbol}: "
                       f"{overall_risk['level']} ({overall_risk['score']:.2f})")
            
            return output
            
        except Exception as e:
            logger.error(f"Error in RiskAgent execution: {str(e)}")
            return {
                "symbol": input_data.get("symbol", ""),
                "error": str(e),
                "status": "failed"
            }
    
    def _format_news(self, news_data: Any) -> str:
        """Format news data for prompt."""
        if isinstance(news_data, str):
            return news_data
        elif isinstance(news_data, dict):
            findings = news_data.get("findings", "")
            return findings[:1000] if findings else "No news data"
        return "No news data available"
    
    def _format_market_data(self, market_data: Any) -> str:
        """Format market data for prompt."""
        if isinstance(market_data, str):
            return market_data
        elif isinstance(market_data, dict):
            return f"""
            Price Change: {market_data.get('price_change', 'N/A')}
            Volume: {market_data.get('volume', 'N/A')}
            Volatility: {market_data.get('volatility', 'N/A')}
            """
        return "No market data available"
    
    def _format_sentiment(self, sentiment: Dict[str, Any]) -> str:
        """Format sentiment data for prompt."""
        if not sentiment:
            return "No sentiment data available"
        
        return f"""
        Overall Sentiment: {sentiment.get('overall_sentiment', 'N/A')}
        Score: {sentiment.get('sentiment_score', 'N/A')}
        Confidence: {sentiment.get('confidence', 'N/A')}
        """
    
    def _parse_risks(self, llm_analysis: str) -> List[Dict[str, Any]]:
        """
        Parse risks from LLM analysis.
        
        This is a simplified parser. In production, use more sophisticated NLP.
        """
        risks = []
        
        # Define risk keywords for each category
        risk_keywords = {
            "regulatory": ["investigation", "lawsuit", "compliance", "regulation", "fine"],
            "financial": ["debt", "loss", "bankruptcy", "cash flow", "revenue decline"],
            "market": ["competition", "market share", "downturn", "sector decline"],
            "operational": ["supply chain", "management", "strike", "disruption"],
            "volatility": ["volatile", "swing", "fluctuation", "unstable"]
        }
        
        analysis_lower = llm_analysis.lower()
        
        for category, keywords in risk_keywords.items():
            for keyword in keywords:
                if keyword in analysis_lower:
                    # Found a potential risk
                    risks.append({
                        "category": category,
                        "description": f"Potential {category} risk: {keyword} detected",
                        "severity": "MEDIUM",  # Default severity
                        "likelihood": 0.6,
                        "detected_keyword": keyword
                    })
                    break  # Only add one risk per category
        
        return risks
    
    def _calculate_risk_score(
        self, 
        risks: List[Dict[str, Any]], 
        sentiment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate overall risk score.
        
        Args:
            risks: List of identified risks
            sentiment: Sentiment analysis results
        
        Returns:
            Dictionary with risk score and level
        """
        if not risks:
            base_score = 0.2  # Baseline risk
        else:
            # Calculate weighted risk score
            severity_weights = {"LOW": 0.25, "MEDIUM": 0.5, "HIGH": 0.75, "CRITICAL": 1.0}
            
            risk_scores = []
            for risk in risks:
                severity = risk.get("severity", "MEDIUM")
                likelihood = risk.get("likelihood", 0.5)
                risk_scores.append(severity_weights[severity] * likelihood)
            
            base_score = sum(risk_scores) / len(risk_scores) if risk_scores else 0.3
        
        # Adjust for sentiment
        sentiment_score = sentiment.get("sentiment_score", 0.5)
        if sentiment_score < 0.4:  # Negative sentiment increases risk
            base_score *= 1.2
        elif sentiment_score > 0.6:  # Positive sentiment decreases risk
            base_score *= 0.9
        
        # Cap at 1.0
        final_score = min(base_score, 1.0)
        
        # Determine risk level
        if final_score >= 0.75:
            level = "CRITICAL"
        elif final_score >= 0.6:
            level = "HIGH"
        elif final_score >= 0.4:
            level = "MEDIUM"
        else:
            level = "LOW"
        
        return {
            "score": final_score,
            "level": level,
            "risk_count": len(risks)
        }
    
    def _generate_alerts(
        self, 
        risks: List[Dict[str, Any]], 
        overall_risk: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate alerts for high-priority risks.
        
        Args:
            risks: List of identified risks
            overall_risk: Overall risk assessment
        
        Returns:
            List of alerts
        """
        alerts = []
        
        # Alert on overall high risk
        if overall_risk["level"] in ["HIGH", "CRITICAL"]:
            alerts.append({
                "type": "OVERALL_RISK",
                "severity": overall_risk["level"],
                "message": f"Overall risk level is {overall_risk['level']} "
                          f"(score: {overall_risk['score']:.2f})",
                "timestamp": datetime.now().isoformat()
            })
        
        # Alert on specific high-severity risks
        for risk in risks:
            if risk.get("severity") in ["HIGH", "CRITICAL"]:
                alerts.append({
                    "type": "SPECIFIC_RISK",
                    "category": risk["category"],
                    "severity": risk["severity"],
                    "message": risk["description"],
                    "timestamp": datetime.now().isoformat()
                })
        
        return alerts
    
    def _generate_recommendations(
        self, 
        risks: List[Dict[str, Any]], 
        overall_risk: Dict[str, Any]
    ) -> List[str]:
        """
        Generate risk mitigation recommendations.
        
        Args:
            risks: List of identified risks
            overall_risk: Overall risk assessment
        
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if overall_risk["level"] in ["HIGH", "CRITICAL"]:
            recommendations.append("Consider reducing position size or exiting position")
            recommendations.append("Implement stop-loss orders to limit downside")
        
        # Category-specific recommendations
        risk_categories = {r["category"] for r in risks}
        
        if "regulatory" in risk_categories:
            recommendations.append("Monitor regulatory developments closely")
        
        if "financial" in risk_categories:
            recommendations.append("Review financial statements and debt levels")
        
        if "volatility" in risk_categories:
            recommendations.append("Consider hedging strategies or options")
        
        if not recommendations:
            recommendations.append("Continue monitoring with standard risk management")
        
        return recommendations
    
    def monitor_realtime(
        self, 
        symbol: str, 
        alert_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Set up real-time risk monitoring for a symbol.
        
        Args:
            symbol: Stock symbol to monitor
            alert_threshold: Risk score threshold for alerts
        
        Returns:
            Monitoring configuration
        """
        return {
            "symbol": symbol,
            "monitoring": True,
            "alert_threshold": alert_threshold,
            "check_interval": "15m",
            "notification_channels": ["dashboard", "email"]
        }
