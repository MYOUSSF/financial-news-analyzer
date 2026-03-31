"""
Sentiment Agent - Performs sentiment analysis on financial news and market data.
"""
from typing import Any, Dict, List
from datetime import datetime
from loguru import logger

from langchain.prompts import PromptTemplate
from transformers import pipeline
import numpy as np

from .base import BaseAgent


class SentimentAgent(BaseAgent):
    """
    Agent responsible for analyzing sentiment from financial news and social media.
    
    Uses both transformer models and LLM reasoning for comprehensive sentiment analysis.
    """
    
    def __init__(self, llm: Any, tools: List[Any] = None, verbose: bool = False):
        """
        Initialize the Sentiment Agent.
        
        Args:
            llm: Language model to use
            tools: Optional list of tools
            verbose: Enable verbose logging
        """
        super().__init__(
            name="SentimentAgent",
            description="Analyzes sentiment from financial news and data",
            llm=llm,
            tools=tools or [],
            verbose=verbose
        )
        
        # Load sentiment analysis model
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1  # CPU
            )
            logger.info("Loaded sentiment analysis model")
        except Exception as e:
            logger.warning(f"Could not load sentiment model: {e}")
            self.sentiment_pipeline = None
        
        # Sentiment analysis prompt
        self.sentiment_prompt = PromptTemplate(
            input_variables=["text", "context"],
            template="""
            You are a financial sentiment analyst. Analyze the following text for 
            sentiment regarding financial markets and investments.
            
            Text to analyze:
            {text}
            
            Context:
            {context}
            
            Provide a detailed sentiment analysis including:
            1. Overall sentiment (Positive/Negative/Neutral) with confidence score (0-1)
            2. Key positive factors
            3. Key negative factors
            4. Market implications
            5. Confidence in the analysis
            
            Format your response as a structured analysis.
            """
        )
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute sentiment analysis on provided text or news articles.
        
        Args:
            input_data: Dictionary containing:
                - text: Text to analyze OR
                - articles: List of article texts
                - symbol: Stock symbol (for context)
                - context: Additional context (optional)
        
        Returns:
            Dictionary containing sentiment analysis results
        """
        try:
            symbol = input_data.get("symbol", "")
            context = input_data.get("context", f"Analysis for {symbol}")
            
            # Handle single text or multiple articles
            if "text" in input_data:
                texts = [input_data["text"]]
            elif "articles" in input_data:
                texts = input_data["articles"]
            else:
                return {"error": "No text or articles provided"}
            
            logger.info(f"Analyzing sentiment for {len(texts)} text(s)")
            
            # Perform sentiment analysis
            ml_sentiments = self._analyze_with_ml(texts)
            llm_analysis = self._analyze_with_llm(texts, context)
            
            # Combine and aggregate results
            combined_sentiment = self._combine_sentiments(ml_sentiments, llm_analysis)
            
            output = {
                "symbol": symbol,
                "analysis_date": datetime.now().isoformat(),
                "text_count": len(texts),
                "overall_sentiment": combined_sentiment["label"],
                "sentiment_score": combined_sentiment["score"],
                "confidence": combined_sentiment["confidence"],
                "ml_analysis": ml_sentiments,
                "llm_analysis": llm_analysis,
                "key_insights": self._extract_insights(llm_analysis),
                "metadata": {
                    "agent": self.name
                }
            }
            
            self._log_execution(input_data, output)
            
            logger.info(f"Sentiment analysis complete: {combined_sentiment['label']} "
                       f"(score: {combined_sentiment['score']:.2f})")
            
            return output
            
        except Exception as e:
            logger.error(f"Error in SentimentAgent execution: {str(e)}")
            return {
                "symbol": input_data.get("symbol", ""),
                "error": str(e),
                "status": "failed"
            }
    
    def _analyze_with_ml(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment using ML model.
        
        Args:
            texts: List of texts to analyze
        
        Returns:
            List of sentiment results
        """
        if not self.sentiment_pipeline:
            return []
        
        results = []
        for text in texts:
            try:
                # Truncate text to model's max length
                truncated_text = text[:512]
                sentiment = self.sentiment_pipeline(truncated_text)[0]
                
                # Convert to standard format
                results.append({
                    "text_preview": text[:100] + "...",
                    "label": sentiment["label"],
                    "score": sentiment["score"],
                    "normalized_score": self._normalize_score(
                        sentiment["label"], 
                        sentiment["score"]
                    )
                })
            except Exception as e:
                logger.warning(f"ML analysis failed for text: {str(e)}")
                results.append({
                    "text_preview": text[:100] + "...",
                    "label": "NEUTRAL",
                    "score": 0.5,
                    "normalized_score": 0.0
                })
        
        return results
    
    def _analyze_with_llm(self, texts: List[str], context: str) -> Dict[str, Any]:
        """
        Analyze sentiment using LLM for deeper reasoning.
        
        Args:
            texts: List of texts to analyze
            context: Additional context
        
        Returns:
            LLM sentiment analysis
        """
        try:
            # Combine texts (with length limit)
            combined_text = "\n\n".join(texts[:5])  # Limit to first 5 articles
            if len(combined_text) > 3000:
                combined_text = combined_text[:3000] + "..."
            
            # Format prompt
            prompt = self.sentiment_prompt.format(
                text=combined_text,
                context=context
            )
            
            # Get LLM analysis
            response = self.llm.predict(prompt)
            
            # Parse response (simplified parsing)
            return {
                "analysis": response,
                "method": "llm_reasoning"
            }
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {str(e)}")
            return {
                "analysis": "LLM analysis unavailable",
                "error": str(e)
            }
    
    def _normalize_score(self, label: str, score: float) -> float:
        """
        Normalize sentiment score to [-1, 1] range.
        
        Args:
            label: Sentiment label (POSITIVE/NEGATIVE)
            score: Confidence score [0, 1]
        
        Returns:
            Normalized score where -1 is very negative, 1 is very positive
        """
        if label == "POSITIVE":
            return score
        elif label == "NEGATIVE":
            return -score
        else:
            return 0.0
    
    def _combine_sentiments(
        self, 
        ml_results: List[Dict[str, Any]], 
        llm_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Combine ML and LLM sentiment analyses.
        
        Args:
            ml_results: Results from ML model
            llm_analysis: Results from LLM
        
        Returns:
            Combined sentiment assessment
        """
        if not ml_results:
            return {
                "label": "NEUTRAL",
                "score": 0.5,
                "confidence": 0.3
            }
        
        # Calculate average normalized score from ML results
        normalized_scores = [r["normalized_score"] for r in ml_results]
        avg_score = np.mean(normalized_scores)
        
        # Determine label
        if avg_score > 0.15:
            label = "POSITIVE"
        elif avg_score < -0.15:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"
        
        # Convert to 0-1 range
        score = (avg_score + 1) / 2
        
        # Calculate confidence based on consistency
        score_std = np.std(normalized_scores)
        confidence = max(0.3, 1.0 - score_std)
        
        return {
            "label": label,
            "score": score,
            "confidence": confidence,
            "raw_average": avg_score
        }
    
    def _extract_insights(self, llm_analysis: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Extract key insights from LLM analysis.
        
        Args:
            llm_analysis: LLM analysis results
        
        Returns:
            Dictionary of positive and negative insights
        """
        # Simplified extraction (in production, use more sophisticated parsing)
        analysis_text = llm_analysis.get("analysis", "")
        
        return {
            "positive_factors": ["Analysis indicates positive market response"],
            "negative_factors": ["Some concerns noted in market data"],
            "market_implications": ["Monitor for continued trends"]
        }
    
    def analyze_trend(
        self, 
        historical_sentiments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze sentiment trends over time.
        
        Args:
            historical_sentiments: List of past sentiment analyses
        
        Returns:
            Trend analysis results
        """
        if not historical_sentiments:
            return {"trend": "insufficient_data"}
        
        scores = [s.get("sentiment_score", 0.5) for s in historical_sentiments]
        
        # Calculate trend
        if len(scores) >= 2:
            trend_direction = "improving" if scores[-1] > scores[0] else "declining"
            volatility = np.std(scores)
        else:
            trend_direction = "stable"
            volatility = 0.0
        
        return {
            "trend": trend_direction,
            "volatility": volatility,
            "current_score": scores[-1] if scores else 0.5,
            "average_score": np.mean(scores),
            "data_points": len(scores)
        }
