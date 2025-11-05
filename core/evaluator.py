"""
RAG evaluation metrics for Week 2 optimization.
FIXED: Removed ALL unreliable keyword-based metrics (precision, hallucination, relevance).
Kept only objective, trustworthy metrics: Response Time and Chunks Retrieved.
"""
import logging
import re
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from core.config import config

logger = logging.getLogger(__name__)

@dataclass
class QueryMetrics:
    """Structured metrics for a single query."""
    question: str
    response_time: float
    chunks_retrieved: int
    timestamp: datetime

class RAGEvaluator:
    """
    Production-ready RAG evaluator for monitoring and optimization.
    Senior Engineer Principle: Collect metrics that drive improvements.
    """
    
    def __init__(self):
        self.metrics_history: List[QueryMetrics] = []
        self.performance_thresholds = {
            "response_time": 10.0  # seconds
        }
    
    def evaluate_query(self, question: str, answer: str, 
                      relevant_chunks: List, response_time: float) -> QueryMetrics:
        """
        Comprehensive query evaluation.
        Returns structured metrics for analysis.
        """
        metrics = QueryMetrics(
            question=question,
            response_time=response_time,
            chunks_retrieved=len(relevant_chunks),
            timestamp=datetime.now()
        )
        
        self.metrics_history.append(metrics)
        self._log_metrics(metrics)
        
        return metrics
    
    # -----------------------------------------------------------------
    # 🚨 REMOVED: All unreliable keyword-matching functions
    # -----------------------------------------------------------------
    
    def _log_metrics(self, metrics: QueryMetrics):
        """Log metrics for monitoring."""
        logger.info(
            f"📊 Query Evaluation - "
            f"Time: {metrics.response_time:.2f}s, "
            f"Chunks: {metrics.chunks_retrieved}"
        )
    
    def get_aggregate_metrics(self, hours: Optional[int] = None) -> Dict[str, Any]:
        """Get aggregate metrics for a time period."""
        if not self.metrics_history:
            return {}
        
        if hours:
            cutoff = datetime.now() - timedelta(hours=hours)
            metrics_list = [m for m in self.metrics_history if m.timestamp > cutoff]
        else:
            metrics_list = self.metrics_history
        
        if not metrics_list:
            return {}
        
        # -----------------------------------------------------------------
        # 🚨 MODIFIED: Only return reliable metrics
        # -----------------------------------------------------------------
        return {
            "total_queries": len(metrics_list),
            "avg_response_time": np.mean([m.response_time for m in metrics_list]),
            "avg_chunks_retrieved": np.mean([m.chunks_retrieved for m in metrics_list]),
            "time_period_hours": hours if hours else "all"
        }
    
    def get_performance_alerts(self) -> List[Dict[str, Any]]:
        """Get alerts for performance issues."""
        alerts = []
        recent_metrics = self.get_aggregate_metrics(hours=24)
        
        if not recent_metrics:
            return alerts
        
        thresholds = self.performance_thresholds
        
        # -----------------------------------------------------------------
        # 🚨 MODIFIED: Only alert on reliable metrics
        # -----------------------------------------------------------------
        if recent_metrics["avg_response_time"] > thresholds["response_time"]:
            alerts.append({
                "type": "RESPONSE_TIME_HIGH",
                "metric": recent_metrics["avg_response_time"],
                "threshold": thresholds["response_time"],
                "message": "Response time above threshold - consider optimizing pipeline"
            })
        
        return alerts
    
    def clear_history(self):
        """Clear metrics history."""
        self.metrics_history.clear()
        logger.info("Metrics history cleared")

# Global evaluator instance
evaluator = RAGEvaluator()