"""
RAG evaluation metrics for Week 2 optimization.
Senior Engineer Principle: Measure what matters.
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
    retrieval_precision: float
    answer_relevance: float
    hallucination_score: float
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
            "retrieval_precision": 0.7,
            "answer_relevance": 0.6,
            "hallucination_score": 0.3,
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
            retrieval_precision=self._calculate_retrieval_precision(question, relevant_chunks),
            answer_relevance=self._calculate_answer_relevance(question, answer),
            hallucination_score=self._detect_hallucinations(answer, relevant_chunks),
            response_time=response_time,
            chunks_retrieved=len(relevant_chunks),
            timestamp=datetime.now()
        )
        
        self.metrics_history.append(metrics)
        self._log_metrics(metrics)
        
        return metrics
    
    def _calculate_retrieval_precision(self, question: str, relevant_chunks: List) -> float:
        """Calculate how relevant retrieved chunks are to the question."""
        if not relevant_chunks:
            return 0.0
        
        question_terms = set(question.lower().split())
        relevant_count = 0
        
        for chunk, score, metadata in relevant_chunks:
            chunk_terms = set(chunk.lower().split())
            
            # Count chunks that share significant terms with question
            common_terms = question_terms.intersection(chunk_terms)
            if len(common_terms) >= max(1, len(question_terms) * 0.3):  # At least 30% overlap
                relevant_count += 1
        
        return relevant_count / len(relevant_chunks)
    
    def _calculate_answer_relevance(self, question: str, answer: str) -> float:
        """Calculate how relevant the answer is to the question."""
        if not question or not answer:
            return 0.0
        
        question_terms = set(question.lower().split())
        answer_terms = set(answer.lower().split())
        
        if not question_terms:
            return 0.0
        
        # Calculate term overlap
        overlap = len(question_terms.intersection(answer_terms))
        return overlap / len(question_terms)
    
    def _detect_hallucinations(self, answer: str, context_chunks: List) -> float:
        """
        Simple hallucination detection based on context grounding.
        Returns: 0.0 = no hallucination, 1.0 = complete hallucination
        """
        if not answer or not context_chunks:
            return 1.0
        
        answer_lower = answer.lower()
        context_text = " ".join([chunk[0].lower() for chunk in context_chunks])
        
        # Split answer into meaningful sentences
        sentences = re.split(r'[.!?]+', answer_lower)
        meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if not meaningful_sentences:
            return 1.0
        
        grounded_sentences = 0
        for sentence in meaningful_sentences:
            # Check if key phrases from sentence appear in context
            words = sentence.split()
            if len(words) > 3:
                # Check for key phrase matches
                key_phrases = [' '.join(words[i:i+3]) for i in range(0, len(words)-2)]
                matches = sum(1 for phrase in key_phrases if phrase in context_text)
                
                if matches / len(key_phrases) > 0.4:  # 40% of key phrases matched
                    grounded_sentences += 1
        
        hallucination_rate = 1.0 - (grounded_sentences / len(meaningful_sentences))
        return max(0.0, min(1.0, hallucination_rate))
    
    def _log_metrics(self, metrics: QueryMetrics):
        """Log metrics for monitoring."""
        logger.info(
            f"📊 Query Evaluation - "
            f"Precision: {metrics.retrieval_precision:.3f}, "
            f"Relevance: {metrics.answer_relevance:.3f}, "
            f"Hallucination: {metrics.hallucination_score:.3f}, "
            f"Time: {metrics.response_time:.2f}s"
        )
    
    def get_aggregate_metrics(self, hours: Optional[int] = None) -> Dict[str, Any]:
        """Get aggregate metrics for a time period."""
        if not self.metrics_history:
            return {}
        
        # Filter by time if specified
        if hours:
            cutoff = datetime.now() - timedelta(hours=hours)
            metrics_list = [m for m in self.metrics_history if m.timestamp > cutoff]
        else:
            metrics_list = self.metrics_history
        
        if not metrics_list:
            return {}
        
        return {
            "total_queries": len(metrics_list),
            "avg_retrieval_precision": np.mean([m.retrieval_precision for m in metrics_list]),
            "avg_answer_relevance": np.mean([m.answer_relevance for m in metrics_list]),
            "avg_hallucination_score": np.mean([m.hallucination_score for m in metrics_list]),
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
        
        if recent_metrics["avg_retrieval_precision"] < thresholds["retrieval_precision"]:
            alerts.append({
                "type": "RETRIEVAL_PRECISION_LOW",
                "metric": recent_metrics["avg_retrieval_precision"],
                "threshold": thresholds["retrieval_precision"],
                "message": "Retrieval precision below threshold - consider adjusting chunking or embedding model"
            })
        
        if recent_metrics["avg_hallucination_score"] > thresholds["hallucination_score"]:
            alerts.append({
                "type": "HALLUCINATION_HIGH",
                "metric": recent_metrics["avg_hallucination_score"],
                "threshold": thresholds["hallucination_score"],
                "message": "High hallucination rate detected - consider improving context or prompt engineering"
            })
        
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