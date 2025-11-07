"""
RAG evaluation metrics for Week 2 optimization.
FIXED: Removed ALL unreliable keyword-based metrics (precision, hallucination, relevance).
Kept only objective, trustworthy metrics: Response Time and Chunks Retrieved.

UPDATE: Implementing robust, production-safe evaluation metrics:
1.  **Avg. Similarity Score:** (Retrieval Quality) The average similarity
    of all chunks retrieved from the vector store for a query.
2.  **Grounding Confidence:** (Generation Quality) A proxy for hallucination.
    This measures the semantic similarity between the LLM's final answer
    and the context it was given.
    
UPDATE: Added granular latency tracking and a boolean 'hallucination_detected' flag.
"""
import logging
import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from core.config import config

from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)

@dataclass
class QueryMetrics:
    """Structured metrics for a single query."""
    question: str
    response_time: float  # This is the total pipeline time
    chunks_retrieved: int
    timestamp: datetime
    avg_similarity_score: float
    grounding_confidence: float
    routing_time: float
    hyde_time: float
    retrieval_time: float
    generation_time: float
    # -----------------------------------------------------------------
    # 🚨 START: NEW BOOLEAN METRIC
    # -----------------------------------------------------------------
    hallucination_detected: bool  # True if grounding_confidence < threshold
    # -----------------------------------------------------------------

class RAGEvaluator:
    """
    Production-ready RAG evaluator for monitoring and optimization.
    Senior Engineer Principle: Collect metrics that drive improvements.
    """
    
    def __init__(self):
        self.metrics_history: List[QueryMetrics] = []
        self.performance_thresholds = {
            "response_time": 10.0,
            "min_similarity": 0.35,
            "min_grounding": 0.4,    # 🚨 This is the 0.5 (or 0.4) you mentioned
            "max_retrieval_time": 4.0,
            "max_generation_time": 8.0
        }
        
        try:
            self.grounding_model = SentenceTransformer(
                config.rag.EMBEDDING_MODEL,
                device=config.rag.EMBEDDING_DEVICE
            )
            logger.info(f"✅ RAGEvaluator loaded grounding model: {config.rag.EMBEDDING_MODEL}")
        except Exception as e:
            logger.error(f"❌ Failed to load grounding model for evaluator: {e}")
            self.grounding_model = None
    
    def _calculate_avg_similarity(self, relevant_chunks: List[Tuple]) -> float:
        # ... (no changes)
        if not relevant_chunks:
            return 0.0
        try:
            scores = [score for _, score, _ in relevant_chunks]
            return np.mean(scores)
        except Exception as e:
            logger.warning(f"Error calculating avg similarity: {e}")
            return 0.0

    def _calculate_grounding_confidence(self, answer: str, context: str) -> float:
        # ... (no changes)
        if not self.grounding_model or not answer or not context:
            return 0.0
        
        try:
            answer_embedding = self.grounding_model.encode(answer)
            context_embedding = self.grounding_model.encode(context)
            similarity = util.cos_sim(answer_embedding, context_embedding)[0][0]
            return max(0.0, min(1.0, float(similarity)))
        except Exception as e:
            logger.warning(f"Failed to calculate grounding confidence: {e}")
            return 0.0

    def evaluate_query(self, 
                      question: str, 
                      answer: str,
                      context: str,
                      relevant_chunks: List[Tuple], 
                      response_time: float,
                      timings: Dict[str, float]
                      ) -> QueryMetrics:
        """
        Comprehensive query evaluation.
        Returns structured metrics for analysis.
        """
        
        avg_sim = self._calculate_avg_similarity(relevant_chunks)
        grounding_conf = self._calculate_grounding_confidence(answer, context)

        # -----------------------------------------------------------------
        # 🚨 START: CALCULATE NEW BOOLEAN
        # -----------------------------------------------------------------
        # Check if the score is below our defined threshold
        hallucination_bool = grounding_conf < self.performance_thresholds["min_grounding"]
        # -----------------------------------------------------------------

        metrics = QueryMetrics(
            question=question,
            response_time=response_time,
            chunks_retrieved=len(relevant_chunks),
            timestamp=datetime.now(),
            avg_similarity_score=avg_sim,
            grounding_confidence=grounding_conf,
            routing_time=timings.get("routing", 0.0),
            hyde_time=timings.get("hyde", 0.0),
            retrieval_time=timings.get("retrieval", 0.0),
            generation_time=timings.get("generation", 0.0),
            hallucination_detected=hallucination_bool  # 🚨 Pass boolean here
        )
        
        self.metrics_history.append(metrics)
        self._log_metrics(metrics)
        
        return metrics
    
    def _log_metrics(self, metrics: QueryMetrics):
        """Log metrics for monitoring."""
        # 🚨 Updated to log the new boolean
        logger.info(
            f"📊 Query Evaluation - "
            f"Total Time: {metrics.response_time:.2f}s "
            f"[Routing: {metrics.routing_time:.2f}s, "
            f"HyDE: {metrics.hyde_time:.2f}s, "
            f"Retrieval: {metrics.retrieval_time:.2f}s, "
            f"Generation: {metrics.generation_time:.2f}s] | "
            f"Quality - "
            f"Chunks: {metrics.chunks_retrieved}, "
            f"Avg. Sim: {metrics.avg_similarity_score:.2f}, "
            f"Grounding: {metrics.grounding_confidence:.2f}, "
            f"Hallucination Flag: {metrics.hallucination_detected}"
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
        # 🚨 MODIFIED: Add hallucination_rate
        # -----------------------------------------------------------------
        
        # Calculate the percentage of queries that were flagged
        hallucination_rate = np.mean([
            1 if m.hallucination_detected else 0 for m in metrics_list
        ])
        
        return {
            "total_queries": len(metrics_list),
            "avg_response_time": np.mean([m.response_time for m in metrics_list]),
            "avg_chunks_retrieved": np.mean([m.chunks_retrieved for m in metrics_list]),
            "avg_retrieval_similarity": np.mean([m.avg_similarity_score for m in metrics_list]),
            "avg_grounding_confidence": np.mean([m.grounding_confidence for m in metrics_list]),
            "hallucination_rate": hallucination_rate,  # 🚨 Add new rate
            "avg_routing_time": np.mean([m.routing_time for m in metrics_list]),
            "avg_hyde_time": np.mean([m.hyde_time for m in metrics_list]),
            "avg_retrieval_time": np.mean([m.retrieval_time for m in metrics_list]),
            "avg_generation_time": np.mean([m.generation_time for m in metrics_list]),
            "time_period_hours": hours if hours else "all"
        }
    
    def get_performance_alerts(self) -> List[Dict[str, Any]]:
        """Get alerts for performance issues."""
        alerts = []
        recent_metrics = self.get_aggregate_metrics(hours=24)
        
        if not recent_metrics:
            return alerts
        
        thresholds = self.performance_thresholds
        
        # ... (other alerts remain the same) ...
        
        if recent_metrics.get("avg_retrieval_time", 0.0) > thresholds["max_retrieval_time"]:
            alerts.append({
                "type": "RETRIEVAL_LATENCY_HIGH",
                "metric": recent_metrics["avg_retrieval_time"],
                "threshold": thresholds["max_retrieval_time"],
                "message": "Vector store retrieval time is slow. Check database performance."
            })

        if recent_metrics.get("avg_generation_time", 0.0) > thresholds["max_generation_time"]:
            alerts.append({
                "type": "GENERATION_LATENCY_HIGH",
                "metric": recent_metrics["avg_generation_time"],
                "threshold": thresholds["max_generation_time"],
                "message": "LLM generation time is slow. Check model provider or prompt complexity."
            })
        
        if "avg_retrieval_similarity" in recent_metrics and recent_metrics["avg_retrieval_similarity"] < thresholds["min_similarity"]:
            alerts.append({
                "type": "LOW_RETRIEVAL_QUALITY",
                "metric": recent_metrics["avg_retrieval_similarity"],
                "threshold": thresholds["min_similarity"],
                "message": "Retrieval similarity is low. Check chunking or embedding model."
            })

        if "avg_grounding_confidence" in recent_metrics and recent_metrics["avg_grounding_confidence"] < thresholds["min_grounding"]:
            alerts.append({
                "type": "LOW_GROUNDING_CONFIDENCE",
                "metric": recent_metrics["avg_grounding_confidence"],
                "threshold": thresholds["min_grounding"],
                "message": "Grounding confidence is low. High chance of hallucination. Check prompts."
            })
        
        return alerts
    
    def clear_history(self):
        """Clear metrics history."""
        self.metrics_history.clear()
        logger.info("Metrics history cleared")

# Global evaluator instance
evaluator = RAGEvaluator()