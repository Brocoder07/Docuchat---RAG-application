"""
RAG evaluation metrics for Week 2 optimization and monitoring.
"""
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
import datetime

logger = logging.getLogger(__name__)

class RAGEvaluator:
    """Evaluation metrics for RAG pipeline (Week 2 requirements)."""
    
    def __init__(self):
        self.metrics_history = []
    
    @staticmethod
    def calculate_retrieval_precision(relevant_chunks: List[Tuple], query: str) -> float:
        """
        Calculate precision of retrieval based on query relevance.
        Simple keyword-based precision calculation.
        """
        if not relevant_chunks:
            return 0.0
        
        query_terms = set(query.lower().split())
        relevant_count = 0
        
        for chunk, score, metadata in relevant_chunks:
            chunk_terms = set(chunk.lower().split())
            # Count chunks that share at least one term with query
            if query_terms.intersection(chunk_terms):
                relevant_count += 1
        
        return relevant_count / len(relevant_chunks)
    
    @staticmethod
    def calculate_answer_relevance(answer: str, question: str) -> float:
        """Calculate how relevant the answer is to the question."""
        if not answer or not question:
            return 0.0
            
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        
        if not question_words:
            return 0.0
        
        overlap = len(question_words.intersection(answer_words))
        return overlap / len(question_words)
    
    @staticmethod
    def detect_hallucination(answer: str, context_chunks: List[Tuple]) -> float:
        """
        Simple hallucination detection based on context grounding.
        Returns hallucination rate (0.0 = no hallucination, 1.0 = complete hallucination)
        """
        if not answer or not context_chunks:
            return 1.0
        
        answer_lower = answer.lower()
        context_text = " ".join([chunk[0].lower() for chunk in context_chunks])
        
        # Split answer into meaningful sentences/phrases
        import re
        sentences = re.split(r'[.!?]+', answer_lower)
        meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
        
        if not meaningful_sentences:
            return 1.0
        
        grounded_sentences = 0
        for sentence in meaningful_sentences:
            # Check if sentence appears in context (simple substring match)
            if sentence in context_text:
                grounded_sentences += 1
            else:
                # Check for partial matches (key phrases)
                words = sentence.split()
                if len(words) > 3:
                    # Check if majority of key phrases are in context
                    key_phrases = [' '.join(words[i:i+3]) for i in range(0, len(words)-2)]
                    matches = sum(1 for phrase in key_phrases if phrase in context_text)
                    if matches / len(key_phrases) > 0.5:
                        grounded_sentences += 1
        
        hallucination_rate = 1.0 - (grounded_sentences / len(meaningful_sentences))
        return max(0.0, min(1.0, hallucination_rate))
    
    def evaluate_query(self, question: str, answer: str, relevant_chunks: List[Tuple]) -> Dict[str, float]:
        """Comprehensive evaluation for a single query."""
        metrics = {
            "retrieval_precision": self.calculate_retrieval_precision(relevant_chunks, question),
            "answer_relevance": self.calculate_answer_relevance(answer, question),
            "hallucination_rate": self.detect_hallucination(answer, relevant_chunks),
            "chunks_retrieved": len(relevant_chunks),
            "avg_similarity_score": np.mean([score for _, score, _ in relevant_chunks]) if relevant_chunks else 0.0,
            "max_similarity_score": max([score for _, score, _ in relevant_chunks]) if relevant_chunks else 0.0
        }
        
        # Store in history
        self.metrics_history.append({
            "question": question,
            "metrics": metrics,
            "timestamp": datetime.datetime.now()
        })
        
        logger.info(f"Query evaluation - Precision: {metrics['retrieval_precision']:.3f}, "
                   f"Hallucination: {metrics['hallucination_rate']:.3f}, "
                   f"Relevance: {metrics['answer_relevance']:.3f}")
        
        return metrics
    
    def get_aggregate_metrics(self) -> Dict[str, float]:
        """Get aggregate metrics across all evaluated queries."""
        if not self.metrics_history:
            return {}
        
        metrics_list = [item["metrics"] for item in self.metrics_history]
        
        return {
            "avg_retrieval_precision": np.mean([m["retrieval_precision"] for m in metrics_list]),
            "avg_answer_relevance": np.mean([m["answer_relevance"] for m in metrics_list]),
            "avg_hallucination_rate": np.mean([m["hallucination_rate"] for m in metrics_list]),
            "total_queries_evaluated": len(self.metrics_history),
            "avg_chunks_per_query": np.mean([m["chunks_retrieved"] for m in metrics_list]),
            "avg_similarity_score": np.mean([m["avg_similarity_score"] for m in metrics_list])
        }
    
    def get_recent_metrics(self, hours: int = 24) -> Dict[str, float]:
        """Get metrics from recent queries only."""
        cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=hours)
        recent_metrics = [item for item in self.metrics_history 
                         if item["timestamp"] > cutoff_time]
        
        if not recent_metrics:
            return {}
            
        metrics_list = [item["metrics"] for item in recent_metrics]
        
        return {
            "recent_queries": len(recent_metrics),
            "recent_avg_precision": np.mean([m["retrieval_precision"] for m in metrics_list]),
            "recent_avg_relevance": np.mean([m["answer_relevance"] for m in metrics_list]),
            "recent_avg_hallucination": np.mean([m["hallucination_rate"] for m in metrics_list])
        }
    
    def clear_history(self):
        """Clear evaluation history."""
        self.metrics_history.clear()
        logger.info("Evaluation history cleared")

# Global evaluator instance
evaluator = RAGEvaluator()