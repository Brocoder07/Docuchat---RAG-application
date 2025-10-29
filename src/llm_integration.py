# src/llm_integration.py - FIXED VERSION
import logging
import requests
import time
import traceback
import json
from typing import Optional, List, Tuple, Dict, Any

logger = logging.getLogger(__name__)


class LLMIntegration:
    """
    Integration helper for Ollama-based LLMs with proper streaming response handling.
    """

    def __init__(self, model_name: str = "llama3.2:1b-instruct-q4_1", temperature: float = 0.2):
        self.model_name = model_name
        self.temperature = temperature
        self.initialized = False

    def initialize(self, use_openai: bool = False):
        logger.info("Ollama is running with model: %s", self.model_name)
        logger.info("Model %s is ready to use!", self.model_name)
        self.initialized = True

    def _build_optimized_context(self, relevant_chunks: List[Tuple[str, Optional[float], Dict]]) -> str:
        """
        Convert a list of (text, score, metadata) into a single context string.
        """
        parts = []
        for i, (text, score, meta) in enumerate(relevant_chunks):
            header = f"[CHUNK {i+1} | score={score}]"
            meta_line = f"[meta={meta}]" if meta else ""
            parts.append(f"{header} {meta_line}\n{text}")
        return "\n\n".join(parts)

    def _build_dynamic_prompt(self, question: str, context: str) -> str:
        """
        Build a robust, universal prompt that works for ANY document type.
        """
        # Universal rules that work for all document types
        universal_rules = (
            "CRITICAL INSTRUCTIONS - READ CAREFULLY:\n"
            "You are an assistant that answers using ONLY the information present in the provided Context.\n"
            "You MUST follow these rules exactly:\n\n"
        
            "CONTEXT RULES:\n"
            "1. Use ONLY the information from the Context below. Do not use any prior knowledge.\n"
            "2. If information is missing from Context, say 'The context does not provide information about X'\n"
            "3. Do not add, infer, or assume any information not explicitly stated\n"
            "4. Do not make comparisons, draw conclusions, or provide analysis beyond what's directly stated\n"
            "5. If you cannot answer based on Context, say so explicitly\n\n"
        
            "CITATION RULES:\n" 
            "6. ALWAYS reference which chunks contain the information (e.g., 'Based on Chunk 1 and Chunk 3')\n"
            "7. When listing items, cite the specific chunk for each item\n"
            "8. If information appears in multiple chunks, mention all relevant chunks\n\n"
        
            "LANGUAGE RULES:\n"
            "9. Avoid speculative language (probably, might be, could be, I think, I believe)\n"
            "10. Avoid absolute statements (always, never, all, every) unless explicitly stated\n"
            "11. Avoid vague quantifiers (many, several, some) - be specific about what's in Context\n"
            "12. Do not reference external knowledge, studies, or common practices\n\n"
        
            "STRUCTURE RULES:\n"
            "13. For lists or collections mentioned in Context, list them exactly as they appear\n"
            "14. For technical terms, use the exact terminology from Context\n"
            "15. For names, titles, or proper nouns, use the exact spelling from Context\n"
            "16. For numerical data, use the exact values from Context\n\n"
        
            "SAFETY RULES:\n"
            "17. If Context contains conflicting information, acknowledge the conflict\n"
            "18. If you're unsure, err on the side of caution and admit limitations\n"
            "19. Format your answer clearly but do not invent structure not in Context\n"
            "20. Do not combine information from different documents unless explicitly connected in Context\n\n"
        )
    
        # Add question-type specific guidance (not document-specific)
        question_lower = question.lower()
        question_specific_guidance = self._get_question_specific_guidance(question_lower)
    
        prompt_template = (
            f"{universal_rules}"
            f"{question_specific_guidance}"
            "CONTEXT:\n{context}\n\n"
            "QUESTION: {question}\n\n"
            "YOUR ANSWER (following all rules above):"
        )
        return prompt_template.format(context=context, question=question)

    def _get_question_specific_guidance(self, question_lower: str) -> str:
        """Get question-type specific guidance that works for any document."""
        guidance = ""
    
        # List/collection questions (works for projects, products, categories, etc.)
        if any(keyword in question_lower for keyword in ['list', 'what are', 'which', 'name the']):
            guidance += (
                "LIST-SPECIFIC RULES:\n"
                "21. List items exactly as they appear in Context, in the order they appear\n"
                "22. Do not group, categorize, or reorganize items unless Context does so\n"
                "23. If Context lists items with bullet points or numbers, maintain that structure\n"
                "24. Cite the specific chunk where each listed item appears\n\n"
            )
    
        # Technical/description questions
        if any(keyword in question_lower for keyword in ['describe', 'explain', 'what is', 'how does']):
            guidance += (
                "DESCRIPTION RULES:\n"
                "25. Use the exact technical terms and definitions from Context\n"
                "26. Do not simplify or rephrase technical concepts unless Context does so\n"
                "27. Include specific measurements, specifications, or details mentioned in Context\n"
                "28. Reference the chunks that contain the detailed descriptions\n\n"
            )
    
        # Comparison questions
        if any(keyword in question_lower for keyword in ['compare', 'difference', 'versus', 'vs']):
            guidance += (
                "COMPARISON RULES:\n"
                "29. Only compare elements that are explicitly compared in Context\n"
                "30. Do not infer comparisons that are not directly stated\n"
                "31. If Context doesn't compare items, say 'The context does not compare X and Y'\n"
                "32. Reference chunks where comparison information appears\n\n"
            )
    
        # Quantitative questions
        if any(keyword in question_lower for keyword in ['how many', 'how much', 'percentage', 'number of']):
            guidance += (
                "QUANTITATIVE RULES:\n"
                "33. Use exact numbers, percentages, or quantities from Context\n"
                "34. Do not calculate, estimate, or approximate unless Context does so\n"
                "35. Include units of measurement exactly as they appear in Context\n"
                "36. Reference chunks containing the quantitative data\n\n"
            )
    
        return guidance

    def _parse_ollama_streaming_response(self, response_text: str) -> str:
        """
        Parse Ollama's streaming JSON response and extract the complete answer.
        """
        try:
            lines = response_text.strip().split('\n')
            full_response = ""
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    # Parse each JSON object
                    data = json.loads(line)
                    
                    # Extract the response part
                    if 'response' in data:
                        full_response += data['response']
                        
                    # Check if this is the final chunk
                    if data.get('done', False):
                        break
                        
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON line: {line}")
                    continue
                    
            return full_response.strip()
            
        except Exception as e:
            logger.error(f"Error parsing Ollama streaming response: {str(e)}")
            # Fallback: try to extract any text that looks like a response
            if 'response' in response_text:
                import re
                matches = re.findall(r'"response":"([^"]*)"', response_text)
                return ' '.join(matches).strip()
            return ""

    def _call_ollama_with_retry(self, prompt: str, max_retries: int = 3, base_timeout: int = 60) -> Optional[str]:
        """
        Low-level HTTP call to Ollama with proper streaming response handling.
        """
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": self.temperature,
            "stream": False,  # IMPORTANT: Set to False to get complete response
        }
        headers = {"Content-Type": "application/json"}
        last_exc = None

        for attempt in range(1, max_retries + 1):
            try:
                logger.debug("Calling Ollama (attempt %d) with timeout=%ds", attempt, base_timeout)
                resp = requests.post(url, json=payload, headers=headers, timeout=base_timeout)
                resp.raise_for_status()
                
                # Parse the response
                try:
                    data = resp.json()
                    
                    # Handle both streaming and non-streaming responses
                    if 'response' in data:
                        return data['response']
                    else:
                        # If we get streaming format, parse it
                        response_text = resp.text
                        if response_text.strip().startswith('{'):
                            return self._parse_ollama_streaming_response(response_text)
                        else:
                            return response_text
                            
                except json.JSONDecodeError:
                    # Response is not JSON, return as text
                    return resp.text
                    
            except requests.exceptions.RequestException as e:
                last_exc = e
                wait = 2 ** (attempt - 1)
                logger.warning("Ollama call attempt %d failed: %s (waiting %ds before retry)", attempt, str(e), wait)
                time.sleep(wait)

        logger.error("Ollama call failed after %d attempts. Last error: %s", max_retries, repr(last_exc))
        return None

    def _call_ollama_api(self, question: str, relevant_chunks: Optional[List[Tuple[str, Optional[float], Dict]]] = None) -> Optional[str]:
        """
        Call Ollama API with optimized prompt and parameters.
        """
        try:
            # If relevant_chunks is None, assume `question` is already a full prompt.
            if relevant_chunks is None:
                logger.info("Using prompt-only mode for Ollama call (caller provided full prompt).")
                return self._call_ollama_with_retry(question)

            # Build optimized context and dynamic prompt from question + relevant chunks
            context = self._build_optimized_context(relevant_chunks)
            prompt = self._build_dynamic_prompt(question, context)

            logger.info("Calling Ollama with generated prompt (length: %d)", len(prompt))
            return self._call_ollama_with_retry(prompt)
        except Exception as e:
            logger.error("Error calling Ollama API: %s", str(e))
            logger.debug(traceback.format_exc())
            return None