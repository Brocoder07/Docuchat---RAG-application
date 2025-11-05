"""
Comprehensive pipeline testing with multiple test scenarios.
Adjusted to provide a test user id for pipeline calls.
"""
import sys
import os
import logging
import time
from typing import Dict, Any, List
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.rag_pipeline import rag_pipeline
from core.evaluator import evaluator

def setup_test_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler('test_results.log', mode='w', encoding='utf-8')]
    )

class PipelineTester:
    def __init__(self):
        self.results = []
        self.start_time = None
        # default test user id
        self.test_user_id = "__test_user__"

    def run_all_tests(self) -> bool:
        self.start_time = datetime.now()
        logging.info("🧪 Starting DocuChat Pipeline Tests")
        tests = [self.test_initialization, self.test_document_processing, self.test_query_processing, self.test_error_handling, self.test_performance]
        all_passed = True
        for test_func in tests:
            test_name = test_func.__name__.replace('test_', '').replace('_', ' ').title()
            logging.info(f"\n🔬 Running: {test_name}")
            try:
                result = test_func()
                self.results.append({'test': test_name, 'passed': result, 'timestamp': datetime.now()})
                if result:
                    logging.info(f"✅ {test_name} - PASSED")
                else:
                    logging.error(f"❌ {test_name} - FAILED")
                    all_passed = False
            except Exception as e:
                logging.error(f"❌ {test_name} - ERROR: {str(e)}")
                self.results.append({'test': test_name, 'passed': False, 'error': str(e), 'timestamp': datetime.now()})
                all_passed = False
        self._generate_report(all_passed)
        return all_passed

    def test_initialization(self) -> bool:
        try:
            if not rag_pipeline.initialize():
                logging.error("Pipeline initialization failed")
                return False
            status = rag_pipeline.get_status()
            checks = [status.get("initialized", False) == True, status["vector_store"].get("initialized", False) == True, status["llm_service"].get("initialized", False) == True]
            return all(checks)
        except Exception as e:
            logging.error(f"Initialization test failed: {str(e)}")
            return False

    def test_document_processing(self) -> bool:
        try:
            test_content = """Test Document for Pipeline Validation

This is a test document to validate the RAG pipeline.

Key Topics:
- Artificial Intelligence (AI)
- Machine Learning (ML)
- Deep Learning
- Natural Language Processing (NLP)

AI refers to the simulation of human intelligence in machines.
Machine learning is a subset of AI that enables computers to learn without explicit programming.
"""
            test_file = "data/uploads/test_document.txt"
            os.makedirs(os.path.dirname(test_file), exist_ok=True)
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(test_content)

            result = rag_pipeline.process_document(test_file, "test_document.txt", user_id=self.test_user_id)

            if os.path.exists(test_file):
                os.remove(test_file)

            if result.get("success"):
                logging.info(f"✅ Document processed successfully: {result['chunks_count']} chunks created")
                return True
            else:
                logging.error(f"Document processing failed: {result.get('error')}")
                return False
        except Exception as e:
            logging.error(f"Document processing test failed: {str(e)}")
            return False

    def test_query_processing(self) -> bool:
        test_queries = ["What is artificial intelligence?", "What are the key topics in this document?", "Explain machine learning", "What does NLP stand for?"]
        passed_queries = 0
        for query in test_queries:
            try:
                result = rag_pipeline.query(query, user_id=self.test_user_id)
                if result.get("success") and result.get("answer"):
                    logging.info(f"✅ Query '{query}' - Answer generated ({len(result.get('sources', []))} sources)")
                    passed_queries += 1
                else:
                    logging.warning(f"⚠️ Query '{query}' - No answer generated")
            except Exception as e:
                logging.error(f"Query '{query}' failed: {str(e)}")
        success_rate = passed_queries / len(test_queries)
        logging.info(f"Query success rate: {success_rate:.1%} ({passed_queries}/{len(test_queries)})")
        return success_rate >= 0.5

    def test_error_handling(self) -> bool:
        invalid_scenarios = [("", "Empty query"), ("   ", "Whitespace-only query")]
        handled_errors = 0
        for invalid_input, description in invalid_scenarios:
            try:
                result = rag_pipeline.query(invalid_input, user_id=self.test_user_id)
                if result is not None:
                    handled_errors += 1
                    logging.info(f"✅ {description} - Handled gracefully")
                else:
                    logging.error(f"❌ {description} - Not handled properly")
            except Exception as e:
                logging.error(f"❌ {description} - Crashed: {str(e)}")
        return handled_errors == len(invalid_scenarios)

    def test_performance(self) -> bool:
        test_queries = ["What is AI?", "Explain machine learning", "What are the main topics?"]
        response_times = []
        for query in test_queries:
            start_time = time.time()
            result = rag_pipeline.query(query, user_id=self.test_user_id)
            end_time = time.time()
            response_time = end_time - start_time
            response_times.append(response_time)
            if result.get("success"):
                logging.info(f"✅ '{query}' - {response_time:.2f}s")
            else:
                logging.warning(f"⚠️ '{query}' - Failed in {response_time:.2f}s")
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        logging.info(f"Performance - Avg: {avg_response_time:.2f}s, Max: {max_response_time:.2f}s")
        return avg_response_time < 5.0 and max_response_time < 10.0

    def _generate_report(self, all_passed: bool):
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        logging.info("\n" + "=" * 60)
        logging.info("📊 TEST REPORT")
        logging.info("=" * 60)
        passed_tests = sum(1 for r in self.results if r['passed'])
        total_tests = len(self.results)
        logging.info(f"Total Tests: {total_tests}")
        logging.info(f"Passed: {passed_tests}")
        logging.info(f"Failed: {total_tests - passed_tests}")
        logging.info(f"Success Rate: {passed_tests/total_tests:.1%}")
        logging.info(f"Duration: {duration:.2f} seconds")
        logging.info(f"Overall: {'✅ PASSED' if all_passed else '❌ FAILED'}")
        logging.info("\n📋 Detailed Results:")
        for result in self.results:
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            logging.info(f"  {status} - {result['test']}")
            if 'error' in result:
                logging.info(f"     Error: {result['error']}")

def main():
    setup_test_logging()
    tester = PipelineTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()