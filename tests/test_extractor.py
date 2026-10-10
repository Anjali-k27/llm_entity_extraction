import unittest
from unittest.mock import patch

from langchain_core.runnables import RunnableLambda

import llm_pipeline.extractor as extractor_module


class TestExtractor(unittest.TestCase):
    def test_build_extraction_chain_uses_supported_gemini_model(self):
        with patch.object(extractor_module, "ChatGoogleGenerativeAI") as mock_llm:
            mock_llm.return_value.with_structured_output.return_value = RunnableLambda(lambda value: value)

            extractor_module.build_extraction_chain()

            _, kwargs = mock_llm.call_args
            self.assertEqual(kwargs["model"], "gemini-3.8-flash")


if __name__ == "__main__":
    unittest.main()
