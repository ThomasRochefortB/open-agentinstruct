"""Avoid pricing downloads and local dotenv credentials during hermetic tests."""
import os

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
