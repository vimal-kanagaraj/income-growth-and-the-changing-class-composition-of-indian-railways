"""Income growth and the changing class composition of Indian Railways, 2015-16 to 2024-25.

Analysis code behind the DA241O Managerial Economics term paper by The Cartel (Group 1):
builds the panel, estimates the models and generates the paper's tables and figures.
"""

from railclass.pipeline import Results, run, run_analysis

__version__ = "1.0.0"
__all__ = ["Results", "__version__", "run", "run_analysis"]
