"""
Main entry point for Black Myth: Wukong Steam reviews analysis project
This file orchestrates the complete analysis workflow
"""
import logging
from updated_analysis import main

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

if __name__ == "__main__":
    setup_logging()
    main()