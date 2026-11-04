"""
Module: signal_control.py
Root entry point for Member 4 – Adaptive Signal Control
Vision-Based Adaptive Traffic Signal Management System

Executes src/signal_control.py:
  - Input: Traffic metrics from Member 3 (reports/module3/)
  - Output: Dynamic signal decisions and reports (reports/module4/)
  - Calculates road priority scores
  - Dynamically allocates green-light durations
  - Handles pedestrian and emergency vehicle priorities
  - Benchmarks against fixed-time signal control
"""

from pathlib import Path
import sys

# Ensure src/ is on Python search path
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from signal_control import main

if __name__ == "__main__":
    main()
