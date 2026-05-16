"""
brain_forecast — forecasting future brain activity from movie stimuli and brain history.

A modular harness for evaluating multiple model families (TFT, Banded Ridge,
AR, Moving Average, Persistence) on subject-out CV across multiple horizons.

Hypothesis under test:
    future_brain(t + H) = f(past_brain(t-k..t), stimulus(t-k..t))
"""

__version__ = "0.1.0"
