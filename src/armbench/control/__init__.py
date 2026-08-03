"""Controllers and simplified joint-space tracking simulation."""

from armbench.control.controllers import DiscreteLQR, PDController
from armbench.control.simulation import TrackingResult, simulate_tracking

__all__ = ["DiscreteLQR", "PDController", "TrackingResult", "simulate_tracking"]

