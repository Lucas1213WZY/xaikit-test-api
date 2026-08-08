"""Data-source adapter for the XAI desiderata sim-to-real corpus."""

from .coax_adapter import CoAXDataSource


class Sim2RealDataSource(CoAXDataSource):
    """Load the CoAX-shaped sim-to-real corpus while retaining its source id."""

    def __init__(self, normalizer=None):
        super().__init__(normalizer=normalizer)
        self.source_type = "sim2real"


__all__ = ["Sim2RealDataSource"]
