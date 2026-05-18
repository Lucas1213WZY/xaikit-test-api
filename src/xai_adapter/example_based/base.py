"""Base class for example-based explanation adapters."""

from __future__ import annotations

from typing import Optional

from ..base import (
    ArrayLike,
    PostprocessFn,
    PreprocessFn,
    XAIAdapter,
)


class ExampleBasedAdapter(XAIAdapter):
    """Marker base class for example-based explanation methods."""

    def __init__(
        self,
        *,
        target: int = 1,
        preprocessing_fn: Optional[PreprocessFn] = None,
        postprocessing_fn: Optional[PostprocessFn] = None,
    ):
        from ..base import identity_postprocess, identity_preprocess
        self.target = target
        self.preprocessing_fn = preprocessing_fn or identity_preprocess
        self.postprocessing_fn = postprocessing_fn or identity_postprocess
        self.is_fitted = False

    def fit(self, X: ArrayLike = None, y: ArrayLike = None, **kwargs):
        self.is_fitted = True
        return self


__all__ = ["ExampleBasedAdapter"]
