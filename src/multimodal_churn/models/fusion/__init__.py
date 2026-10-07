"""Interchangeable fusion modules."""

from multimodal_churn.models.fusion.base import BaseFusion
from multimodal_churn.models.fusion.concat import ConcatFusion

__all__ = ["BaseFusion", "ConcatFusion"]
