"""Plugin point for an actual segmentation backbone + intervention operators.

This package contains no MedSAM/nnInteractive proprietary checkpoints and
makes no claim to reproduce those models' own correction mechanisms.

Implement `execute` using REAL saved pretrained model weights and freeze all
prompt/localizer decisions *without* test labels. Return a mask on precisely
the same voxel grid as the input and archive backend version/checkpoint.
"""
from abc import ABC, abstractmethod
import numpy as np
from .actions_demo import demo_action


class RepairBackend(ABC):
    @abstractmethod
    def execute(self,image,current_mask,action,case):
        """Return binary ndarray for the requested action, without touching case['gt']."""
        raise NotImplementedError


class DemoHeuristicBackend(RepairBackend):
    """ONLY for functionality checks; never use for medical study results."""
    def execute(self,image,current_mask,action,case):
        return demo_action(image,current_mask,action)
