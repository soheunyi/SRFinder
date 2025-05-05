import os
import pickle
import tempfile
from pathlib import Path
import pytest
import numpy as np
import sys
import torch


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_info import TrainingInfo
from fvt_classifier import FvTClassifier


def test_fvt_classifier():
    hash_ = "250427_130555_021949_U9NBeQ"
    tinfo = TrainingInfo.load(hash_)
    model = tinfo.load_trained_model("best")
    assert isinstance(model, FvTClassifier)
