import os
import pickle
import tempfile
from pathlib import Path
import pytest
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_info import TrainingInfo


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)


@pytest.fixture
def sample_tinfo():
    """Create a sample TrainingInfo object."""
    hparams = {
        "experiment_name": "test_experiment",
        "data_seed": 42,
        "val_ratio": 0.2,
        "model": "FvTClassifier",
    }
    return TrainingInfo(
        hparams=hparams, ms_hash="test_hash", ms_idx=np.array([True, False, True])
    )


def test_save_and_load(temp_dir, sample_tinfo):
    """Test saving and loading a TrainingInfo object."""
    # Set up test directory
    TrainingInfo.SAVE_DIR = temp_dir / "training_info"
    TrainingInfo.META_DIR = temp_dir / "metadata" / "training_info.pkl"
    TrainingInfo.SAVE_DIR.mkdir(parents=True)
    TrainingInfo.META_DIR.parent.mkdir(parents=True)

    # Save the object
    sample_tinfo.save()

    # Test loading with metadata
    loaded_tinfo = TrainingInfo.load(sample_tinfo.hash, use_metadata=True)
    assert loaded_tinfo.hash == sample_tinfo.hash
    assert loaded_tinfo.experiment_name == sample_tinfo.experiment_name
    assert loaded_tinfo.hparams == sample_tinfo.hparams

    # Test loading without metadata
    loaded_tinfo = TrainingInfo.load(sample_tinfo.hash, use_metadata=False)
    assert loaded_tinfo.hash == sample_tinfo.hash


def test_delete_with_metadata(temp_dir, sample_tinfo):
    """Test deleting TrainingInfo objects using metadata."""
    # Set up test directory
    TrainingInfo.SAVE_DIR = temp_dir / "training_info"
    TrainingInfo.META_DIR = temp_dir / "metadata" / "training_info.pkl"
    TrainingInfo.SAVE_DIR.mkdir(parents=True)
    TrainingInfo.META_DIR.parent.mkdir(parents=True)

    # Save the object
    sample_tinfo.save()

    # Test deletion with metadata
    TrainingInfo.delete([sample_tinfo.hash], use_metadata=True, yes=True)

    # Verify file is deleted
    experiment_dir = TrainingInfo.SAVE_DIR / sample_tinfo.experiment_name
    assert not (experiment_dir / sample_tinfo.hash).exists()


def test_multiple_experiments(temp_dir):
    """Test handling multiple experiments."""
    # Set up test directory
    TrainingInfo.SAVE_DIR = temp_dir / "training_info"
    TrainingInfo.META_DIR = temp_dir / "metadata" / "training_info.pkl"
    TrainingInfo.SAVE_DIR.mkdir(parents=True)
    TrainingInfo.META_DIR.parent.mkdir(parents=True)

    # Create two TrainingInfo objects in different experiments
    tinfo1 = TrainingInfo(
        hparams={
            "experiment_name": "exp1",
            "data_seed": 42,
            "val_ratio": 0.2,
            "model": "FvTClassifier",
        },
        ms_hash="test_hash1",
        ms_idx=np.array([True, False, True]),
    )

    tinfo2 = TrainingInfo(
        hparams={
            "experiment_name": "exp2",
            "data_seed": 42,
            "val_ratio": 0.2,
            "model": "FvTClassifier",
        },
        ms_hash="test_hash2",
        ms_idx=np.array([True, False, True]),
    )

    # Save both objects
    tinfo1.save()
    tinfo2.save()

    # Test loading each from their respective experiments
    loaded1 = TrainingInfo.load(tinfo1.hash, use_metadata=True)
    loaded2 = TrainingInfo.load(tinfo2.hash, use_metadata=True)

    assert loaded1.experiment_name == "exp1"
    assert loaded2.experiment_name == "exp2"


def test_metadata_caching(temp_dir, sample_tinfo):
    """Test metadata caching functionality."""
    # Set up test directory
    TrainingInfo.SAVE_DIR = temp_dir / "training_info"
    TrainingInfo.META_DIR = temp_dir / "metadata" / "training_info.pkl"
    TrainingInfo.SAVE_DIR.mkdir(parents=True)
    TrainingInfo.META_DIR.parent.mkdir(parents=True)

    # Save the object
    sample_tinfo.save()

    # Load metadata twice - second call should use cache
    metadata1 = TrainingInfo.load_cached_metadata()
    metadata2 = TrainingInfo.load_cached_metadata()

    # Verify metadata is the same
    assert metadata1 == metadata2


def test_find_with_metadata(temp_dir):
    """Test finding TrainingInfo objects using metadata."""
    # Set up test directory
    TrainingInfo.SAVE_DIR = temp_dir / "training_info"
    TrainingInfo.META_DIR = temp_dir / "metadata" / "training_info.pkl"
    TrainingInfo.SAVE_DIR.mkdir(parents=True)
    TrainingInfo.META_DIR.parent.mkdir(parents=True)

    # Create and save multiple TrainingInfo objects
    tinfos = []
    for i in range(3):
        tinfo = TrainingInfo(
            hparams={
                "experiment_name": f"exp{i}",
                "data_seed": 42,
                "val_ratio": 0.2,
                "model": "FvTClassifier",
            },
            ms_hash=f"test_hash{i}",
            ms_idx=np.array([True, False, True]),
        )
        tinfo.save()
        tinfos.append(tinfo)

    # Test finding with metadata
    # But should update metadata first
    TrainingInfo.update_metadata()

    hashes = TrainingInfo.find({"model": "FvTClassifier"}, from_metadata=True)
    assert len(hashes) == 3
    assert all(h in [t.hash for t in tinfos] for h in hashes)


def test_error_handling(temp_dir):
    """Test error handling for invalid operations."""
    # Set up test directory
    TrainingInfo.SAVE_DIR = temp_dir / "training_info"
    TrainingInfo.META_DIR = temp_dir / "metadata" / "training_info.pkl"
    TrainingInfo.SAVE_DIR.mkdir(parents=True)
    TrainingInfo.META_DIR.parent.mkdir(parents=True)

    # Test loading non-existent hash
    with pytest.raises(FileNotFoundError):
        TrainingInfo.load("nonexistent_hash")

    # Test deleting non-existent hash
    TrainingInfo.delete(["nonexistent_hash"], yes=True)  # Should not raise error


def test_aux_info(temp_dir, sample_tinfo):
    """Test auxiliary information handling."""
    # Set up test directory
    TrainingInfo.SAVE_DIR = temp_dir / "training_info"
    TrainingInfo.META_DIR = temp_dir / "metadata" / "training_info.pkl"
    TrainingInfo.SAVE_DIR.mkdir(parents=True)
    TrainingInfo.META_DIR.parent.mkdir(parents=True)

    # Add auxiliary information
    sample_tinfo.update_aux_info(test_key="test_value")
    sample_tinfo.save()

    # Load and verify auxiliary information
    loaded_tinfo = TrainingInfo.load(sample_tinfo.hash)
    assert loaded_tinfo.aux_info["test_key"] == "test_value"
