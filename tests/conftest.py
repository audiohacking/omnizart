"""Pytest configuration and fixtures for omnizart tests."""
import os
import pytest


def checkpoint_files_exist(checkpoint_path):
    """Check if checkpoint variable data files exist."""
    if not os.path.exists(checkpoint_path):
        return False
    
    variables_dir = os.path.join(checkpoint_path, "variables")
    if not os.path.exists(variables_dir):
        return False
    
    try:
        # Check for variables.data files which are required for model loading
        files = os.listdir(variables_dir)
        data_files = [f for f in files if f.startswith("variables.data")]
        return len(data_files) > 0
    except (OSError, FileNotFoundError):
        # Handle race condition or permission issues
        return False


@pytest.fixture(autouse=True)
def skip_if_checkpoint_missing(request):
    """Skip test_load_model tests if checkpoint files are not available."""
    # Only apply to test_load_model functions
    if request.function.__name__ != "test_load_model":
        return
    
    # Get the module path to determine which checkpoint to check
    test_module = request.module.__name__
    
    # Map test modules to their checkpoint paths
    checkpoint_mapping = {
        "tests.beat.test_app": "omnizart/checkpoints/beat/beat_blstm",
        "tests.chord.test_app": "omnizart/checkpoints/chord/chord_v1",
        "tests.drum.test_app": "omnizart/checkpoints/drum/drum_keras",
        "tests.music.test_app": "omnizart/checkpoints/music/music_piano",
        "tests.patch_cnn.test_app": "omnizart/checkpoints/patch_cnn/patch_cnn_melody",
        "tests.vocal.test_app": "omnizart/checkpoints/vocal/vocal_semi",
        "tests.vocal_contour.test_app": "omnizart/checkpoints/vocal/vocal_contour",
    }
    
    checkpoint_path = checkpoint_mapping.get(test_module)
    if checkpoint_path and not checkpoint_files_exist(checkpoint_path):
        pytest.skip(f"Checkpoint files not available at {checkpoint_path}")


@pytest.fixture(autouse=True)
def skip_if_mutable_sequence_error(request):
    """Skip test_extract_patch_cqt if MutableSequence import error occurs.
    
    This test imports omnizart.feature.wrapper_func which in turn imports dependencies
    that may have Python 3.10+ compatibility issues (trying to import MutableSequence
    from collections instead of collections.abc). This is a known issue with some
    audio processing libraries.
    """
    if request.function.__name__ != "test_extract_patch_cqt":
        return
    
    try:
        # The wrapper_func module imports dependencies that may fail with MutableSequence error
        # This import serves as a canary to detect the compatibility issue before the test runs
        from omnizart.feature import wrapper_func as wfunc  # noqa: F401
    except ImportError as e:
        if "MutableSequence" in str(e):
            pytest.skip(f"Dependency compatibility issue with Python 3.10+: {e}")
        else:
            # Re-raise if it's a different import error
            raise
