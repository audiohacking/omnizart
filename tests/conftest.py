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
    
    # Check for variables.data files which are required for model loading
    data_files = [f for f in os.listdir(variables_dir) if f.startswith("variables.data")]
    return len(data_files) > 0


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
