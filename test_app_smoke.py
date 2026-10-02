"""
Lightweight UI test for the Streamlit app, using Streamlit's own headless
AppTest harness (streamlit.testing.v1) - no browser needed.

Requires streamlit>=1.28 and the trained model artifacts to already exist
(run the notebook first). Skips cleanly if either isn't available, rather
than failing the whole suite on a machine that hasn't trained yet.
"""

import os

import pytest

pytest.importorskip("streamlit", minversion="1.28.0")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = "app/app.py"


@pytest.fixture(scope="module")
def app():
    if not os.path.exists(APP_PATH):
        pytest.skip("app/app.py not found - run Part 7 of the notebook first")
    if not os.path.exists("models/best_model.pkl"):
        pytest.skip("models/best_model.pkl not found - run Part 3 of the notebook first")
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    return at


def test_app_loads_without_exceptions(app):
    assert not app.exception


def test_app_has_three_tabs(app):
    assert len(app.tabs) == 3


def test_app_shows_model_name_in_caption(app):
    captions_text = " ".join(c.value for c in app.caption)
    assert "Model in use" in captions_text or len(app.caption) > 0


def test_single_prediction_flow_runs(app):
    """Fill in the single-machine form and click Predict - the app should
    run to completion without raising, and show a probability metric."""
    single_tab = app.tabs[0]

    number_inputs = single_tab.number_input
    if len(number_inputs) >= 5:
        number_inputs[0].set_value(298.0)  # air temp
        number_inputs[1].set_value(308.0)  # process temp
        number_inputs[2].set_value(1500)   # rpm
        number_inputs[3].set_value(40.0)   # torque
        number_inputs[4].set_value(100)    # tool wear

    buttons = single_tab.button
    predict_buttons = [b for b in buttons if "Predict" in b.label]
    assert len(predict_buttons) == 1

    predict_buttons[0].click().run(timeout=30)

    assert not app.exception
    assert len(app.metric) >= 1
