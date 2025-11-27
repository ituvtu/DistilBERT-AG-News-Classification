import pytest
import json
import os
from unittest.mock import patch, MagicMock


from app import load_example_headlines, classify_headline


def test_load_example_headlines_valid(tmp_path):
    """Check whether the function reads a valid JSON file."""

    data = [["Headline 1"], ["Headline 2"]]
    file = tmp_path / "test_examples.json"
    file.write_text(json.dumps(data), encoding="utf-8")
    
    result = load_example_headlines(str(file))
    assert result == data

def test_load_example_headlines_invalid():
    """Check whether the function returns an empty list in case of an error."""
    result = load_example_headlines("non_existent_file.json")
    assert result == []



@patch('app.news_classifier')
def test_classify_headline_success(mock_classifier):
    """
    Verify the successful classification.
    """
    mock_classifier.return_value = [
        {"label": "Sports", "score": 0.95},
        {"label": "World", "score": 0.05}
    ]
    
    text = "Team wins the match"
    result = classify_headline(text)
    
    assert result["Sports"] == 0.95
    assert result["World"] == 0.05
    mock_classifier.assert_called_with(text, top_k=None)

@patch('app.news_classifier')
def test_classify_headline_empty_input(mock_classifier):
    """Checking the handling of empty input."""
    result = classify_headline("")
    assert "Please enter valid text" in result
    assert result["Please enter valid text"] == 0.0
    
    mock_classifier.assert_not_called()

def test_classify_headline_model_none():
    """Check the case when the model did not load (news_classifier is None)."""
    with patch('app.news_classifier', None):
        result = classify_headline("Some text")
        assert "Model failed to initialize" in result