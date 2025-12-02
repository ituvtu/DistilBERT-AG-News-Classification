import json
from unittest.mock import patch

from app import load_example_headlines, classify_headline

def test_load_example_headlines_valid(tmp_path):
    data = [["Headline 1"], ["Headline 2"]]
    file = tmp_path / "test_examples.json"
    file.write_text(json.dumps(data), encoding="utf-8")
    result = load_example_headlines(str(file))
    assert result == data

def test_load_example_headlines_invalid_file_returns_empty():
    result = load_example_headlines("non_existent_file.json")
    assert result == []

@patch('app.news_classifier')
def test_classify_headline_returns_label_scores(mock_classifier):
    mock_classifier.return_value = [
        {"label": "Sports", "score": 0.95},
        {"label": "World", "score": 0.05}
    ]
    
    text = "Team wins the match"
    result = classify_headline(text)
    assert result == {"Sports": 0.95, "World": 0.05}
    mock_classifier.assert_called_with(text, top_k=None)

@patch('app.news_classifier')
def test_classify_headline_empty_input_returns_prompt(mock_classifier):
    result = classify_headline("")
    assert result == {"Please enter valid text": 0.0}
    mock_classifier.assert_not_called()

def test_classify_headline_when_model_is_none_returns_init_error():
    with patch('app.news_classifier', None):
        result = classify_headline("Some text")
        assert result == {"Model failed to initialize": 0.0}

def test_classify_headline_when_model_raises_returns_error_key():
    with patch('app.news_classifier') as mock_classifier:
        mock_classifier.side_effect = Exception("Model failure")
        result = classify_headline("Some headline")
        assert any(str(k).startswith("Error:") for k in result.keys())

def test_classify_headline_unexpected_prediction_format_returns_error_key():
    with patch('app.news_classifier') as mock_classifier:
        mock_classifier.return_value = [{"bad_key": "value"}]
        result = classify_headline("Another headline")
        assert any(str(k).startswith("Error:") for k in result.keys())

def test_load_example_headlines_reads_repository_examples():
    data = load_example_headlines("examples.json")
    with open("examples.json", "r", encoding="utf-8") as f:
        expected = json.load(f)
    assert data == expected
