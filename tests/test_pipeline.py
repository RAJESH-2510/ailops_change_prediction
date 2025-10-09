import pytest
from src.pipeline import AIOpsPipeline

@pytest.fixture
def pipeline():
    return AIOpsPipeline()

def test_ingest_data(pipeline):
    sample_data = {'deployment_id': 'build123', 'lines_changed': 450}
    result = pipeline.ingest_data('metrics', sample_data)
    assert 'quality_score' in result
    assert result['quality_score'] > 0
