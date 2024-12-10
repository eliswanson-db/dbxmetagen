from src.dbxmetagen.sampling import determine_sampling_ratio

def test_determine_sampling_ratio():
    nrows = 100
    sample_size = 10
    expected_ratio = 0.1
    assert determine_sampling_ratio(nrows, sample_size) == expected_ratio

    nrows = 50
    sample_size = 50
    expected_ratio = 1.0
    assert determine_sampling_ratio(nrows, sample_size) == expected_ratio

    nrows = 30
    sample_size = 100
    expected_ratio = 1.0
    assert determine_sampling_ratio(nrows, sample_size) == expected_ratio

    nrows = 0
    sample_size = 10
    expected_ratio = 1.0
    assert determine_sampling_ratio(nrows, sample_size) == expected_ratio
