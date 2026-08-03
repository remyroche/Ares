from scripts.materialize_aug2022_inverse_pi_gap_download_request import END, START, PRODUCTS


def test_gap_request_is_exactly_the_uncovered_interval() -> None:
    assert START.isoformat() == "2022-08-01T12:00:00+00:00"
    assert END.isoformat() == "2022-08-30T12:00:00+00:00"
    assert len(PRODUCTS) == 5
