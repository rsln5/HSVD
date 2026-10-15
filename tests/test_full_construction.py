from checks.verify_full_construction import verify


def test_general_construction_in_independent_exact_arithmetic():
    assert verify()["status"] == "passed"
