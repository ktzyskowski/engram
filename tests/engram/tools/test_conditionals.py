from engram.tools.conditionals import Every, Ratio


def test_every_trips_on_nth_call():
    every = Every(n=3)
    results = [every() for _ in range(9)]
    assert results == [False, False, True, False, False, True, False, False, True]


def test_every_n_one_always_true():
    every = Every(n=1)
    assert all(every() for _ in range(10))


def test_ratio_one_returns_one_each_call():
    ratio = Ratio(1.0)
    assert [ratio() for _ in range(5)] == [1, 1, 1, 1, 1]


def test_ratio_two_returns_two_each_call():
    ratio = Ratio(2.0)
    assert [ratio() for _ in range(5)] == [2, 2, 2, 2, 2]


def test_ratio_fractional_accumulates_to_floor():
    # ratio=0.5 should yield 1 every other call on average
    ratio = Ratio(0.5)
    results = [ratio() for _ in range(10)]
    assert sum(results) == 5
    # never returns >1 with ratio < 1
    assert all(r in (0, 1) for r in results)


def test_ratio_no_drift_over_many_calls():
    # 0.3 isn't exactly representable in float, so allow a +/-1 slack from FP drift
    ratio = Ratio(0.3)
    total = sum(ratio() for _ in range(1000))
    assert abs(total - 300) <= 1
