import pandas as pd

from backtest_pregame import evaluate_spread_bets, evaluate_totals_bets


def test_spread_bet_grading_with_push_and_cover():
    df = pd.DataFrame(
        {
            "game_date": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "pred_margin": [6.0, -7.0, 4.0],
            "actual_margin": [5.0, -4.0, 3.0],
            "market_spread_open": [-3.5, 2.5, -3.0],
            "market_spread_close": [-4.0, 3.0, -3.0],
        }
    )

    bets = evaluate_spread_bets(df, edge_threshold=1.0, stake=1.0)
    assert len(bets) == 3

    # row1: home edge, home covers 5 + (-4) = +1 => win
    assert bool(bets.iloc[0]["won"]) is True
    assert float(bets.iloc[0]["profit"]) > 0

    # row2: away edge, home cover delta -4 + 3 = -1 => away covers => win
    assert bool(bets.iloc[1]["won"]) is True
    assert float(bets.iloc[1]["profit"]) > 0

    # row3: push at 3 + (-3) = 0
    assert bool(bets.iloc[2]["push"]) is True
    assert float(bets.iloc[2]["profit"]) == 0.0


def test_totals_bet_grading_with_push():
    df = pd.DataFrame(
        {
            "game_date": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "pred_total": [230.0, 205.0, 220.0],
            "actual_total": [228.0, 200.0, 218.0],
            "market_total_open": [224.0, 204.0, 218.0],
            "market_total_close": [225.0, 203.0, 218.0],
        }
    )

    bets = evaluate_totals_bets(df, edge_threshold=1.0, stake=1.0)
    assert len(bets) == 3

    # row1: over bet wins
    assert bool(bets.iloc[0]["won"]) is True
    # row2: under bet wins
    assert bool(bets.iloc[1]["won"]) is True
    # row3: exact close total => push
    assert bool(bets.iloc[2]["push"]) is True
    assert float(bets.iloc[2]["profit"]) == 0.0
