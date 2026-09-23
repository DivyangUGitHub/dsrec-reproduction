import pandas as pd

from src.data.preprocessing import compute_statistics, remap_ids, sort_chronologically


def _toy_df() -> pd.DataFrame:
    # user 7 has two interactions with the SAME timestamp, in a deliberately
    # unsorted, non-chronological row order, to test both the stable-sort
    # and the remap in one small fixture.
    return pd.DataFrame(
        {
            "user_id": [42, 7, 7, 42, 7],
            "item_id": [500, 900, 100, 500, 900],
            "rating": [5, 3, 4, 5, 3],
            "timestamp": [20, 10, 10, 10, 30],
        }
    )


def test_sort_chronologically_orders_by_user_then_time():
    df = sort_chronologically(_toy_df())
    # user 7's rows come before user 42's (lower ID first from sort_values),
    # and within user 7, timestamp order is 10, 10, 30.
    assert list(df["user_id"]) == [7, 7, 7, 42, 42]
    assert list(df["timestamp"]) == [10, 10, 30, 10, 20]


def test_sort_chronologically_is_stable_on_ties():
    # Two rows share user_id=7, timestamp=10. Original file order was
    # item_id 900 then item_id 100 (rows at positions 1 and 2). A stable
    # sort must preserve that relative order.
    df = sort_chronologically(_toy_df())
    tied = df[(df["user_id"] == 7) & (df["timestamp"] == 10)]
    assert list(tied["item_id"]) == [900, 100]


def test_remap_ids_reserves_zero_and_is_contiguous():
    df = sort_chronologically(_toy_df())
    out, user_map, item_map = remap_ids(df)

  
    assert 0 not in user_map.internal_to_raw
    assert 0 not in item_map.internal_to_raw

    assert sorted(user_map.internal_to_raw.keys()) == list(range(1, len(user_map) + 1))
    assert sorted(item_map.internal_to_raw.keys()) == list(range(1, len(item_map) + 1))

    
    for internal, raw in user_map.internal_to_raw.items():
        assert user_map.raw_to_internal[raw] == internal


def test_remap_ids_is_applied_to_the_dataframe():
    df = sort_chronologically(_toy_df())
    out, user_map, item_map = remap_ids(df)
    
    raw_users_recovered = out["user_id"].map(user_map.internal_to_raw)
    assert list(raw_users_recovered) == list(df["user_id"])


def test_compute_statistics_matches_hand_calculation():
    # 2 users, 3 items, 5 interactions.
    df = pd.DataFrame({"user_id": [1, 1, 2, 2, 2], "item_id": [1, 2, 1, 2, 3]})
    stats = compute_statistics(df, n_users=2, n_items=3)
    assert stats["n_users"] == 2
    assert stats["n_items"] == 3
    assert stats["n_interactions"] == 5
    assert stats["avg_actions_per_user"] == 2.5
    assert stats["avg_actions_per_item"] == round(5 / 3, 1)
    assert stats["sparsity_pct"] == round((1 - 5 / 6) * 100, 3)
