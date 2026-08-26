"""
Load, filter and label the raw Kaggle CSVs into the modelling dataset.

Output: one row per eligible NBA player-season (2000-2025) with a binary
label `y`, where y = 1 means the player was named an All-Star that season.

Every filtering choice in this module is a recorded decision. The numbered
references below (Decision 1, Decision 2, ...) point to
docs/DATA_DECISIONS.md, which holds the options that were considered and the
reasoning for each choice.
"""

import pandas as pd

# Basketball-Reference gives a traded player one row per team PLUS a combined
# row covering the whole season, whose team is "2TM", "3TM", etc. The data
# contains 2TM (1,461 rows), 3TM (97), 4TM (2) and 5TM (1), so we match the
# digit rather than hard-coding the two values the spec happened to name.
COMBINED_TEAM_PATTERN = r"^\d+TM$"


def load_raw(data_dir="data"):
    """
    Read the three CSVs exactly as downloaded, with no transformation.

    Loading is kept separate from cleaning so that every change made to the
    data afterwards happens in build_dataset(), where it is explicit,
    inspectable, and traceable to a recorded decision.

    Args:
        data_dir: directory containing the three Kaggle CSVs.

    Returns:
        (advanced, allstar, per100) — raw DataFrames.
        per100 is loaded for completeness but not consumed yet: whether any
        of its columns join the feature set is Decision 5, not yet made.
    """
    advanced = pd.read_csv(f"{data_dir}/Advanced.csv")
    allstar = pd.read_csv(f"{data_dir}/All-Star Selections.csv")
    per100 = pd.read_csv(f"{data_dir}/Per 100 Poss.csv")
    return advanced, allstar, per100


def build_dataset(advanced, allstar, *, season_min=2000, season_max=2025,
                  min_minutes=500, verbose=True):
    """
    Turn the raw frames into the modelling dataset.

    Steps, in order:
      1. Keep NBA rows only. The ABA ran a separate All-Star game under a
         different selection process; mixing the leagues would conflate two
         different labels.
      2. Keep seasons season_min..season_max (Decision 4: 2000-2025 — 1999
         had no All-Star Game because of the lockout, 2026 is incomplete,
         and older selection formats differ too much for the label to keep
         a constant meaning).
      3. Resolve traded players to their season-combined TM row (Decision 2).
      4. Join labels: y = 1 for every player-season named in
         All-Star Selections.csv (Decision 1 — includes injured selectees
         flagged `replaced`; the label means "was named an All-Star").
      5. Apply the eligibility threshold mp >= min_minutes (Decision 3),
         reporting by name any All-Star seasons this drops — never silently.

    The labels are joined BEFORE the eligibility filter (step 4 before 5),
    although the filter is conceptually independent, because we want to name
    the All-Stars the threshold excludes. Filtering first would make those
    rows invisible.

    Args:
        advanced: raw Advanced.csv frame — the feature source.
        allstar:  raw All-Star Selections.csv frame — the label source.
        season_min, season_max: inclusive season range to keep.
        min_minutes: eligibility threshold on total minutes played (mp).
        verbose: print the row-count funnel, dropped All-Stars, and the
                 final class balance.

    Returns:
        DataFrame, one row per eligible player-season: all Advanced.csv
        columns plus
          y        — 0/1 label, 1 = named an All-Star that season
          replaced — True only for originally selected All-Stars who were
                     later replaced (usually injury); False for everyone
                     else, including all non-All-Stars.
    """
    adv = advanced.copy()
    asg = allstar.copy()

    # ---- Steps 1 & 2: league and season filters (Decision 4) ----------------
    adv = adv[adv["lg"] == "NBA"]
    adv = adv[(adv["season"] >= season_min) & (adv["season"] <= season_max)]
    asg = asg[asg["lg"] == "NBA"]
    asg = asg[(asg["season"] >= season_min) & (asg["season"] <= season_max)]
    if verbose:
        print(f"NBA, seasons {season_min}-{season_max}: "
              f"{len(adv)} stat rows, {len(asg)} All-Star selections")

    # ---- Step 3: resolve multi-team seasons (Decision 2) --------------------
    # A traded player has one partial row per team plus a combined "2TM"-style
    # row. All-Star selection considers the whole season, so we keep the
    # combined row and drop the partial stints. Keeping both would count the
    # same season twice; keeping only stints would split it into fragments.
    is_combined_row = adv["team"].str.match(COMBINED_TEAM_PATTERN).fillna(False)

    # For each (player_id, season) group: does a combined row exist anywhere
    # in the group? True for every row of a traded player's season.
    group_has_combined = is_combined_row.groupby(
        [adv["player_id"], adv["season"]]).transform("any")

    # A partial stint is a row belonging to a traded player's season that is
    # not itself the combined row. These are the rows to drop.
    is_partial_stint = group_has_combined & ~is_combined_row
    adv = adv[~is_partial_stint]
    if verbose:
        print(f"After resolving multi-team seasons: {len(adv)} player-seasons "
              f"({int(is_partial_stint.sum())} partial stint rows dropped)")

    # Sanity checks for Decision 2. The real double-counting guard is
    # uniqueness: exactly one row per player-season.
    assert not adv.duplicated(subset=["player_id", "season"]).any(), \
        "duplicate (player_id, season) rows survived multi-team resolution"

    # A single team plays 82 games, so no single-team row can exceed that.
    # Combined rows CAN legitimately reach g=85: the two teams' schedules are
    # offset at trade time, so a traded player can appear in extra games.
    single_team = ~adv["team"].str.match(COMBINED_TEAM_PATTERN).fillna(False)
    assert adv.loc[single_team, "g"].max() <= 82, \
        "a single-team row has g > 82 — the data is not what we think it is"

    # ---- Step 4: join labels (Decision 1) -----------------------------------
    # Every row of All-Star Selections.csv counts as y=1, including the 12%
    # flagged `replaced` (originally selected, then replaced through injury —
    # see DATA_DECISIONS.md for why the flag means this and not "was a
    # replacement"). The file must be unique per player-season, otherwise the
    # left join below would multiply rows.
    assert not asg.duplicated(subset=["player_id", "season"]).any(), \
        "All-Star Selections.csv has duplicate (player_id, season) rows"

    labels = asg[["player_id", "season", "replaced"]].copy()
    labels["y"] = 1

    df = adv.merge(labels, on=["player_id", "season"], how="left")

    # Non-All-Stars matched nothing in the join, so their y and replaced are
    # NaN. Fill BEFORE casting: astype(bool) would turn NaN into True.
    df["y"] = df["y"].fillna(0).astype(int)
    df["replaced"] = df["replaced"].fillna(False).astype(bool)

    # Every label row must have found exactly one stats row. If any did not,
    # the player_id slugs disagree between the two files and that selection
    # would silently become a false negative — fail loudly instead.
    n_matched = int(df["y"].sum())
    if n_matched != len(labels):
        unmatched = asg.merge(adv[["player_id", "season"]],
                              on=["player_id", "season"],
                              how="left", indicator=True)
        unmatched = unmatched[unmatched["_merge"] == "left_only"]
        raise ValueError(
            "All-Star selections with no matching stats row:\n"
            + unmatched[["player", "player_id", "season"]].to_string(index=False))

    # ---- Step 5: eligibility threshold (Decision 3) -------------------------
    # mp >= 500 removes tiny-sample players whose advanced metrics are
    # unstable, and (verified) every row with a missing candidate feature.
    # Rows with missing mp also fail the comparison (NaN >= x is False) —
    # they are exactly the tiny-sample rows the threshold exists to remove.
    is_eligible = df["mp"] >= min_minutes

    # Decision 3's cost is that a handful of injury/reputation All-Star
    # seasons fall under the threshold. Report them by name — the decision
    # record promises this is never silent.
    dropped_allstars = df[~is_eligible & (df["y"] == 1)]
    if verbose and len(dropped_allstars) > 0:
        print(f"Eligibility mp >= {min_minutes} drops "
              f"{len(dropped_allstars)} All-Star season(s):")
        for _, row in dropped_allstars.iterrows():
            print(f"  - {row['player']} {row['season']}: g={row['g']}, "
                  f"mp={row['mp']:.0f}, replaced={row['replaced']}")

    df = df[is_eligible]

    # ---- Final shape and class balance --------------------------------------
    df = df.sort_values(["season", "player"]).reset_index(drop=True)
    if verbose:
        n_total = len(df)
        n_pos = int(df["y"].sum())
        print(f"Final dataset: {n_total} player-seasons, {n_pos} All-Stars "
              f"({100 * n_pos / n_total:.1f}% positive)")
    return df


if __name__ == "__main__":
    # Quick verification run: python -m src.data (from the repo root)
    advanced, allstar, per100 = load_raw("data")
    dataset = build_dataset(advanced, allstar)
