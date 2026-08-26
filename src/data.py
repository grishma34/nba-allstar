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


def load_team_summaries(data_dir="data"):
    """
    Read Team Summaries.csv — the fourth and final CSV, added by the
    Decision 5 amendment as the source of the team win-rate feature (it
    also served the Criterion C error analysis). Loaded separately from
    load_raw() so existing callers keep working.

    Args:
        data_dir: directory containing the Kaggle CSVs.

    Returns:
        Raw DataFrame, one row per team-season.
    """
    return pd.read_csv(f"{data_dir}/Team Summaries.csv")


def add_team_win_rate(df, team_summaries, advanced):
    """
    Attach each player-season's team win rate as a feature column.

        win_rate = w / (w + l), minutes-weighted across stints for traded
        players.

    Why minutes-weighted: a traded player's season (Decision 2 keeps the
    combined TM row) spans several teams, and voters saw the whole season —
    so each stint's team quality is weighted by the time actually spent
    there. For an untraded player the weighted average collapses to their
    team's plain win rate, so one code path covers everyone.

    Adopted as a feature by the Decision 5 amendment after a controlled
    experiment (log loss −9%, recall +0.10); the production × win-rate
    interaction was tested at the same time and rejected.

    Args:
        df: dataset from build_dataset() — one row per player-season.
        team_summaries: raw frame from load_team_summaries().
        advanced: raw Advanced.csv frame — needed for the per-team stint
                  rows that build_dataset() deliberately dropped, because
                  the weighting happens over stints.

    Returns:
        df with a `win_rate` column added. Raises if any player-season
        cannot be matched to a team — a silent NaN here would poison
        training (see features.select_features).
    """
    seasons = (df["season"].min(), df["season"].max())

    teams = team_summaries.copy()
    teams = teams[(teams["lg"] == "NBA") & teams["abbreviation"].notna()]
    teams = teams[teams["season"].between(*seasons)]
    teams["win_rate"] = teams["w"] / (teams["w"] + teams["l"])
    team_rates = teams[["season", "abbreviation", "win_rate"]]

    # Per-team stint rows (the combined TM rows carry no team identity).
    stints = advanced[(advanced["lg"] == "NBA")
                      & advanced["season"].between(*seasons)]
    is_combined = stints["team"].str.match(COMBINED_TEAM_PATTERN).fillna(False)
    stints = stints[~is_combined]

    stints = stints.merge(team_rates, left_on=["season", "team"],
                          right_on=["season", "abbreviation"], how="left")
    assert not stints["win_rate"].isna().any(), (
        "a stint team abbreviation has no Team Summaries row — "
        "the two files disagree about team codes")

    # Minutes-weighted mean win rate per player-season:
    #   Σ(win_rate · mp) / Σ(mp) over that player's stints.
    stints = stints.assign(weighted=stints["win_rate"] * stints["mp"])
    per_player = stints.groupby(["player_id", "season"]).agg(
        weighted_sum=("weighted", "sum"), minutes_sum=("mp", "sum"))
    per_player["win_rate"] = (per_player["weighted_sum"]
                              / per_player["minutes_sum"])

    out = df.merge(per_player[["win_rate"]], on=["player_id", "season"],
                   how="left")
    assert not out["win_rate"].isna().any(), (
        "a player-season has no win rate — stint rows missing or key mismatch")
    return out


if __name__ == "__main__":
    # Quick verification run: python -m src.data (from the repo root)
    advanced, allstar, per100 = load_raw("data")
    dataset = build_dataset(advanced, allstar)
    dataset = add_team_win_rate(dataset, load_team_summaries("data"), advanced)
    print(f"win_rate attached: min {dataset['win_rate'].min():.3f}, "
          f"max {dataset['win_rate'].max():.3f}, no NaNs")
