import numpy as np

# Mortal action space layout (from consts.rs):
# Indices 0-36:  discard (37 tiles: 34 normal + 3 aka) / kan choice
# Index 37:      riichi
# Indices 38-40: chi (3 variants)
# Index 41:      pon
# Index 42:      kan (decide)
# Index 43:      agari (tsumo/ron)
# Index 44:      ryukyoku (abortive draw)
# Index 45:      pass
ACTION_SPACE = 46

# Special action indices (non-discard)
IDX_RIICHI = 37
IDX_CHI_LOW = 38
IDX_CHI_MID = 39
IDX_CHI_HIGH = 40
IDX_PON = 41
IDX_KAN = 42
IDX_AGARI = 43
IDX_RYUKYOKU = 44
IDX_PASS = 45


def compute_criticality(state, q_values=None, masks=None):
    """Compute decision criticality score in [0, 1].

    Higher score = more important decision = warrants more search budget.

    Uses observable features from PlayerState (via its public getters)
    plus optionally the q_values from the policy network.

    Args:
        state: A libriichi.state.PlayerState instance.
        q_values: Optional numpy array of shape (ACTION_SPACE,) with
                  q-values from the DQN. Used for entropy computation.
                  If None, entropy factor is skipped.
        masks: Optional numpy array of shape (ACTION_SPACE,) with
               legal action mask (True = legal). If None, derived from
               q_values (nonzero entries).

    Returns:
        Float in [0, 1] representing decision criticality.
    """
    score = 0.0

    # --- Factor 1: Opponent riichi (major threat indicator) ---
    riichi = state.riichi_accepted()  # [bool; 4], index 0 = self
    riichi_count = sum(1 for i in range(1, 4) if riichi[i])
    if riichi_count >= 2:
        score += 0.35
    elif riichi_count == 1:
        score += 0.25

    # --- Factor 2: Our hand state (tenpai or close) ---
    shanten = state.shanten
    if shanten == 0:
        score += 0.20  # Tenpai: riichi/damaten decision is critical
    elif shanten == 1:
        score += 0.10  # Iishanten: approaching tenpai

    # --- Factor 3: Policy uncertainty (entropy of q-values) ---
    if q_values is not None:
        score += _compute_entropy_factor(q_values, masks)

    # --- Factor 4: Game phase ---
    # state.kyoku is within-wind (0-3). Compute absolute kyoku using bakaze.
    # bakaze is a tile ID: E=27, S=28, W=29. Absolute kyoku = (bakaze-27)*4 + kyoku.
    bakaze = state.bakaze  # u8 tile ID
    abs_kyoku = (bakaze - 27) * 4 + state.kyoku
    if abs_kyoku >= 4:  # South round or later
        score += 0.10
    if abs_kyoku == 7:  # All-Last (South 4)
        score += 0.10

    # --- Factor 5: Score proximity (close placement battle) ---
    scores = state.scores()  # Relative: scores[0] = self
    our_score = scores[0]

    # Check if we're in 2nd or 3rd and close to adjacent placement
    all_scores = sorted(scores, reverse=True)
    our_rank = all_scores.index(our_score)  # 0 = 1st

    if our_rank in (1, 2):
        # Gap to player above us
        gap_above = all_scores[our_rank - 1] - our_score
        # Gap to player below us
        gap_below = our_score - all_scores[our_rank + 1] if our_rank < 3 else 99999
        if min(gap_above, gap_below) < 8000:
            score += 0.10

    # --- Factor 6: Dangerous discard situation ---
    # Heuristic: if opponent(s) in riichi and we have tiles to discard,
    # and we're not tenpai ourselves, discarding is dangerous.
    cans = state.last_cans
    if cans.can_discard and riichi_count > 0 and shanten > 0:
        score += 0.15

    return min(score, 1.0)


def _compute_entropy_factor(q_values, masks=None):
    """Compute normalized entropy of the policy distribution.

    Converts q-values to a probability distribution via softmax,
    then computes normalized entropy as a measure of uncertainty.

    Returns a value in [0, 0.15] to add to criticality score.
    """
    q = np.asarray(q_values, dtype=np.float64)

    if masks is not None:
        mask = np.asarray(masks, dtype=bool)
    else:
        # Infer mask from finite q-values
        mask = np.isfinite(q) & (q > -1e30)

    n_legal = np.sum(mask)
    if n_legal <= 1:
        return 0.0  # Only one legal action, no uncertainty

    # Softmax over legal actions only
    legal_q = q[mask]
    legal_q = legal_q - legal_q.max()  # Numerical stability
    exp_q = np.exp(legal_q)
    probs = exp_q / exp_q.sum()

    # Shannon entropy
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    max_entropy = np.log(n_legal)

    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    return 0.15 * normalized_entropy
