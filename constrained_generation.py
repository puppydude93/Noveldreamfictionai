"""
Constrained generation algorithm that steers token-category frequencies
toward user-defined targets by adjusting model logits at each step.

Based on Algorithm 1: logit-bias constrained generation.
References are preserved from the original pseudocode [2, 3, 5, 6, 7].
"""

import math
import re
from dataclasses import dataclass, field
from typing import Optional

# Scaling factor: keeps |f_omega| bounded and readable [5].
OMEGA: float = 30.0

# Alpha calibrated so that |f_omega(x, alpha)| ≈ 0.99 * OMEGA when |x| = 1 [5].
# Derived from: OMEGA * tanh(alpha) = 0.99 * OMEGA  =>  alpha = atanh(0.99)
ALPHA: float = math.atanh(0.99)  # ≈ 2.647


def f_omega(x: float, alpha: float = ALPHA) -> float:
    """
    Scaled tanh transformation applied to the deviation delta [2].
    Bounds the logit adjustment to (-OMEGA, +OMEGA) to avoid
    distribution collapse while still steering generation.
    """
    return OMEGA * math.tanh(alpha * x)


def calculate_delta(f_star: float, f_c: float) -> float:
    """
    LSM-inspired deviation between desired frequency f_star and
    current frequency f_c [2, 6].

    Returns 0 when both frequencies are zero to avoid division by zero.
    Positive delta → category is under-represented → logits boosted.
    Negative delta → category is over-represented → logits suppressed.
    """
    denom = f_star + f_c
    if denom == 0.0:
        return 0.0
    return (f_star - f_c) / denom


@dataclass
class Category:
    """
    A token/word category with a target frequency and an optional
    human-readable label used in the UI.
    """
    name: str
    token_indices: set[int]
    f_star: float  # desired fraction of total tokens, in [0, 1]
    pattern: Optional[re.Pattern] = field(default=None, repr=False)


class ConstrainedGenerator:
    """
    Algorithm 1: Constrained generation [3].

    Given a language model that exposes raw logits, this class adjusts
    those logits at each generation step so that the frequency of tokens
    in each predefined category tracks its target frequency f_star.

    Usage::

        gen = ConstrainedGenerator(categories)
        tokens = gen.generate(lm, prompt_tokens, n_max=512)
    """

    def __init__(self, categories: list[Category], alpha: float = ALPHA):
        self.categories = categories
        self.alpha = alpha

    def generate(self, lm, tokens: list[int], n_max: int) -> list[int]:
        """
        Run constrained generation, returning the full token sequence
        including the initial prompt tokens [3].
        """
        tokens = list(tokens)
        n = len(tokens)

        while n < n_max:
            logits = lm.get_logits(tokens)

            for cat in self.categories:
                count = sum(1 for t in tokens if t in cat.token_indices)
                f_c = count / n
                delta = calculate_delta(cat.f_star, f_c)
                bias = f_omega(delta, self.alpha)

                # Apply the bias only to logits of tokens in this category [3, 7].
                for tok_idx in cat.token_indices:
                    if tok_idx < len(logits):
                        logits[tok_idx] += bias

            next_token = lm.sample(logits)
            tokens.append(next_token)
            n += 1

        return tokens


# ---------------------------------------------------------------------------
# Text-level frequency tracker (for use with opaque APIs such as Vertex AI)
# ---------------------------------------------------------------------------

# Patterns for fiction-relevant prose categories.
_CATEGORY_PATTERNS: dict[str, re.Pattern] = {
    "dialogue": re.compile(r'"[^"]*"|\'[^\']*\'|"[^"]*"', re.DOTALL),
    "action": re.compile(
        r"\b(ran|walked|grabbed|turned|jumped|fell|pushed|pulled|moved|rushed|"
        r"stepped|looked|glanced|reached|threw|caught|opened|closed|stood|sat|"
        r"rose|spun|lunged|dodged|struck|fled|chased|leapt)\b",
        re.IGNORECASE,
    ),
    "description": re.compile(
        r"\b(golden|dark|pale|bright|vast|narrow|ancient|glistening|crimson|"
        r"silent|soft|rough|cold|warm|heavy|light|towering|shadowed|dim|vivid|"
        r"murky|serene|jagged|twisted|smooth|gleaming)\b",
        re.IGNORECASE,
    ),
    "emotion": re.compile(
        r"\b(fear|joy|sorrow|hope|anger|grief|dread|love|hate|longing|despair|"
        r"relief|terror|wonder|shame|pride|guilt|jealous|anxious|calm|brave|"
        r"weary|trembled|smiled|wept|sighed|laughed|frowned|gasped)\b",
        re.IGNORECASE,
    ),
}


@dataclass
class FrequencyStats:
    category: str
    target: float
    actual: float
    delta: float
    bias: float


class TextFrequencyTracker:
    """
    Tracks prose-category frequencies in plain text (no logit access required).

    Used with closed APIs (Vertex AI, OpenAI, etc.) to measure how well the
    generated text meets frequency targets, and to produce prompt guidance
    for the next turn.
    """

    def __init__(self, targets: dict[str, float], alpha: float = ALPHA):
        """
        targets: mapping from category name to desired frequency in [0, 1],
                 expressed as fraction of total word tokens.
        """
        unknown = set(targets) - set(_CATEGORY_PATTERNS)
        if unknown:
            raise ValueError(f"Unknown categories: {unknown}. "
                             f"Valid: {list(_CATEGORY_PATTERNS)}")
        self.targets = targets
        self.alpha = alpha
        self._history: list[str] = []

    def record(self, text: str) -> None:
        self._history.append(text)

    def _combined_text(self) -> str:
        return " ".join(self._history)

    def _word_count(self, text: str) -> int:
        return max(len(text.split()), 1)

    def stats(self) -> list[FrequencyStats]:
        text = self._combined_text()
        n = self._word_count(text)
        result = []
        for cat, f_star in self.targets.items():
            matches = _CATEGORY_PATTERNS[cat].findall(text)
            # For dialogue, count words inside quotes; for others, count match tokens.
            if cat == "dialogue":
                match_words = sum(len(m.split()) for m in matches)
            else:
                match_words = len(matches)
            f_c = match_words / n
            delta = calculate_delta(f_star, f_c)
            bias = f_omega(delta, self.alpha)
            result.append(FrequencyStats(
                category=cat,
                target=f_star,
                actual=round(f_c, 4),
                delta=round(delta, 4),
                bias=round(bias, 2),
            ))
        return result

    def guidance_prompt(self) -> str:
        """
        Returns a natural-language instruction derived from current frequency
        stats, suitable for injecting into a chat system prompt.
        """
        lines = []
        for s in self.stats():
            if abs(s.delta) < 0.05:
                continue
            if s.delta > 0:
                lines.append(
                    f"- Use more {s.category} (currently {s.actual:.0%}, "
                    f"target {s.target:.0%})."
                )
            else:
                lines.append(
                    f"- Use less {s.category} (currently {s.actual:.0%}, "
                    f"target {s.target:.0%})."
                )
        if not lines:
            return ""
        return "Writing frequency guidance:\n" + "\n".join(lines)

    def reset(self) -> None:
        self._history.clear()
