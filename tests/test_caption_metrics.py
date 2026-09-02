"""The captioning metrics, implemented in-repo rather than imported.

Hand-rolled BLEU and ROUGE-L are easy to get subtly wrong in ways that do not
raise -- an off-by-one in the n-gram window or a flipped precision/recall gives
plausible-looking numbers that are simply false. These pin the properties that
must hold.
"""
import math

from eval.vrsbench_caption import (
    bleu, content_recall, ngrams, rouge_l, tokens,
)


class TestTokens:
    def test_lowercases_and_drops_punctuation(self):
        assert tokens("A Forest, near water.") == ["a", "forest", "near", "water"]

    def test_empty_input(self):
        assert tokens("") == []


class TestNgrams:
    def test_counts_overlapping_windows(self):
        assert ngrams(["a", "b", "c"], 2) == {("a", "b"): 1, ("b", "c"): 1}

    def test_window_longer_than_sequence_is_empty(self):
        assert ngrams(["a"], 3) == {}


class TestBleu:
    def test_identical_sequences_score_one(self):
        seq = tokens("a forest beside a river")
        assert bleu(seq, seq, 4) == 1.0

    def test_disjoint_sequences_score_zero(self):
        assert bleu(tokens("alpha beta gamma delta"),
                    tokens("one two three four"), 4) == 0.0

    def test_empty_candidate_scores_zero(self):
        assert bleu([], tokens("a forest"), 4) == 0.0

    def test_score_is_bounded(self):
        c, r = tokens("forest near the water"), tokens("a large forest beside water")
        assert 0.0 <= bleu(c, r, 4) <= 1.0

    def test_brevity_penalty_punishes_short_candidates(self):
        ref = tokens("a b c d e f g h")
        short = bleu(tokens("a b"), ref, 1)
        full = bleu(ref, ref, 1)
        assert short < full


class TestRougeL:
    def test_identical_sequences_score_one(self):
        seq = tokens("urban fabric and arable land")
        assert rouge_l(seq, seq) == 1.0

    def test_disjoint_sequences_score_zero(self):
        assert rouge_l(tokens("alpha beta"), tokens("gamma delta")) == 0.0

    def test_rewards_subsequences_not_just_contiguous_runs(self):
        """LCS must credit order-preserving gaps, which is the point of it."""
        assert rouge_l(tokens("a c e"), tokens("a b c d e")) > 0.5

    def test_is_symmetric_in_f_measure(self):
        a, b = tokens("a b c d"), tokens("a x c y")
        assert math.isclose(rouge_l(a, b), rouge_l(b, a), rel_tol=1e-9)

    def test_empty_input_scores_zero(self):
        assert rouge_l([], tokens("a b")) == 0.0


class TestContentRecall:
    def test_ignores_stopwords_and_scene_boilerplate(self):
        """Words every caption contains would inflate the score for free."""
        cand = tokens("This satellite image shows forest")
        ref = tokens("A photo of dense woodland")
        # "satellite", "image", "shows" are boilerplate; only "forest" counts,
        # and it is absent from the reference.
        assert content_recall(cand, ref) == 0.0

    def test_matching_content_word_scores(self):
        assert content_recall(tokens("forest"), tokens("a dense forest here")) == 1.0

    def test_empty_candidate_scores_zero(self):
        assert content_recall([], tokens("forest")) == 0.0

    def test_is_bounded(self):
        c, r = tokens("forest water urban"), tokens("forest and water")
        assert 0.0 <= content_recall(c, r) <= 1.0
