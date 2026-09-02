# Yes/no decision rules on RSVQA-LR presence questions

133 scene-level presence questions, balanced (66 yes / 67 no), so the majority
baseline is 50.4%.

| rule | accuracy | says "yes" |
|---|---|---|
| **A — target ranks top-3 and mass > prior** (adopted) | **55.7%** | 18% |
| C — softmax mass > uniform prior | 55.7% | 18% |
| D — contrastive paired prompts ("containing X" vs "with no X") | 48.9% | 27% |
| B — target ranks top-7 | 38.9% | 64% |
| E — target similarity > vocabulary median | 36.6% | 79% |

## Reading this

The adopted rule wins, but by roughly five points over guessing, which is a
weak effect and is reported as such.

The informative result is B and E. Both say "yes" often, and both score *below
chance*. The backbone's ranking is therefore anti-correlated with RSVQA's
notion of presence: adaptation to BigEarthNet's CORINE land-cover vocabulary on
120x120 Sentinel-2 does not transfer to RSVQA-LR's object-centric annotation of
the same sensor family.

This is the same distribution-shift finding that shaped the backbone choice:
remote-sensing adaptation is corpus-specific, not a single transferable
property. Closing this would need adaptation on RSVQA itself, not a better
decision rule on top of the current one — which is why no further rule tuning
was attempted.
