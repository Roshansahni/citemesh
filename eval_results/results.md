| Config | Runs | Cited-claim rate | Claim support (lexical) | Citation validity | Evidence used | Distinct domains | Avg sec | Avg tokens |
|---|---|---|---|---|---|---|---|---|
| baseline | 3 | 0% | 0% | n/a | 0% | 0.0 | 33 | 719 |
| no_revision | 3 | 21% | 15% | 100% | 36% | 1.0 | 111 | 5425 |
| full | 3 | 33% | 20% | 100% | 18% | 1.3 | 163 | 9463 |

_Quality rows use answerable questions where the system did not refuse. Baseline has no evidence list, so its claims are untraceable by construction._

| Config | Correct refusals (unanswerable Qs) | False refusals (answerable Qs) |
|---|---|---|
| no_revision | 0/0 | 0/3 |
| full | 0/0 | 0/3 |
