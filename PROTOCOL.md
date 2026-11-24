# Fed-TRIAD simulation protocol

## Data and seeds

The base experiment has 30 simulated clients, 200 rounds indexed from 0 to 199
and a change at round 100. The initial concepts of clients 0–11, 12–23 and
24–29 are 1, 2 and 3. In the base setting, clients 0–3 switch to concept 2 and
clients 4–7 switch to concept 4. Setting `s` partitions the first `s` clients
into `ceil(s/2)` and the rest for these two destinations. With `s=0` the
concept assignments persist and the empirical splits are independently redrawn
at the same phase boundary.

For a seed `b`, initialize a NumPy PCG64 generator with `b`. Draw the common
3-by-16 Gaussian coefficient matrix, the four concept deviations in order and
the private deviations in client order. For each phase in chronological order
and each client in index order, draw 60 path samples, 30 gate samples and 2,000
held-out inputs in the base case. For each split, draw the inputs and then the
label noise. The noise draw for the held-out split advances the generator and
only the conditional means are retained as evaluation targets. The features
are the eight input coordinates followed by the same coordinates multiplied by
`1[x_1 > 0]`. The gate sees `(1, 1[x_1 > 0], x_1)`. Reuse a client's
evaluation inputs across rounds within its phase.

The participation, path batches and end-of-run Gram-matrix diagnostic use a
second PCG64 generator initialized with `10000 + b`. Ditto uses an independent
generator initialized with `20000 + b`. Each round draws all participants
uniformly without replacement and processes them in the drawn order. For each
client, path minibatches are drawn in order for the personal model, the global
model and the fixed, adaptive and oracle cluster regimes. Each step samples
indices without replacement within its batch. The three cluster regimes
receive separate minibatch draws from this stream in the manuscript grid.
The `match_cluster_batches` control applies the same indices to all three
cluster regimes while advancing the original stream as usual. The common
global and personal trajectories are shared wherever those predictors are used.

## A round of training

1. Initialize the matrices and gate logits to zero. The fixed and adaptive
   regimes start with one cluster. The oracle starts with the known partition
   and switches to the true new partition at round 100. It initializes a newly
   appearing concept from the global model.
2. For each selected client, take five gradient steps of size 0.05 from each
   applicable model using minibatches of up to 16 path samples. The objective
   for a local update is mean half squared error plus
   `0.5 * 0.001 * squared_norm(weights)`. Fit the client's mixture on its
   separate gate split after updating the personal model and using the shared
   models the client received at the start of the round.
3. The learned gate uses ten steps of size 0.5 on mean squared mixture loss
   plus `0.5 * 0.01 * squared_norm(gate_params)`. Its inputs and active paths
   determine the logits. For constant routing the logits are per-client
   constants. Masked paths receive weight zero. Previously learned logits are
   carried into the next round.
4. The server averages the global deltas over participants, and cluster deltas
   over participating members of each cluster. The server step is 1. An
   unrepresented cluster retains its model.
5. From round 20, check every tenth round for adaptive reclustering. Take
   flattened personal coefficients from clients last seen at most 20 rounds
   ago. Score candidates using the Euclidean silhouette. Use scikit-learn
   k-means++ with Lloyd iterations, ten initializations, at most 300 iterations,
   tolerance 0.0001 and the experiment seed. At the first check consider 2
   through 6 clusters. Later consider the current count and its adjacent counts.
   Cap all candidates by `K_max=8`, the number of distinct embeddings and
   one less than the number of fresh embeddings. An unusable silhouette
   candidate is skipped. With equal scores, retain the first candidate in
   ascending order. For later checks accept an improvement strictly above
   `tau_imp=0.02` with a changed partition.
6. A stale client follows the new cluster with most fresh members of its old
   cluster. Ties in the counts go to the lowest sorted candidate label. If
   the old cluster has no fresh members, the stale client retains
   its old assignment through matching. Use a Hungarian assignment maximizing
   membership overlap to transfer existing models through SciPy's
   `linear_sum_assignment` on the negative overlap matrix. An unmatched new group
   copies the most-overlapping prior model. Evaluate after aggregation and
   matching, on all clients.

The exact 23-setting grid is in `src/run_sweep.py`. It includes the 22
sensitivity-table configurations and the separate scarce-data configuration.
The manuscript grid reuses seeds 1–10 across settings. The generator draws a
fresh dataset for each setting and seed. Within a paired comparison the data
and participating clients are common to all methods.

## Baselines and ablations

| Code | Implementation |
|---|---|
| Local | Personal SGD only. |
| FedAvg | Equal-weight aggregation of local deltas with equal data counts and uniform client sampling. The base server step is 1. [McMahan et al., AISTATS 2017](https://proceedings.mlr.press/v54/mcmahan17a.html) |
| Ditto | A persistent client model uses its local loss and the proximal gradient `lambda * (V - G)` to the received FedAvg model. The simulator maintains parallel personal models for `lambda ∈ {0.1, 1, 10}` and selects on the held-out gate split each time the client participates. The parallel fit and held-out selection specify the adaptation used here. [Li et al., ICML 2021](https://proceedings.mlr.press/v139/li21h.html) |
| PG-const | APFL-style constant personal/global mixture. Separate path training and a held-out gate give the constant-mixture control within this protocol. [Deng et al., 2020](https://arxiv.org/abs/2003.13461) |
| PG-gate | Flow-style per-input personal/global mixture. A masked softmax combines linear predictions in the same simulation. [Panchal et al., NeurIPS 2023](https://papers.neurips.cc/paper_files/paper/2023/hash/3baf4eeffad860ca9c54aeab632716b4-Abstract-Conference.html) |
| CFL-static / CFL-adaptive | Use only the cluster path under one-shot or periodic clustering, respectively. |
| PC-gate / TRIAD | Input-dependent softmax using personal and adaptive-cluster predictions, or those two predictions plus the global prediction. |
| TRIAD-uniform / TRIAD-const | Three paths with uniform weights or learned constant weights per client. |
| TRIAD-static / TRIAD-oracle | Three paths with a learned input gate under the fixed or known partition, respectively. |

## Evaluation

Excess risk is the mean squared distance to the generating conditional mean,
summed across output coordinates and averaged across held-out inputs and
clients. Table 3 reports the average over the final 20 rounds, cumulative
post-change excess risk relative to the oracle and the slope fitted over the
final 50 rounds. Across seeds the table reports the sample standard
deviation. Table 6 requires the risk to be strictly lower or higher on at
least 8 of 10 paired seeds to label a comparison. The strongest competitor is
selected by the lowest mean among Local, FedAvg, both cluster-only variants,
the personal/global and personal/cluster baselines and Ditto.

For Table 4, freeze the learned paths. Use 8 quantile bins of the first input,
with edges estimated from 100,000 standard normal draws. Fit residual second
moments from 4,000 instances per client and evaluate on 4,000 independent
instances. The simplex solver enumerates possible supports and uses a
`1e-12` diagonal stabilization for the linear systems. Report the fitted
quadratic-program gate, the best single path per bin, the learned gate, raw
inverse-second-moment weights and uniform weights. Also report the mean
normalized cross-path residual moment and the pointwise diversity bound
diagnostic. These computations specify the Monte Carlo procedure used for
the reported estimates.

Independent checks in `tests/test_protocol.py` calculate the first round's
personal and shared models with scalar arithmetic, check Ditto's proximal
term, and verify the gate gradient by finite differences. A reimplementation
that targets the archived finite-sample values must also match the generator
call order, minibatches, solver conventions and rounding environment.
