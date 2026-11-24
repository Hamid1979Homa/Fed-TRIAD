"""Independent arithmetic fixtures and regression checks for replay integrity."""

import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import fedtriad_sim as sim
from check_against_reference import audit, differences, float_summary
from run_sweep import job, prepare_result_dir
from run_files import directory_lock


def scalar_steps(weights, x, y, batches, lr, ridge, g=None, lam=0.0):
    w = weights.tolist()
    for ids in batches:
        batch_errors = [[math.fsum(a * b for a, b in zip(row, x[i])) - y[i, j]
                         for j, row in enumerate(w)] for i in ids]
        grads = [[math.fsum(batch_errors[k][j] * x[i, h] for k, i in enumerate(ids)) / len(ids)
                  + ridge * w[j][h] + (lam * (w[j][h] - g[j, h]) if g is not None else 0)
                  for h in range(len(w[j]))] for j in range(len(w))]
        w = [[a - lr * b for a, b in zip(row, grad)] for row, grad in zip(w, grads)]
    return np.array(w)


def scalar_risk(w, stored):
    residuals = (math.fsum(a * b for a, b in zip(row, x)) - y[j]
                 for x, y in zip(stored["Ft"], stored["Mt"]) for j, row in enumerate(w))
    return math.fsum(e * e for e in residuals) / len(stored["Ft"])


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.cfg = dict(sim.CFG, T=3, tau=1, t0=2, d=2, q=2, n_path=8, n_gate=6,
                        n_test=18, E=2, E_g=2, batch=3)

    def test_first_round_personal_global_and_group_against_scalar_calculation(self):
        cfg, seed, m = self.cfg, 7, 4
        world = sim.World(seed, cfg)
        rng = np.random.default_rng(10000 + seed)
        clients = rng.choice(cfg["N"], size=m, replace=False)
        zero = np.zeros((cfg["q"], 2 * cfg["d"]))
        personal = [zero] * cfg["N"]
        updates = {name: [] for name in ("FedAvg", "CFL-static", "CFL-adaptive")}
        for i in clients:
            stored = world.data(0, i)
            for name in ("Local", *updates, "oracle"):
                batches = [rng.choice(len(stored["Yp"]), cfg["batch"], replace=False)
                           for _ in range(cfg["E"])]
                learned = scalar_steps(zero, stored["Fp"], stored["Yp"], batches, cfg["lr"], cfg["ridge"])
                if name == "Local":
                    personal[i] = learned
                elif name in updates:
                    updates[name].append(learned)
        expected_local = math.fsum(scalar_risk(w, world.data(0, i))
                                   for i, w in enumerate(personal)) / cfg["N"]
        traces = sim.run(seed, cfg, m)["excess"]
        self.assertAlmostEqual(traces["Local"][0], expected_local, places=12)
        for name, models in updates.items():
            mean = np.mean(models, axis=0)
            risk = math.fsum(scalar_risk(mean, world.data(0, i)) for i in range(cfg["N"])) / cfg["N"]
            self.assertAlmostEqual(traces[name][0], risk, places=12, msg=name)

    def test_ditto_matches_independent_proximal_gradient(self):
        x = np.array([[1.0, 2.0], [-1.0, 1.0], [0.0, 1.0], [3.0, -2.0]])
        y = np.array([[2.0, 0.0], [1.0, -1.0], [2.0, 1.0], [-2.0, 1.0]])
        w = np.array([[0.5, -0.2], [0.4, -0.5]])
        g = np.array([[0.3, 0.1], [-0.4, 0.6]])
        lam = 0.8
        r = np.random.default_rng(3)
        batches = [r.choice(4, self.cfg["batch"], replace=False) for _ in range(self.cfg["E"])]
        expected = scalar_steps(w, x, y, batches, self.cfg["lr"], self.cfg["ridge"], g, lam)
        actual = sim.ditto_sgd(w, g, x, y, self.cfg, np.random.default_rng(3), lam)
        np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)
        self.assertGreater(np.max(np.abs(actual - sim.local_sgd(w, x, y, self.cfg, np.random.default_rng(3)))),
                           1e-4)

    def test_gate_gradient_matches_finite_difference_on_held_out_data(self):
        cfg = dict(self.cfg, E_g=1)
        info = np.array([[1.0, 0.0, -2.0], [1.0, 1.0, 1.5]])
        predictions = np.array([[[0.8], [2.0], [-0.3]], [[0.7], [1.0], [1.5]]])
        labels = np.array([[0.1], [1.0]])
        params = np.array([[0.2, -0.3, 0.15], [0.3, 0.1, -0.2], [0.0, 0.2, -0.1]])

        def objective(p):
            sq = []
            for x, forecasts, label in zip(info, predictions, labels):
                weights = [math.exp(math.fsum(a * b for a, b in zip(row, x))) for row in p]
                prediction = math.fsum(a * f[0] for a, f in zip(weights, forecasts)) / math.fsum(weights)
                sq.append((prediction - label[0]) ** 2)
            return math.fsum(sq) / len(labels) + cfg["reg_g"] * math.fsum(p.ravel() ** 2) / 2

        gradient = np.zeros_like(params)
        for index in np.ndindex(params.shape):
            step = np.zeros_like(params)
            step[index] = 1e-6
            gradient[index] = (objective(params + step) - objective(params - step)) / (2e-6)
        fit = sim.gate_step("input", params, info, predictions, labels, np.ones(3), cfg)
        np.testing.assert_allclose((params - fit) / cfg["lr_g"], gradient, atol=2e-9, rtol=0)
        weights = sim.softmax_masked(np.array([[-1e10, 3.0, -1e10]]), np.array([1.0, 0.0, 1.0]))
        np.testing.assert_array_equal(weights, [[0.5, 0.0, 0.5]])

    def test_matched_cluster_batches_equalize_same_partition(self):
        cfg = dict(self.cfg, match_cluster_batches=True, E_g=0, s=0)
        matched = sim.run(3, cfg, 2)["excess"]
        default = sim.run(3, dict(cfg, match_cluster_batches=False), 2)["excess"]
        np.testing.assert_array_equal(matched["CFL-static"], matched["CFL-adaptive"])
        np.testing.assert_array_equal(matched["TRIAD"], matched["TRIAD-uniform"])
        np.testing.assert_array_equal(matched["FedAvg"], default["FedAvg"])

    def test_cluster_checks_with_sparse_or_identical_embeddings(self):
        labels = np.array([0, 0])
        z = np.zeros((2, 3))
        np.testing.assert_array_equal(sim.algorithm3(z, labels, 1, self.cfg, 3, initial=True)[0], labels)
        self.assertFalse(sim.algorithm3(np.zeros((8, 3)), np.zeros(8, dtype=int),
                                        1, self.cfg, 3, initial=True)[2])

    def test_hysteresis_controls_admission_of_a_better_partition(self):
        embeddings = np.array([[0.0], [0.1], [0.2], [10.0], [10.1], [10.2]])
        current = np.array([0, 1, 0, 1, 0, 1])
        accepted, _, changed = sim.algorithm3(embeddings, current, 2, self.cfg, 9)
        self.assertTrue(changed)
        self.assertEqual(sim.adjusted_rand_score(accepted, [0, 0, 0, 1, 1, 1]), 1.0)
        held, _, changed = sim.algorithm3(embeddings, current, 2, dict(self.cfg, tau_imp=2.0), 9)
        self.assertFalse(changed)
        np.testing.assert_array_equal(held, current)

    def test_cluster_transfer_uses_overlap_and_copies_weights(self):
        regime = sim.ClusterRegime("adaptive", 7, np.zeros((1, 2)))
        regime.labels = np.array([0, 0, 0, 0, 1, 1, 1])
        regime.models[0] = np.array([[2.0, 3.0]])
        regime.models[1] = np.array([[5.0, 7.0]])
        regime.relabel_and_transfer(np.array([10, 10, 10, 20, 30, 30, 30]))
        np.testing.assert_array_equal(regime.labels, [0, 0, 0, 2, 1, 1, 1])
        np.testing.assert_array_equal(regime.models[0], [[2.0, 3.0]])
        np.testing.assert_array_equal(regime.models[1], [[5.0, 7.0]])
        np.testing.assert_array_equal(regime.models[2], [[2.0, 3.0]])
        regime.models[2][0, 0] = 11.0
        self.assertEqual(regime.models[0][0, 0], 2.0)


class ReproductionTests(unittest.TestCase):
    def test_saved_tables_match_and_the_last_digit_is_a_failure(self):
        ref = json.loads((ROOT / "reference/numbers_reference.json").read_text())
        printed = json.loads((ROOT / "reference/paper_tables.json").read_text(encoding="utf-8"))
        self.assertTrue(audit(ref, ref, printed, "exact")["passed"])
        altered = copy.deepcopy(ref)
        altered["drift_m12"]["TRIAD"]["final"]["mean"] += 0.001
        report = audit(altered, ref, printed, "tables")
        self.assertFalse(report["passed"])
        self.assertEqual(report["mismatching_cells"], 1)
        self.assertFalse(audit(altered, ref, printed, "exact")["passed"])
        floats = float_summary()
        self.assertEqual(list(differences(float("nan"), 0.5, floats=floats))[0][1], "nonfinite")

    def test_cache_rejected_on_manifest_change_and_rebuilt_after_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with directory_lock(out):
                prepare_result_dir(out, ["drift_m12"], False)
                prepare_result_dir(out, ["drift_m12"], True)
                with self.assertRaisesRegex(ValueError, "Run identity differs"):
                    prepare_result_dir(out, ["drift_m30"], True)
                with patch.object(sim, "run", return_value={"ok": np.array([3])}) as run:
                    job(("drift_m12", 1, out))
                    job(("drift_m12", 1, out))
                    self.assertEqual(run.call_count, 1)
                    (out / "runs/drift_m12/seed1.npy").write_bytes(b"corrupt")
                    job(("drift_m12", 1, out))
                    self.assertEqual(run.call_count, 2)


if __name__ == "__main__":
    unittest.main()
