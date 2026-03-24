"""Tests for electrochemical sub-model."""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "v1_baseline"))

import numpy as np
import pytest
from models.electrolyzer import Electrolyzer


@pytest.fixture
def ely():
    return Electrolyzer()


class TestVoltageModel:
    def test_V_rev_at_25C(self, ely):
        assert abs(ely.V_rev(298.15) - 1.229) < 1e-6

    def test_V_rev_at_80C(self, ely):
        V = ely.V_rev(353.15)
        expected = 1.229 - 8.5e-4 * (353.15 - 298.15)
        assert abs(V - expected) < 1e-6

    def test_V_ohm_positive(self, ely):
        assert ely.V_ohm(353.15, 300.0) > 0

    def test_V_act_positive(self, ely):
        assert ely.V_act(353.15, 300.0) > 0

    def test_V_cell_reasonable(self, ely):
        V = ely.V_cell(353.15, 300.0)
        assert 1.5 < V < 2.5

    def test_V_stack_is_N_times_V_cell(self, ely):
        T, I = 353.15, 300.0
        assert abs(ely.V_stack(T, I) - ely.p.N_cells * ely.V_cell(T, I)) < 1e-10


class TestFaradaicEfficiency:
    def test_eta_F_in_range(self, ely):
        for I in [50.0, 100.0, 200.0, 300.0, 500.0]:
            eta = ely.eta_F(I)
            assert 0 < eta <= 1.0

    def test_eta_F_increases_with_current(self, ely):
        assert ely.eta_F(100.0) < ely.eta_F(300.0)


class TestH2Production:
    def test_n_dot_H2_positive(self, ely):
        assert ely.n_dot_H2(353.15, 300.0) > 0

    def test_n_dot_H2_increases_with_current(self, ely):
        assert ely.n_dot_H2(353.15, 100.0) < ely.n_dot_H2(353.15, 400.0)


class TestStep:
    def test_step_returns_tuple(self, ely):
        assert len(ely.step(300.0, 353.15, 30.0)) == 5

    def test_step_clamps_current(self, ely):
        V1, *_ = ely.step(10.0, 353.15, 30.0)
        V2, *_ = ely.step(ely.p.I_min, 353.15, 30.0)
        assert abs(V1 - V2) < 1e-10

    def test_energy_conservation(self, ely):
        V, P, n, eta, sec = ely.step(300.0, 353.15, 30.0)
        assert abs(P - V * 300.0) < 1e-6
