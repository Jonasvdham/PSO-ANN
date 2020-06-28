# -*- coding: utf-8 -*-
import numpy as np


class ParticleSwarm(object):
    def __init__(
        self,
        cost_func,
        num_dimensions,
        num_particles,
        inertia,
        phi_p,
        phi_g,
        v_max,
    ):
        self.cost_func = cost_func
        self.num_dimensions = num_dimensions

        self.num_particles = num_particles
        self.inertia = inertia
        self.phi_p = phi_p
        self.phi_g = phi_g
        self.v_max = v_max

        # Initialize X and V -> uniform
        self.X = np.random.uniform(
            size=(self.num_particles, self.num_dimensions)
        )
        self.V = np.random.uniform(
            size=(self.num_particles, self.num_dimensions)
        )

        self.P = self.X.copy()
        self.neighbor_best = self.X.copy()
        self.S = self.cost_func(weights=self.X)
        self.g = self.P[self.S.argmin()]
        self.best_score = self.S.min()

    def update(self, num_iterations, current_iter):
        R_p = np.random.uniform(size=(self.num_particles, self.num_dimensions))
        R_g = np.random.uniform(size=(self.num_particles, self.num_dimensions))

        # Linearly decreasing inertia
        self.inertia = 0.9 - (0.5 / num_iterations) * self.inertia

        # V_max cycle
        if (current_iter % 10 == 0) and (self.v_max > 0.01):
            print(current_iter, self.v_max)
            self.v_max = self.v_max * 0.75
        elif (current_iter % 30 == 0) and (self.v_max <= 0.05):
            self.v_max = 2

        self.V = 0.7 * (
            self.inertia * self.V
            + (
                self.phi_p * R_p * (self.P - self.X)
                + self.phi_g * R_g * (self.neighbor_best - self.X)
            )
        )

        # Apply V_max and UB/LB on X
        self.X = np.clip(self.X, -1.5, 1.5)
        self.V = np.clip(self.V, -self.v_max, self.v_max)
        # Positions update
        self.X = self.X + self.V

        # Best scores
        scores = self.cost_func(weights=self.X)
        better_scores_idx = scores < self.S
        self.P[better_scores_idx] = self.X[better_scores_idx]
        self.S[better_scores_idx] = scores[better_scores_idx]

        # Ring topology
        neighbor_best = []
        for i in range(self.num_particles):
            neighbor_best.append(
                (
                    np.argmin(
                        [
                            self.S[i - 1],
                            self.S[i % self.num_particles],
                            self.S[(i + 1) % self.num_particles],
                        ]
                    )
                    + (i - 1)
                )
                % self.num_particles
            )

        self.neighbor_best = self.P[neighbor_best]
        self.g = self.P[self.S.argmin()]
        self.best_score = self.S.min()
