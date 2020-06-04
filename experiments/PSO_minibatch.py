# -*- coding: utf-8 -*-
import numpy as np

class ParticleSwarm(object):
    def __init__(self, cost_func, num_dimensions, num_particles, inertia=0.7, phi_p=1, phi_g=3, v_max=3, batch_size=5):
        self.cost_func = cost_func
        self.num_dimensions = num_dimensions

        self.num_particles = num_particles
        self.inertia = inertia
        self.phi_p = phi_p
        self.phi_g = phi_g
        self.v_max = v_max
        self.batch_size = batch_size
        
        # Initialize X and V -> uniform
        self.X = np.random.uniform(size=(self.num_particles, self.num_dimensions))
        self.V = np.random.uniform(size=(self.num_particles, self.num_dimensions))

        # Store velocities for 
        self.stored_velocities = np.zeros((self.batch_size, self.num_particles, self.num_dimensions))
        
        self.P = self.X.copy()
        self.neighbor_best = self.X.copy()
        self.S = self.cost_func(weights=self.X)
        self.g = self.P[self.S.argmin()]
        self.best_score = self.S.min()


    def update(self, num_iterations, current_iter):
        R_p = np.random.uniform(size=(self.num_particles, self.num_dimensions))
        R_g = np.random.uniform(size=(self.num_particles, self.num_dimensions))

        # Linearly decreasing inertia
        self.inertia = 0.9 - (0.5/(1+current_iter))*self.inertia
        
        # V_max cycle
        if ((current_iter % 10 == 0) and (self.v_max > 0.5)):
            print(current_iter, "velocity", self.v_max)
            self.v_max = self.v_max*0.9
        elif ((current_iter % 10 == 0) and (self.v_max <= 0.5)):
            self.v_max = 2
        
        # Store velocities for minibatch
        self.stored_velocities[current_iter%self.batch_size] = 0.7*(self.inertia * self.V + (self.phi_p * R_p * (self.P - self.X) +
                                                                             self.phi_g * R_g * (self.neighbor_best - self.X)))
       
    def apply_batch(self):       
        # Apply V_max and UB/LB on X
        self.X = np.clip(self.X, -1.5, 1.5)
        
        # Possibly change order, first average then apply v_max
        self.stored_velocities = np.clip(self.stored_velocities, -self.v_max, self.v_max)
        
        for i in range(self.batch_size):
            # Positions update
            new_pos = self.X + self.stored_velocities[i]

            # Check for improvements
            scores = self.cost_func(weights=new_pos) 
            worse_scores_idx = scores > self.S
         
            print("Worse velocity updates:", sum(worse_scores_idx), "/ ", self.num_particles)
            self.stored_velocities[i][worse_scores_idx] = np.zeros(self.num_dimensions)
        
        
        print(self.stored_velocities.T[0][0])
        # Average of improving velocities
        self.V = np.mean(self.stored_velocities, axis=0)
        
        # Apply v_max
        self.V = np.clip(self.V, -self.v_max, self.v_max)
        # Apply to X
        self.X = self.X + self.V
        
        # Check score improvements again
        scores = self.cost_func(weights=self.X) 
        better_scores_idx = scores < self.S
        print("Eventual improved scores:", sum(better_scores_idx))
        self.P[better_scores_idx] = self.X[better_scores_idx]
        self.S[better_scores_idx] = scores[better_scores_idx]
        
        # Ring topology
        neighbor_best=[]
        for i in range(self.num_particles):
             neighbor_best.append((np.argmin([self.S[i-1], 
                                              self.S[i%self.num_particles],
                                              self.S[(i+1)%self.num_particles]]) + (i-1))
                                  % self.num_particles)
            
        self.neighbor_best = self.P[neighbor_best]
        self.g = self.P[self.S.argmin()]
        self.best_score = self.S.min()