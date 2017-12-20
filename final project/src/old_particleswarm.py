import csv
import os
from sklearn.linear_model import LinearRegression
from datetime import datetime as dt
import numpy as np
import copy
import pandas as pd
from tqdm import tqdm

class Parameters(object):
    def __init__(self, names, constraints):
        self.__param = dict()
        self.__param['names'] = names

        low_params = [i[0] for i in constraints]
        high_params = [i[1] for i in constraints]
        self.__param['param'] = np.random.uniform(low_params, high_params)

        constraints = np.array(constraints, dtype=np.float64)

        self.__param['constraints'] = constraints

    def get_params(self):
        return self.__param['param']

    def get_constraints(self):
        return self.__param['constraints']

    def get_constraint(self, name):
        pos_index = self.__param['names'].index(name)

        return self.__param['constraints'][pos_index]

    def get_names(self):
        return self.__param['names']

    def get_param(self, name):
        pos_index = self.__param['names'].index(name)

        return self.__param['param'][pos_index]

    def set_param(self, name, value):
        pos_index = self.__param['names'].index(name)

        value = np.float64(value)

        if value > self.__param['constraints'][pos_index]:
            raise ValueError('new value must be less than upper bound')

        if value < self.__param['constraints'][pos_index]:
            raise ValueError('new value must be greater than lower bound')

        self.__param['param'][pos_index] = value

    def get_copy(self):
        return copy.deepcopy(self)


class ParticleSwarm(object):
    def __init__(self, size, volatility, pov, cost, maxiter=100,
                 nparticles=500, inertia_weight=0.5, c1=1, c2=2):
        self.__swarm = [Particle(size, volatility, pov, cost, inertia_weight,
                                 c1, c2)
                        for _ in range(nparticles)]
        self.__nparticles = nparticles

        self.__global_inertia_weight = inertia_weight
        self.__global_c1 = c1
        self.__global_c2 = c2

        self.__best_position = None
        self.__best_error = None
        self.__best_particle_index = None

        self.__all_best_position = list()
        self.__all_best_error = list()
        self.__all_best_particle_index = list()

        self.__set_the_best()

        self.__itercounter = 0
        self.__maxiter = maxiter

    def evaluate(self, size, volatility, pov, cost):
        for i in tqdm(range(self.__maxiter), total=self.__maxiter, ncols=80):
            for a_particle in self.__swarm:
                a_particle.evaluate(size, volatility, pov, cost,
                                    self.__best_position)

            self.__all_best_position.append(self.get_best_position())
            self.__all_best_error.append(self.get_best_error())
            self.__all_best_particle_index.append(self.get_best_particle())

            self.__itercounter += 1

        self.__set_the_best()

    def get_best_position(self):
        return self.__best_position

    def get_position_list(self):
        return self.__all_best_position

    def get_best_error(self):
        return self.__best_error

    def get_error_list(self):
        return self.__all_best_error

    def get_best_particle(self):
        return self.__swarm[self.__best_particle_index]

    def get_particle_list(self):
        return [self.__swarm[i] for i in self.__all_best_particle_index]

    def get_total_runs(self):
        return self.__itercounter

    def __set_the_best(self):
        best_error = self.__swarm[0].get_best_error()
        best_error_index = 0
        best_position = self.__swarm[0].get_best_position()

        for i in range(1, len(self.__swarm)):
            particle_error = self.__swarm[i].get_best_error()

            if particle_error < best_error:
                best_error = particle_error
                best_error_index = i
                best_position = self.__swarm[i].get_best_position()

        self.__best_position = best_position
        self.__best_error = best_error
        self.__best_particle_index = best_error_index


class Particle(object):
    def __init__(self, size, volatility, pov, cost, inertia_weight=0.5, c1=1,
                 c2=2):
        self.costfun = estcost
        self.params = Parameters(['a1', 'a2', 'a3', 'a4', 'b1'],
                                 [[0.0001, 2000], [0.0001, 1], [0.0001, 1],
                                  [0.0001, 1], [0.5, 1]])

        self.ndim = len(self.params.get_params())
        self.position = self.params.get_copy().get_params()
        self.velocity = Parameters(['a1', 'a2', 'a3', 'a4', 'b1'],
                                   [[0.0001, 2000], [0.0001, 1], [0.0001, 1],
                                    [0.0001, 1], [0.5, 1]])
        self.velocity = self.velocity.get_params()

        current_error = self.compute_error(size, volatility, pov, cost)
        self.best_position = {
            'error': current_error,
            'position': self.position
        }

        self.current_error = current_error

        self.inertia_weight = inertia_weight
        self.c1 = c1
        self.c2 = c2

    def __create_param_vector(self, a1, a2, a3, a4, b1):
        r = np.array([a1, a2, a3, a4, b1], dtype=np.float64)

        con = self.params.get_constraint('a1')
        r[0] = con[0] if r[0] < con[0] else r[0]
        r[0] = con[1] if r[0] > con[1] else r[0]

        con = self.params.get_constraint('a2')
        r[1] = con[0] if r[1] < con[0] else r[1]
        r[1] = con[1] if r[1] > con[1] else r[1]

        con = self.params.get_constraint('a3')
        r[2] = con[0] if r[2] < con[0] else r[2]
        r[2] = con[1] if r[2] > con[1] else r[2]

        con = self.params.get_constraint('a4')
        r[3] = con[0] if r[3] < con[0] else r[3]
        r[3] = con[1] if r[3] > con[1] else r[3]

        con = self.params.get_constraint('b1')
        r[4] = con[0] if r[4] < con[0] else r[4]
        r[4] = con[1] if r[4] > con[1] else r[4]

        return r

    def set_position(self, a1, a2, a3, a4, b1):
        self.position = self.__create_param_vector(a1, a2, a3, a4, b1)

    def set_velocity(self, a1, a2, a3, a4, b1):
        self.velocity = np.array([a1, a2, a3, a4, b1], dtype=np.float64)

    def compute_error(self, size, volatility, pov, cost):
        error = self.costfun(self.position[0], self.position[1],
                             self.position[2], self.position[3],
                             self.position[4], size, volatility, pov)

        error = np.sum(np.power((error - cost), 2))

        return error

    def __compute_velocity(self, bspos):
        r1 = np.random.uniform()
        r2 = np.random.uniform()

        social_term = self.c2 * r2 * (bspos - self.position)
        cognitive_term = self.c1 * r1
        cognitive_term *= (self.best_position['position'] - self.position)

        v = self.inertia_weight * self.velocity + social_term + cognitive_term

        return v

    def evaluate(self, size, volatility, pov, cost, bspos):
        position_error = self.compute_error(size, volatility, pov, cost)

        if position_error <= self.current_error:
            self.best_position = {
                'error': position_error,
                'position': self.position
            }

        v = self.__compute_velocity(bspos)
        self.set_velocity(v[0], v[1], v[2], v[3], v[4])
        p = self.position + self.velocity
        self.set_position(p[0], p[1], p[2], p[3], p[4])

    def get_best_position(self):
        return self.best_position['position']

    def get_best_error(self):
        return self.best_position['error']

    def get_best_position_error(self):
        return self.best_position

    def get_position(self):
        return self.position

    def get_error(self):
        return self.current_error

    def get_velocity(self):
        return self.velocity

def estcost(a1, a2, a3, a4, b1, size, volatility, pov):
    istar = a1 * (np.power(size, a2)) * (np.power(volatility, a3))

    result = (b1 * istar * (np.power(pov, a4))) + (1 - b1)

    return result
