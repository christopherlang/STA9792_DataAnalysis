import numpy as np
import pdb



class ParticleSwarm(object):
    def __init__(self, size, volatility, pov, cost, nparticles=500,
                 inertia_weight=0.5, c1=1, c2=2):
        self.w0 = inertia_weight
        self.c1 = c1
        self.c2 = c2
        self.nparticles = nparticles

        self.size = size
        self.volatility = volatility
        self.pov = pov
        self.cost = cost

        self.constraints = [[0.0001, 2000],
                            [0.0001, 1],
                            [0.0001, 1],
                            [0.0001, 1],
                            [0.5, 1]]
        self.constraints = np.array(self.constraints)

        nseq = range(self.nparticles)
        self.min_const = np.array([self.constraints[:, 0] for i in nseq])
        self.max_const = np.array([self.constraints[:, 1] for i in nseq])

        self.sw_p = [self.randparam() for _ in range(self.nparticles)]
        self.sw_p = np.array(self.sw_p, dtype=np.float64)

        self.sw_v = [self.randparam() for _ in range(self.nparticles)]
        self.sw_v = np.array(self.sw_p, dtype=np.float64)

        self.nparams = len(self.sw_v[0])

        poserrors = self.__compute_position_cost()
        poserrors = self.__compute_error(poserrors)

        best_index = self.__best_index(poserrors)

        self.gbspos = self.sw_p[best_index]
        self.gbserr = poserrors[best_index]

        self.gbsposl = list()
        self.gbserrl = list()

        self.__update_pos_list(self.gbspos)
        self.__update_err_list(self.gbserr)

        self.itercount = 1

    def get_best_error(self):
        return self.gbserr

    def get_best_position(self):
        return self.gbspos

    def get_error_list(self):
        return self.gbserrl

    def get_position_list(self):
        return self.gbsposl

    def randparam(self):
        constraints = np.array(self.constraints, dtype=np.float64)

        params = np.random.uniform(constraints[:, 0], constraints[:, 1])

        return params

    def __update_pos_list(self, pos):
        self.gbsposl.append(pos)

    def __update_err_list(self, err):
        self.gbserrl.append(err)

    def __best_index(self, errors):
        return np.argmin(errors)

    def evaluate_fitness(self, maxiter=1000, thres=0.001):
        eval_count = 0
        should_stop = False  # because threshold was reached before maxiter

        print("")
        print("Particle Swarm Optimization run")
        print("-------------------------------")
        msg = "Running fitness algorithms on {0} particles, {1} max iterations"
        print(msg.format(self.nparticles, maxiter))

        while eval_count < maxiter:
            if eval_count % 100 == 0:
                msg = "{0} evaluations completed"
                print(msg.format(eval_count))

            # For all particles, compute new velocities and positions
            nw_v = self.__compute_new_velocities()
            nw_p = self.__compute_new_positions(nw_v)

            # Set new velocities and positions for later iterations
            self.__set_positions(nw_p)
            self.__set_velocities(nw_v)

            # Estimate error for getting "cost"
            nw_err = self.__compute_position_cost()
            nw_err = self.__compute_error(nw_err)

            best_index = self.__best_index(nw_err)
            best_error = nw_err[best_index]
            best_pos = nw_p[best_index]

            # Check if new best error is better than before
            if best_error < self.gbserr:
                print("New position with minimum error found")
                msg = "old error: {0}, new error: {0}"
                print(msg.format(self.gbserr, best_error))

                should_stop = True

                self.gbserr = best_error
                self.gbspos = best_pos

                print(self.__pp_parameters(self.gbspos))

            self.__update_err_list(self.gbserr)
            self.__update_pos_list(self.gbspos)

            eval_count += 1
            self.itercount += 1

            if should_stop is True:
                msg = "Threshold reach. {0} iterations completed, out of {1}"
                msg = msg.format(eval_count, maxiter)
                print(self.__pp_parameters(self.gbspos))

    def __pp_parameters(self, params):
        msg = "a1: {0}, a2: {1}, a3: {2}, a4: {3}, b1: {4}"
        a1 = str(round(params[0], 2))
        a2 = str(round(params[1], 2))
        a3 = str(round(params[2], 2))
        a4 = str(round(params[3], 2))
        b1 = str(round(params[4], 2))
        msg = msg.format(a1, a2, a3, a4, b1)

        return msg

    def __compute_new_velocities(self):
        r1 = np.random.uniform(size=(self.nparticles, self.nparams))
        r2 = np.random.uniform(size=(self.nparticles, self.nparams))

        social_term = self.c2 + r2 * (self.gbspos - self.sw_p)
        cognitive_term = self.c1 * r1

        nw_v = self.w0 * self.sw_v + social_term + cognitive_term

        return nw_v

    def __compute_new_positions(self, nw_v):
        nw_p = self.sw_p + self.sw_v

        nw_p = np.where(nw_p < self.constraints[:, 0], self.min_const, nw_p)
        nw_p = np.where(nw_p > self.constraints[:, 1], self.max_const, nw_p)

        return nw_p

    def __set_positions(self, nw_p):
        self.sw_p = nw_p

    def __set_velocities(self, nw_v):
        self.sw_v = nw_v

    def __compute_position_cost(self):
        a1 = np.array([self.sw_p[:, 0] for _ in range(len(self.size))]).transpose()
        a2 = np.array([self.sw_p[:, 1] for _ in range(len(self.size))]).transpose()
        a3 = np.array([self.sw_p[:, 2] for _ in range(len(self.size))]).transpose()
        a4 = np.array([self.sw_p[:, 3] for _ in range(len(self.size))]).transpose()
        b1 = np.array([self.sw_p[:, 4] for _ in range(len(self.size))]).transpose()

        istar = a1 * (np.power(self.size, a2)) * (np.power(self.volatility, a3))
        est_cost = (b1 * istar * (np.power(self.pov, a4))) + (1 - b1)

        return est_cost

    def __compute_error(self, est_cost):
        error = np.sum(np.power(est_cost - self.cost, 2), 1)

        return error

values = impact[['Size', 'Volatility', 'POV', 'Cost']]

size = values['Size'].as_matrix()
volatility = values['Volatility'].as_matrix()
pov = values['POV'].as_matrix()
cost = values['Cost'].as_matrix()

pso = ParticleSwarm(size, volatility, pov, cost, c1=3)
pso.evaluate_fitness()
