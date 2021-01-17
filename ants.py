from domain import *
import random

class Ant:
    def __init__(self, nest_location, food_location, ant_tag, epsilon=default_epsilon, target_range=default_target_range,
                 init_location = []):
        self.nest_location = np.array(nest_location)
        self.food_location = np.array(food_location)
        self.target_range = target_range

        if any(init_location):
            self.nt = np.array([init_location, init_location])
        else:
            self.nt = np.array([self.nest_location, self.nest_location])  # [nt,nt1]

        self.mode = [0, 0]  # [node_t, node_t1] nt1 has been chosen under mode_t
        self.epsilon = epsilon
        self.trips = 0
        self.move = np.array([[0., 0.], [0.,0.]])
        self.w = np.array([np.NaN, np.NaN])
        self.ant_tag = ant_tag

    def _search_food(self):
        self.mode[0] = self.mode[1]
        self.mode[1] = 0

    def _search_nest(self):
        self.mode[0] = self.mode[1]
        self.mode[1] = 1

    def _reached_food(self):
        return np.linalg.norm(self.nt[1] - self.food_location) <= self.target_range + numeric_margin

    def _reached_nest(self):
        return np.linalg.norm(self.nt[1] - self.nest_location) <= self.target_range + numeric_margin

    def _pick_direction(self,beacons):
        if self.mode[1] == 0:
            w_type = 1
        else:
            w_type = 0
        self.move[0] = self.move[1]

        derv = drv_gaussian(self.nt[1][0], self.nt[1][1], beacons.beacons, w_type) + \
               np.array([x_drv_elips_gaussian(self.nt[1][0], self.nt[1][1]),
                        y_drv_elips_gaussian(self.nt[1][0], self.nt[1][1])])

        # derv = drv_gaussian(self.nt[1][0], self.nt[1][1], beacons.beacons, w_type)

        if np.linalg.norm(derv) < step_threshold or self.epsilon > random.uniform(0,1):
            derv = np.array([random.uniform(-1,1),random.uniform(-1,1)])
            # derv = np.random.normal(scale=dt,size=(2))

        # derv_stay_in_region = np.array([x_drv_elips_gaussian(self.nt[1][0], self.nt[1][1]),
        #                 y_drv_elips_gaussian(self.nt[1][0], self.nt[1][1])])
        # if np.linalg.norm(derv_stay_in_region) > 0:
        #     # derv = derv_stay_in_region
        #     derv = self.normalize(derv_stay_in_region)

        if move_type == 'add':
           self.move[1] = self.normalize(self.normalize(derv)*dt + self.move[1])*dt
        elif move_type == 'der':
            self.move[1] = self.normalize(derv)*dt
        elif move_type == 'add_switch':
            if self.mode[0] != self.mode[1]:
                self.move[1] = -self.move[0]
            else:
                self.move[1] = self.normalize(self.normalize(derv)*dt + self.move[1])*dt
                # # self.move[1] = self.normalize(derv * dt + self.move[1]) * dt
                # self.move[1] = self.normalize(derv + self.move[1]) * dt

        return self.move[1]

        # if self.epsilon > random.uniform(0,1) or np.linalg.norm(derv) <0.000001:
        #     return self.normalize(np.array([random.uniform(-1,1),random.uniform(-1,1)]))*dt
        # else:
        #     print('non random')
        #     return self.normalize(derv)*dt

    def step(self,beacons):
        self.nt[0] = self.nt[1]
        self.nt[1] = self.nt[1] + self._pick_direction(beacons)
        if self.mode[1] == 0 and self._reached_food():
            self.trips += 1
            self._search_nest()
        elif self.mode[1] == 1 and self._reached_nest():
            self.trips += 1
            self._search_food()
        else:
            self.mode[0] = self.mode[1]

    @staticmethod
    def normalize(item):
        return item / np.linalg.norm(item)

    def find_closest_beacon(self, beacons):
        # self.cl_beac = beacons.tree.query(self.nt[1])[1]
        try:
            self.cl_beac = list(beacons.beacons.keys())[beacons.tree.query(self.nt[1])[1]]
        except:
            test = 0
        # cl_node = grid.find_closest_grid_point(self.nt[1])['elem']
        # self.cl_beac = beacons.map_closest_beacon[cl_node[1]][cl_node[0]]

    def update_weights(self, beacons):
        self.w[0] = gaussian(self.nt[1][0], self.nt[1][1], beacons.beacons, 0)
        self.w[1] = gaussian(self.nt[1][0], self.nt[1][1], beacons.beacons, 1)

class Ants:
    def __init__(self, nest_location, food_location, epsilon=default_epsilon, N=default_N):
        # self.ants = [Ant(nest_node, food_node, epsilon=default_epsilon) for _ in range(0, N)]
        # self.ants = {ant_tag: Ant(nest_location, food_location, ant_tag, epsilon=epsilon)
        #              for ant_tag in range(1, N+1)}
        self.ants = dict()

        self.nest_location = nest_location
        self.food_location = food_location
        self.epsilon = epsilon

    def steps(self, beacons):
        for ant_tag in self.ants:
            self.ants[ant_tag].step(beacons)

    def find_closest_beacon(self, beacons):
        for ant_tag in self.ants:
            self.ants[ant_tag].find_closest_beacon(beacons)

    def update_weights(self, beacons):
        for ant_tag in self.ants:
            self.ants[ant_tag].update_weights(beacons)

    def release_ants(self, n, beac_tags):
        next_tag = max(list(self.ants.keys()) + beac_tags,default=-1)+1
        for ant_tag in range(next_tag,next_tag+n):
            self.ants[ant_tag] = Ant(self.nest_location, self.food_location, ant_tag, epsilon=self.epsilon)

    # def ants_mapper(fnc):
    #     def inner(self, beacons):
    #         for ant_tag in self.ants:
    #             self.ants[ant_tag].fnc
    #     return inner
