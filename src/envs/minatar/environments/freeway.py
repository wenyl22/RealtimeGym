import numpy as np

class Env:
    def __init__(self, ramping=None):
        self.action_map = ['N','L','U','R','D','F']
        self.random = np.random.RandomState()
        self.seed = 42

    # Reset to start state for new episode
    def reset(self):
        self.chosen_freeways = self.random.choice(range(0, 8), 8, replace=False)
        self.chosen = [True if i in self.chosen_freeways else False for i in range(8)]
        self._randomize_cars()
        self.pos = 9
        self.terminal = False
        self.reward = 0
        self.game_turn = 0
        self.new_car = True

    # Randomize car speeds and directions
    def _randomize_cars(self):
        directions = np.sign(self.random.rand(8) - 0.5).astype(int)
        self.cars = []
        # Patterns: 
        # 1. Random batch neighbour lanes, share same car distribution
        # 2. Each car distribution is one of with equal probability:
            # a. Long cars, fast speed.
            # b. Consecutive car fleet, with constant speed and spacing.
            # c. Random one car as original.
        batch = self.random.randint(0, 2, 8)
        cur_cars = []
        for i in range(8):
            if batch[i] == 1 and i != 0:
                for j in range(len(cur_cars)):
                    cur_cars[j][1] = i + 1
                self.cars.extend([car for car in cur_car] for cur_car in cur_cars)
                continue
            cur_cars = []
            rnd = self.random.randint(0, 3) # [0, 2]
            pos = 0 if directions[i] == 1 else 8
            if rnd == 0:
                speed = self.random.randint(2, 5) # [2, 4] 
                length = speed
                cur_cars = [[pos, i + 1, 0, 1.0/speed * directions[i], speed]]
            elif rnd == 1:
                num = self.random.randint(2, 4) # [2, 5]
                speed = self.random.randint(1, 4) # [1, 3]
                for j in range(num):
                    cur_cars.append([pos, i + 1, abs(speed) - 1, speed, 1])
                    pos += 9 // num if directions[i] == 1 else -(9 // num)
                    pos = (pos + 9) % 9
            else:
                speed = self.random.randint(1, 5) #[1, 4]
                cur_cars = [[pos, i + 1, abs(speed) - 1, speed, 1]]
            self.cars.extend([car for car in cur_car] for cur_car in cur_cars)
        for i in range(len(self.cars)):
            if self.chosen[self.cars[i][1] - 1] == False:
                self.cars[i] = [None, self.cars[i][1], None, None, None]
    # Update environment according to agent action
    def act(self, a):
        self.r = 0
        self.game_turn += 1
        self.new_car = False
        assert not self.terminal
        if a == 'U':
            self.pos = max(0, self.pos - 1)
        elif a == 'D':
            self.pos = min(9, self.pos + 1)

        # Win condition
        if self.pos == 0:
            self.r += 1
            self.pos = 9
            self.reward += self.r
            self.terminal = True
            return self.r, self.terminal
        # Update cars
        # car: [x, y, timer, speed, length]
        for car in self.cars:
            if car[3] is None:
                continue
            dir = -1 if car[3] > 0 else 1
            if car[0] < 0:
                car[0] = 8
                self.new_car = True
            elif car[0] > 8:
                car[0] = 0
                self.new_car = True
            else:
                if(abs(car[3]) >= 1):
                    car[2] -= 1
                    if car[2] == -1:
                        car[2] += abs(car[3])
                        car[0] += 1 if car[3] > 0 else -1
                else:
                    car[0] += int(1/car[3])
            # collision check
            for l in range(car[4]):
                if car[0] + l*dir == 4 and self.pos == car[1]:
                    self.pos = 9
                    self.r = -1
        self.reward += self.r
        self.terminal = True if self.game_turn >= 100 else False
        if not self.terminal and self.r < 0:
            R = self.reward
            G = self.game_turn
            self.random = np.random.RandomState(self.seed)
            self.reset()
            self.reward = R
            self.game_turn = G
        return self.r, self.terminal
        
    def state_string(self):
        grid_string = ""
        for i in range(10): # rows, i.e. y
            for j in range(9): # columns, i.e. x
                grid_string_add = ""
                if(j == 4 and self.pos == i):
                    grid_string_add += 'P'
                for car in self.cars:
                    if car[3] is None or car[1] != i:
                        continue
                    dir = 1 if car[3] > 0 else -1
                    if(car[0] == j):
                        speed = abs(car[3])
                        speed = '/'+str(speed) if speed >= 1 else str(int(abs(1/car[3])))
                        if car[3] > 0:
                            grid_string_add += speed + '>'
                        else:
                            grid_string_add = '<' + speed
                    else:
                        if car[0] < j < car[0] - dir * car[4]:
                            grid_string_add += 'x'
                        if car[0] - dir * car[4] < j < car[0]:
                            grid_string_add += 'x'
                if grid_string_add == "":
                    grid_string_add = "."
                grid_string += grid_string_add                            
                grid_string += "".join([" "] * (4 - len(grid_string_add)))
                grid_string += " "
            grid_string += "\n"
        return grid_string
    def deep_copy(self):
        env = Env()
        env.action_map = self.action_map
        env.random = self.random
        env.chosen_freeways = self.chosen_freeways
        env.chosen = self.chosen
        env.cars = [[car for car in cars] for cars in self.cars]
        env.pos = self.pos
        env.terminal = self.terminal
        env.game_turn = self.game_turn
        env.reward = self.reward
        return env
    def has_event(self):
        # Check if a new car just appeared
        return self.new_car