import numpy as np
from environment.base import BaseEnv

seed_mapping = {
    'E': { 0: 1000, 1: 1001, 2: 1002, 3: 1003, 4: 1013, 5: 1014, 6: 1016, 7: 1018},
    'M': { 0: 1069, 1: 1093, 2: 1536, 3: 1858, 4: 1338, 5: 2496, 6: 1933, 7: 1863},
    'H': { 0: 1447, 1: 2408, 2: 2418, 3: 2661, 4: 1100, 5: 1944, 6: 1310, 7: 2453}
}

def setup_env(seed, cognitive_load):
    assert seed in seed_mapping[cognitive_load]
    env = FreewayEnv()
    env.set_seed(seed_mapping[cognitive_load][seed])
    env.reset()
    return env, seed_mapping[cognitive_load][seed]

class FreewayEnv(BaseEnv):
    def reset(self):
        self.chosen_freeways = self.random.choice(range(0, 8), 8, replace=False)
        self.chosen = [True if i in self.chosen_freeways else False for i in range(8)]
        self._randomize_cars()
        self.pos = 9
        self.reward = 100
        self.game_turn = 0
        self.new_car = True
        self.terminal = False
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

    def act(self, a):
        # return: (reward, reset)
        self.r = False # reset or not
        self.reward -= 1
        self.game_turn += 1
        self.new_car = False
        assert not self.terminal
        if a == 'U':
            self.pos = max(0, self.pos - 1)
        elif a == 'D':
            self.pos = min(9, self.pos + 1)

        # Win condition
        if self.pos == 0:
            self.pos = 9
            self.terminal = True
            return self.reward, self.r
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
                    self.r = True
        self.terminal = True if self.game_turn >= 100 else False
        if not self.terminal and self.r:
            R = self.reward
            G = self.game_turn
            self.random = np.random.RandomState(self.seed)
            self.reset()
            self.reward = R
            self.game_turn = G
        return self.reward, self.r
        
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

    def llm_state_builder(self):
        player_states = 9 - self.pos
        car_states = []
        for car in self.cars:
            # car: [x, y, timer, speed, length]
            if car[3] is None:
                car_states.append((9 - car[1], None, None, None, None))
                continue
            dir = 'left' if car[3] < 0 else 'right'
            speed = int(12 / abs(car[3]))
            pos = 12 * (car[0] - 4)
            if abs(car[3]) >= 1:
                if dir == 'left':
                    pos -= (abs(car[3]) - car[2] - 1) * speed
                else:
                    pos += (abs(car[3]) - car[2] - 1) * speed
            else:
                pass
            assert car[2] < abs(car[3])
            car_states.append( (9 - car[1], pos, dir, speed, car[4] * 12 - 1) )
        car_states.sort(key=lambda x: x[0])
        assert self.pos > 0
        state_for_llm = {
            'player_states': player_states,
            'car_states': car_states,
        }
        return state_for_llm

    def observe(self):
        state_for_llm = self.llm_state_builder()
        description = f"""**Player Position:** \( (0, {state_for_llm['player_states']}) \)\n"""
        description += f"""**Car State**:
    | Freeway \( k \) | Cars (head \( h \), tail \( \tau \), direction \( d \), speed \( s \)) |  
    |-----------------|------------------------------------------------------------------------|\n"""
        car_info = ""
        lane = 1
        for car in state_for_llm['car_states']:
            if car[0] != lane:
                description += f"| {lane} | \({car_info}\) |\n"
                car_info = ""
                lane = car[0]
            span = car[4] if car[2] == 'left' else -car[4]
            if car_info != "":
                car_info += ", "
            car_info += f"({car[1]}, {car[1] + span}, {car[2]}, {car[3]})"
        description += f"| {lane} | \({car_info}\) |\n"
        model1_description = f"""**Current Turn:** \( t_0 = {self.game_turn} \) \n""" + description
        model2_description = f"""**Current Turn:** \( t_1 = {self.game_turn} \) \n""" + description
        return {
            'model1_description': model1_description,
            'model2_description': model2_description,
            'state_string': self.state_string(),
            'game_turn': self.game_turn
        }
    def summary(self):
        if self.terminal:
            print(f"Seed {self.seed} finished the game in {self.game_turn} turns with reward {self.reward}.")
        else:
            print(f"Seed {self.seed} position: {9 - self.pos}, turn: {self.game_turn}, reward: {self.reward}")
