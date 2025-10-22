from copy import deepcopy

import numpy as np

from .base import BaseEnv
from .render.snake_render import SnakeRender

seed_mapping = {
    "E": {i: 1000 + i for i in range(32)},
    "M": {i: 5000 + i for i in range(32)},
    "H": {i: 8000 + i for i in range(32)},
}


def setup_env(seed, cognitive_load, save_trajectory_gifs=False):
    env = SnakeEnv()
    env.set_seed(seed_mapping[cognitive_load][seed])
    render = None
    if save_trajectory_gifs:
        render = SnakeRender()
    return env, seed_mapping[cognitive_load][seed], render


class SnakeEnv(BaseEnv):
    def reset(self):
        self.B = 8
        self.true_seed = self.seed % 1000
        self.random = np.random.RandomState(self.true_seed)
        self.coords = [
            (x, y) for x in range(1, self.B - 1) for y in range(1, self.B - 1)
        ]
        self.snake = [(self.B // 2 - 1, self.B // 2 - 1)]
        self.num_obstacle = self.seed // 1000
        step = self.num_obstacle
        self.obstacle = []
        while step > 0:
            x = self.random.randint(1, self.B - 1)
            y = self.random.randint(1, self.B - 1)
            if (x, y) not in self.snake and (x, y) not in self.obstacle:
                step -= 1
                self.coords.remove((x, y))
                self.obstacle.append((x, y))
        self.coords.remove(self.snake[0])
        self.random.shuffle(self.coords)
        self.random.shuffle(self.coords)
        if len(self.obstacle) >= self.num_obstacle:
            self.obstacle = self.obstacle[: self.num_obstacle]
        else:
            raise ValueError(
                f"Not enough obstacles generated: {len(self.obstacle)} < {self.num_obstacle}"
            )
        self.food = []
        self.food_attributes = [[0 for _ in range(self.B)] for _ in range(self.B)]

        self.dir = "L"
        self.game_turn = 0
        self.reward = 0
        self.terminal = False
        # random permute coords
        # random choose 30% of index in range(200) and set self.value to -1
        self.idx = 0
        self.spawn_food()
        self.spawn_food()
        self.spawn_food()
        # Return initial observation and done flag
        return self.observe(), self.terminal

    def spawn_food(self):
        x, y = self.coords[self.idx]
        self.idx += 1
        if self.idx >= len(self.coords):
            self.idx -= len(self.coords)
        life_span = 10
        value = 1
        new_food = (x, y)
        assert self.food_attributes[x][y] == 0 and new_food not in self.food, (
            f"Food already exists at {new_food}, attributes: {self.food_attributes[x][y]}, coords: {self.coords}"
        )
        self.food.append(new_food)
        self.food_attributes[x][y] = (life_span, value)

    def step(self, a):
        self.r = 0
        self.game_turn += 1
        if (
            (a == "L" and self.dir == "R")
            or (a == "R" and self.dir == "L")
            or (a == "U" and self.dir == "D")
            or (a == "D" and self.dir == "U")
        ):
            a = self.dir  # prevent reverse direction
        #                raise ValueError(f"Invalid action a = {a}, dir = {self.dir}")
        if a in ["L", "R", "U", "D"]:  # ignore invalid actions
            self.dir = a
        head_x, head_y = self.snake[-1]
        if self.dir == "L":
            new_head = (head_x - 1, head_y)
        elif self.dir == "R":
            new_head = (head_x + 1, head_y)
        elif self.dir == "D":
            new_head = (head_x, head_y - 1)
        elif self.dir == "U":
            new_head = (head_x, head_y + 1)
        else:
            raise ValueError(f"Invalid action a = {a}, dir = {self.dir}")
        x, y = new_head
        # Death trigger: hit body; hit wall; head hits newly grown tail
        if (
            new_head in self.snake[1:]
            or new_head in self.obstacle
            or new_head[0] == 0
            or new_head[1] == 0
            or new_head[0] == self.B - 1
            or new_head[1] == self.B - 1
            or (
                new_head == self.snake[0]
                and new_head in self.food
                and self.food_attributes[x][y][1] > 0
            )
        ):
            self.r -= 1
            self.reward += self.r
            self.terminal = True
            return self.observe(), self.terminal, self.reward, False
        self.snake.append(new_head)

        if new_head in self.food:
            self.r += self.food_attributes[x][y][1]
            self.food.remove(new_head)
            self.food_attributes[x][y] = 0
            if self.r < 0:
                self.snake.pop(0)
        else:
            self.snake.pop(0)

        for food in self.food:
            x, y = food
            lifespan, value = self.food_attributes[x][y]
            self.food_attributes[x][y] = (lifespan - 1, value)
            if lifespan <= 1:
                self.food.remove(food)
                self.food_attributes[x][y] = 0
        if self.game_turn % 3 == 1:
            self.spawn_food()
        self.reward += self.r
        self.terminal = True if self.game_turn >= 100 else False
        return self.observe(), self.terminal, self.reward, False

    def state_string(self):
        grid_string = ""
        snake_length = len(self.snake)
        for i in range(self.B):
            for j in range(self.B):
                output = ""
                x, y = j, self.B - 1 - i
                if (x, y) in self.obstacle:
                    output += "#"
                if (x, y) in self.snake:
                    output += chr(
                        ord("a") + snake_length - 1 - self.snake.index((x, y))
                    )
                if (x, y) in self.food:
                    if self.food_attributes[x][y][1] > 0:
                        output += "+"
                    else:
                        output += "-"
                    output += str(self.food_attributes[x][y][0])
                if x == 0 or x == self.B - 1 or y == 0 or y == self.B - 1:
                    output += "#"
                if output == "":
                    output = "."
                grid_string += output + " " * (6 - len(output))
            grid_string += "\n"
        return grid_string

    def get_possible_actions(self):
        # return 'L', 'R', 'U', 'D' removing the reverse of the current direction
        if self.dir == "L":
            actions = ["L", "U", "D"]
        elif self.dir == "R":
            actions = ["R", "U", "D"]
        elif self.dir == "U":
            actions = ["L", "R", "U"]
        elif self.dir == "D":
            actions = ["L", "R", "D"]
        return actions

    def state_builder(self):
        snake = deepcopy(self.snake[::-1])
        foods = []
        for x, y in self.food:
            lifespan, value = self.food_attributes[x][y]
            foods.append((x, y, lifespan, value))
        return {
            "snake_dir": self.dir,
            "internal_obstacles": self.obstacle,
            "foods": foods,
            "snake": snake,
            "size": self.B,
            "game_turn": self.game_turn,
        }

        # state = self.llm_state_builder()
        # description = "**Cells occupied by walls**:\n"
        # description += f"\t - Border Cells: x=0/x={state['size'] - 1} or y=0/y={state['size'] - 1}.\n"
        # description += f"\t - Internal Obstacles: {state['internal_obstacles'] if len(state['internal_obstacles']) > 0 else 'No internal obstacles'}\n"
        # description += f"**Snake Positions**:{state['snake']}\n**Snake Head Direction**: {state['snake_dir']}\n"
        # description += "**Food Positions, Life Span and Value**:\n"
        # for x, y, life_span, value in state["foods"]:
        #     description += f"\t- ({x}, {y}, {life_span}, {value})\n"
        # model1_description = (
        #     f"**Current Turn**: \( t_0 = {self.game_turn} \)\n" + description
        # )
        # model2_description = (
        #     f"**Current Turn**: \( t_1 = {self.game_turn} \)\n" + description
        # )
        # return {
        #     "model1_description": model1_description,
        #     "model2_description": model2_description,
        #     "game_turn": self.game_turn,
        #     "state_string": self.state_string(),
        # }

    def summary(self):
        print(f"Seed {self.seed} - {self.game_turn} turns, reward: {self.reward}")
