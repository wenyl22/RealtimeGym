# This is Overcooked environment for one player cooperated with a script agent.
from envs.minatar.environments.overcooked_new.Overcooked_Env import Overcooked
from collections import defaultdict
from envs.minatar.environments.overcooked_new.src.overcooked_ai_py.mdp.actions import Action, Direction
import numpy as np
class Env:
    def __init__(self, ramping=None):
        self.all_args = None
        self.run_dir = None
        self.random = np.random.RandomState()
        self.seed = 42

    def reset(self):
        self.gym_env = Overcooked(self.all_args, self.run_dir, featurize_type=("bc", "bc"))
        eval_obs, _, _ = self.gym_env.reset(True)

        self.reward = 0
        self.game_turn = 0
        self.terminal = False
        self.history = [[], []]  # history[0] for player 0, history[1] for player 1

        # self.eval_env_infos = defaultdict(list)

    def act(self, a):
        self.game_turn += 1
        
        if a == "U":
            action = Direction.SOUTH
        elif a == "D":
            action = Direction.NORTH
        elif a == "L":
            action = Direction.WEST
        elif a == "R":
            action = Direction.EAST
        elif a == "I":
            action = Action.INTERACT
        else:
            action = Action.STAY
        self.gym_env.script_agent[0].next_action = action
        eval_ob, eval_share_ob, eval_reward, eval_done, eval_info, eval_available_action, joint_action = self.gym_env.step([[0], [0]])
        self.reward += sum(eval_reward[0])
        self.terminal = eval_done[0]
        self.history[0].append(Action.A_TO_CHAR[joint_action[0]])
        self.history[1].append(Action.A_TO_CHAR[joint_action[1]])
        return self.reward, self.terminal
    def state_string(self):
        ret = self.gym_env.base_mdp.state_string(self.gym_env.base_env.state)
        ret = ret.split('\n')
        ret = ret[::-1]
        ret = "\n".join(ret)
        ret = ret.replace("↑", "x").replace("↓", "y")
        ret = ret.replace("x", "↓").replace("y", "↑")
        return ret
    