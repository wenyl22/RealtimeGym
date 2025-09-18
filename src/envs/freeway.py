import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from minatar.environment import Environment
from minatar.environments.freeway import Env
seed_mapping = {
    'E': { 0: (1000, 13), 1: (1001, 11), 2: (1002, 11), 3: (1003, 11), 4: (1013, 13), 5: (1014, 12), 6: (1016, 11), 7: (1018, 11)},
    'M': { 0: (1069, 12), 1: (1093, 14), 2: (1536, 14), 3: (1858, 16), 4: (1338, 14), 5: (2496, 14), 6: (1933, 15), 7: (1863, 16)},
    'H': { 0: (1447, 19), 1: (2408, 19), 2: (2418, 20), 3: (2661, 21), 4: (1100, 19), 5: (1944, 19), 6: (1310, 21), 7: (2453, 20)}
}

def setup_env(seed, difficulty):
    assert seed in seed_mapping[difficulty]
    env = Environment('freeway', sticky_action_prob=0)
    env.seed(seed_mapping[difficulty][seed][0])
    env.reset()
    return env, seed_mapping[difficulty][seed]

def summarize(seed, difficulty, env):
    smp = seed_mapping[difficulty][seed]
    reset = False
    if env.env.r > 0:
        print(f"Seed {seed} - {smp} get to the other side in {env.env.game_turn} turns.")
        return
    if env.env.game_turn >= 100:
        print(f"Seed {seed} - {smp} failed to get to the other side in 100 turns.")
        return
    if env.env.r < 0:
        print(f"Seed {seed} - {smp} hit by a car, reset the game.")
        reset = True
    print(f"Seed {seed} - {smp} position: {9 - env.env.pos}, turn: {env.env.game_turn}, reward: {env.env.reward}")
    return reset
    

def llm_state_builder(env: Env):
    player_states = 9 - env.pos
    car_states = []
    for car in env.cars:
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
    assert env.pos > 0
    state_for_llm = {
        'player_states': player_states,
        'car_states': car_states,
        'turn': env.game_turn
    }
    return state_for_llm

def state_to_description(state_for_llm, scratch_pad = None, fast = False):
    description = f"""**Current Turn:** \( t_{0 if fast else 1} = {state_for_llm['turn']} \) \n"""
    description += f"""**Player Position:** \( (0, {state_for_llm['player_states']}) \)\n"""
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
    if scratch_pad is not None:
        # add ">" before each line of scratch_pad
        lines = scratch_pad.split('\n')
        for line in lines:
            description += f"> {line.strip()}\n"
    return description

# some utils function
def supervise_collision(state_for_llm, scratch_pad, future_step = 3):
    """
    Check if there is a collision risk by following "scratch_pad" in the next "future_step" turns.
    future_step = min(future_step, len(scratch_pad))
    """
    future_step = min(future_step, len(scratch_pad))
    pos = state_for_llm['player_states']
    for t in range(future_step):
        pos += 1 if scratch_pad[t] == 'U' else -1 if scratch_pad[t] == 'D' else 0
        if pos == 9:
            return False
        pos = max(0, pos)
        for car in state_for_llm['car_states']:
            if car[0] != pos:
                continue
            head = car[1]
            span = car[4] if car[2] == 'left' else -car[4]
            tail =  head + span        
            if car[2] == 'left':
                head = head - car[3] * (t + 1)
                tail = tail - car[3] * (t + 1)
            else:
                head = head + car[3] * (t + 1)
                tail = tail + car[3] * (t + 1)
            if head <= 0 <= tail or tail <= 0 <= head:
                return True
    return False
def check_collision(state, X, action):
    # whether there must be a collision in X steps, no matter what action is taken
    pos = state['player_states']
    pos += 1 if action == 'U' else -1 if action == 'D' else 0
    if pos == 9:
        return False
    def car_collision(car, t, p):
        if car[0] != p:
            return False
        head = car[1]
        span = car[4] if car[2] == 'left' else -car[4]
        tail =  head + span        
        if car[2] == 'left':
            head = head - car[3] * (t + 2)
            tail = tail - car[3] * (t + 2)
        else:
            head = head + car[3] * (t + 2)
            tail = tail + car[3] * (t + 2)
        return head <= 0 <= tail or tail <= 0 <= head
    for car in state['car_states']:
        if car_collision(car, -1, pos):
            return True
    live_pos = set()
    live_pos.add(pos)
    for t in range(X):
        # print(f"a: {action}, t: {t}, live_pos: {live_pos}")
        new_live_pos = set()
        for pos in live_pos:
            temp = pos
            for a in ['S', 'U', 'D']:
                temp = pos + (1 if a == 'U' else -1 if a == 'D' else 0)
                if temp == 9:
                    return False
                flag = False
                for car in state['car_states']:
                    if car_collision(car, t, temp):
                        flag = True
                        break
                if not flag:
                    new_live_pos.add(temp)
        if len(new_live_pos) == 0:
            return True
        live_pos = new_live_pos
    return False
def react_to_collision(state_for_llm, X = 0):
    """
    Returns:
        - stay_collision: bool, True if there is a collision risk by staying
        - preferred_action: str, next action to take if there is a collision risk
    """
    prefer_action = 'S'
    for i in ['U', 'S', 'D']:
        if not check_collision(state_for_llm, X, i):
            prefer_action = i
            break    
    return prefer_action

def tick(state_for_llm, action):
    car_states = []
    pos = state_for_llm['player_states']
    if supervise_collision(state_for_llm, action):
        return None, -1
    for i, car in enumerate(state_for_llm['car_states']):
        ncar = [car[0], car[1], car[2], car[3], car[4]]
        if car[2] == 'left':
            ncar[1] -= ncar[3]
        else:
            ncar[1] += ncar[3]
        car_states.append(ncar)
    r = 0
    if action == 'U':
        pos = min(9, pos + 1)
        if pos == 9:
            r = 1
    elif action == 'D':
        pos = max(0, pos - 1)
    new_state = {
        'player_states': pos,
        'car_states': car_states
    }   
    return new_state, r

def po_navigate(state_for_llm):
    actions = []
    n_state_for_llm = state_for_llm.copy()
    while n_state_for_llm['player_states'] < 9:
        action = react_to_collision(n_state_for_llm, 10)
        actions.append(action)
        n_state_for_llm, r = tick(n_state_for_llm, action)
    return actions