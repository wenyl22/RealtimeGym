SLOW_AGENT_PROMPT = """
Help Alice collaborate with Bob in *Overcooked* to maximize the total reward from delivered soup recipes. Optimize coordination, planning, and decision-making based on the current game state and action history of both players.
---
### **Game Rules & Mechanics**  
#### **1. Game Grid (2D Tile Types)**  
- **Empty Tile:** Walkable space.  
- **Dispensers:** Sources for items (tomato/onion/clean plate).  
- **Pot:** Cooks soup (3 ingredients → auto-starts cooking).  
- **Serving Counter:** Deliver cooked soup for rewards.  
- **Kitchen Counter:** Temporary item storage.  

#### **2. Player Actions per Turn**  
Each player (Alice/Bob) can:  
- **Move** (L/R/U/D): Changes position *and* orientation. Specifically, L (x-1, y), R (x+1, y), U (x, y+1), D (x, y-1).
- **Change Orientation** (L/R/U/D): No move if the adjacent tile is not an empty one.  
- **Interact (I):** Player can interact with a non-empty tile if it is 1) standing on an empty tile adjacent to the tile and 2) facing the tile. Interact effects depend on game state and tile type:
  - **Dispensers:** Grab item (if player is empty-handed).
  - **Pot:**  
    - Add ingredient (if pot is not full and player is holding ingredient) → cooks automatically at 3 ingredients.  
    - Serve soup (if player is holding a clean plate and pot has cooked soup).
  - **Serving Counter:** Deliver soup (if player is holding a soup in plate).
  - **Kitchen Counter:** Pick up items (if player is empty-handed and counter is occupied) or place items (if player is holding an item and counter is empty).
- **Stay (S):** Do nothing.

#### **3. Key Constraints**  
- Players can carry **one item at a time**.  
- Players cannot walk onto/through each other.
- Players cannot walk onto/through non-empty tiles. 

#### **4. Recipes & Rewards**
- Each recipe requires different ingredients, cooking time, and yields varying rewards.
- Cooking time is 20 turns if ingredients do not match any recipe in the order, and serving the recipe does not yield rewards.
---

### **Your Task**  
At turn \( t_1 \), plan a sequence of actions \(\{a_{t_1 + t}\}_{t=1}^{H-1}\) for Alice (and optionally Bob) over next several turns(say, H turns) to maximize total reward from delivered recipes.

"""
# SLOW_AGENT_PROMPT="""
# You are helping Alice playing the cooperative game Overcooked. Your goal is to prepare and deliver soup to maximize the sum of reward of the delivered recipes. You must coordinate with your teammate Bob efficiently, plan ahead, and make optimal decisions based on the current game state.

# ## Game Rules and Mechanics
# - The game is played on a 2D grid, with each tile being one of the following:
#     - Empty Tile: Walkable area
#     - Tomato Dispenser: Source of tomatoes
#     - Onion Dispenser: Source of onions
#     - Plate Dispenser: Source of clean plates
#     - Pot: Place to cook soup
#     - Serving Counter: Place to deliver the cooked soup
#     - Kitchen Counter: Place to put items temporarily

# - The player needs to grab ingredients from onions and tomatoes and put them into one pot to cook a recipe. After a soup is cooked, the player must use a clean plate to serve and deliver it to the delivery point. Different recipes require different ingredients and cooking time, and give different rewards upon delivery. 

# - In each turn, each player can be on an empty tile (x, y) and face one of the four directions: Up(U), Down(D), Left(L), Right(R). They can perform one action from the following options, respectively:
#     - Move to an adjacent tile using action L (x-1, y), R (x+1, y), U (x, y+1), D (x, y-1). Note once the player moves in the direction, its orientation will change to that direction.
#     - Change orientation: If the player wants to interact with a tile, they must be 1) standing on an empty tile adjacent to the tile and 2) facing the tile. They can change their orientation to face the tile using action L, R, U, or D. Note by default, the player will move one unit in that direction. However, if the tile is not an empty tile, they just change their orientation to face the tile.
#     - Interact with a tile: If the player is facing a tile, they can interact with it using the action I. The interaction depends on the tile type and game state, for example:
#         - Tomato/Onion/Plate Dispenser: If the player is not holding an item, 'I' will grab an item from the dispenser.
#         - Pot(Not full): If the player is holding an ingredient, 'I' will place the ingredient into the pot. Once the pot has 3 ingredients, it will **AUTOMATICALLY** start cooking. If the ingredients do not match any recipe in the order, the cooking process will last 20 turns, but serving the soup does not give any reward.
#         - Pot(With cooked soup): If the player is holding a clean plate, 'I' will serve the soup onto the plate.
#         - Serving Counter: If the player is holding a plate with soup, 'I' will deliver the soup and receive the recipe reward.
#         - Kitchen Counter(Empty): If the player is holding an item, 'I' will put the item on the counter.
#         - Kitchen Counter(Occupied): If the player is not holding an item, 'I' will grab the item from the counter.
# - Players can only carry one item at a time
# - Players cannot walk through each other.

# ## Your Task
# Suppose current game turn is \(t_1\). Plan for the next several turns(say, H turns). Find a sequence of actions \(\{a_{t_1 + t\}_{t=1}^{H-1}\) in order to maximize the total reward of delivered recipes.
# """

ACTION_FORMAT_PROMPT = '''
**Answer Format**:

\\boxed{
Turn t_1: a_\{t_1\}
Turn t_1 + 1: a_\{t_1 + 1\}
...
}

Where each action \(a_t \in \{U, D, L, R, I, S\}\).
'''

CONCLUSION_FORMAT_PROMPT = '''
**Answer Format**:

Your answer **must** include both of the following, clearly separated:

**(1) Action Sequence (in order):**

\\boxed{
Turn t_1: a_\{t_1\}
Turn t_1 + 1: a_\{t_1 + 1\}
...
}

Where each action \(a_t \in \{U, D, L, R, I, S\}\).

**(2) Main Thinking Conclusion (one or two sentences):**

A concise summary explaining the main decision strategy behind your chosen sequence. 
'''

FAST_AGENT_PROMPT = """
Help Alice collaborate with Bob in *Overcooked* to maximize the total reward from delivered soup recipes. You need to decide the immediate action for the current Turn \(t_0\) based on:
1. Current game state and action history of both players.
2. Thinking model's past plan. Sometimes you may be given a plan generated by a thinking model at turn \(t_1 \leq t_0\), which might be outdated or inaccurate, but it can still provide useful information for your decision making. You can take it as a **strategic reference**, not a mandatory instruction. 

### **Game Rules & Mechanics**  
#### **1. Game Grid (2D Tile Types)**  
- **Empty Tile:** Walkable space.  
- **Dispensers:** Sources for items (tomato/onion/clean plate).  
- **Pot:** Cooks soup (3 ingredients → auto-starts cooking).  
- **Serving Counter:** Deliver cooked soup for rewards.  
- **Kitchen Counter:** Temporary item storage.  

#### **2. Player Actions per Turn**  
Each player (Alice/Bob) can:  
- **Move** (L/R/U/D): Changes position *and* orientation. Specifically, L (x-1, y), R (x+1, y), U (x, y+1), D (x, y-1).
- **Change Orientation** (L/R/U/D): No move if the adjacent tile is not an empty one.  
- **Interact (I):** Player can interact with a non-empty tile if it is 1) standing on an empty tile adjacent to the tile and 2) facing the tile. Interact effects depend on game state and tile type:
  - **Dispensers:** Grab item (if player is empty-handed).
  - **Pot:**  
    - Add ingredient (if pot is not full and player is holding ingredient) → cooks automatically at 3 ingredients.  
    - Serve soup (if player is holding a clean plate and pot has cooked soup).
  - **Serving Counter:** Deliver soup (if player is holding a soup in plate).
  - **Kitchen Counter:** Pick up items (if player is empty-handed and counter is occupied) or place items (if player is holding an item and counter is empty).
- **Stay (S):** Do nothing.

#### **3. Key Constraints**  
- Players can carry **one item at a time**.  
- Players cannot walk onto/through each other.
- Players cannot walk onto/through non-empty tiles. 

#### **4. Recipes & Rewards**
- Each recipe requires different ingredients, cooking time, and yields varying rewards.
- Cooking time is 20 turns if ingredients do not match any recipe in the order, and serving the recipe does not yield rewards.
---

### **Your Task**
Decide the immediate action \(a_\{t_0\}\) at turn \(t_0\) for Alice based on the current game state and the past plan from a thinking model at turn \(t_1\).

**Answer Format**:
\\boxed{a_{t_0}}

Where \(a_{t_0} \in \{U, D, L, R, I, S\}\).
"""

GAME_STATE_PROMPT="""

## Environment Details

### Tile Types
    - Kitchen Counter: {kitchen_counter}
    - Tomato Dispenser: {tomato}
    - Onion Dispenser: {onion}
    - Plate Dispenser: {plate}
    - Pot: {pot}
    - Serving Counter: {serving_counter}

### Recipe Information
{recipe_infos}

## Current Game State

Game Turn: {t_format}

### Player Information
- **You (Alice)**
    - Position: {my_position}
    - Orientation: {my_orientation}
    - Holding: {my_holding}
    - Action History: {my_action_history}

- **Teammate (Bob)**
    - Position: {he_position}
    - Orientation: {he_orientation}
    - Holding: {he_holding}
    - Action History: {he_action_history}

Note: Action history is a list of actions taken by the player in the passed several turns(at most 5), with the most recent action listed at the end of the array.

### Non-empty Kitchen Counter
{kitchen_counter_state}
### Non-empty Pot State
{pot_state}
"""

ALL_ACTIONS = "UDLRIS"
DEFAULT_ACTION = "S"
