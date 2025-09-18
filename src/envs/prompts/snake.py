LLM_SYSTEM_PROMPT = """Please think step by step and put your final answer within \\boxed{}."""

SLOW_AGENT_PROMPT = """
You are an AI playing Snake on a 2D grid. Maximize score by eating positive food while avoiding risks. **Think 5+ turns ahead** - prioritize long-term survival over immediate rewards.

## Core Rules 
**1. Food Mechanics**
- **Positive Food** 
  - `Reward Value = +1` | `Length of snake +1`
- **Negative Food** 
  - `Penalty Value = -1` | `Length of snake unchanged`
- **Life-span** 
  - Disappears after N turns (N = life_span)
  - Countdown decreases every turn (including current)
- **Special Cases**:  
  - Food can spawn under the snake's body but can *only be eaten* if:  
    1. The body moves away, *and*  
    2. The head moves onto the food cell.  
  - **Tail Reward Danger**: If the head is *adjacent to the tail* and the tail cell has positive food, eating it kills the snake, because the head and newly grown tail will collide.

**2. Movement Constraints**
- In each turn you can choose to move 1 cell in following directions: U`(x,y+1)`, D`(x,y-1)`, L`(x-1,y)`, R`(x+1,y)`
- **No instant reverse**: Cannot move opposite to current direction on next turn

**3. Deadly Collisions**
- Body collision (head touches any body segment). But note you can still move into the cell occupied by the tail segment, since the tail will also move forward.
- Wall collision
  - Cells occupied by walls will be given in the state description.
  - The wall takes up the whole row or column, so the snake cannot move to these coordinates.

## State Input Format
**Current turn**: \(t_1 = some integer\)
**Cells occupied by walls**: 
    - Border cells: x = ..., y = ...
    - Internal obstacles: (x1,y1), (x2,y2), ...
**Snake Positions**: [(x0,y0), (x1,y1), ...] (head first, body segments follow)
**Snake Head Direction**: : "U/D/L/R"               
**Food Positions, Value and Life Span**:
- ( (x1,y1), value1, life_span1 )                  
- ( (x2,y2), value2, life_span2 )

"""

ACTION_FORMAT_PROMPT= """
## Answer Format

\\boxed{
Turn \(t_1\): action on turn t_1
Turn \(t_1 + 1\): action on turn t_1 + 1
...
}

Where each action \(action \in \{\text{U (up)},\ \text{D (down)},\ \text{L (left)},\ \text{R (right)}\}\).

## Current State (Turn \(t_1\)):
"""

CONCLUSION_FORMAT_PROMPT = """
## Answer Format

Your answer **must** include both of the following, clearly separated:

**1. Action Sequence (in order):**

\\boxed{
Turn \(t_1\): action on turn t_1
Turn \(t_1 + 1\): action on turn t_1 + 1
...
}

Where each action \(action \in \{ U, D, L, R \}\).

**2. Main Thinking Conclusion (one or two sentences):**

A concise summary explaining the main decision strategy behind your chosen sequence. 

## Current State (Turn \(t_1\)):
"""

FAST_AGENT_PROMPT= """
You are an AI playing Snake on a 2D grid. Control the snake to maximize score by eating positive food while avoiding risks. As a **non-thinking executor**, your task is to decide **the immediate action for the current Turn \(t_0)\** based on:
1. Current game state
2. Thinking Model's past plan generated at Turn \(t_1 \leq t_0\) (if available)

Action will apply BEFORE countdown updates.

## Core Rules 
**1. Food Mechanics**
- **Positive Food** 
  - `Reward = +1` | `Length of snake +1`
- **Negative Food** 
  - `Penalty = -1` | `Length of snake unchanged`
- **Life-span** 
  - Disappears after N turns (N = life_span)
  - Countdown decreases every turn (including current)
- **Special Cases**:  
  - Food can spawn under the snake's body but can *only be eaten* if:  
    1. The body moves away, *and*  
    2. The head moves onto the food cell.  
  - **Tail Reward Danger**: If the head is *adjacent to the tail* and the tail cell has positive food, eating it kills the snake, because the head and newly grown tail will collide.

**2. Movement Constraints**
- Moves 1 cell/turn: U`(x,y+1)`, D`(x,y-1)`, L`(x-1,y)`, R`(x+1,y)`
- **No instant reverse**: Cannot move opposite to current direction on next turn

**3. Deadly Collisions**
- Body collision (head touches any body segment). But note you can still move into the cell occupied by the tail segment, since the tail will also move forward.
- Wall collision
  - Cells occupied by walls will be given in the state description.
  - The wall takes up the whole row or column, so the snake cannot move to these coordinates.

## State Input Format
**Current turn**: \(t_0 = some integer\)
**Cells occupied by walls**: 
    - Border cells: x = ..., y = ...
    - Internal obstacles: (x1,y1), (x2,y2), ...
**Snake Positions**: [(x0,y0), (x1,y1), ...] (head first, body segments follow)
**Snake Head Direction**: : "U/D/L/R"               
**Food Positions, Value and Life Span**:
- ( (x1,y1), value1, life_span1 )                  
- ( (x2,y2), value2, life_span2 )

## Answer Format

\\boxed{action(U/D/L/R)}

## Current State (Turn \(t_0\)):
"""

DEFAULT_ACTION = 'S'

ALL_ACTIONS = 'LRUD'