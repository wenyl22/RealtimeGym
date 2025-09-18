LLM_SYSTEM_PROMPT = '''Please think step by step. Then put your final answer within \\boxed{}.'''

SLOW_AGENT_PROMPT = '''
Now a player is playing a multi-turn game, and suppose current turn is \{t_1\}. Given the inital position \((0, y_{t_1})\) on a 2D grid (vertical axis \(y = 0, 1, \dots, 9\)), determine the minimal number of turns \(H\) and a sequence of actions \(\{a_{t_1 + t}\}_{t=0}^{H-1}\) to reach \((0, 9)\), avoiding collisions with cars on freeways \(y = 1, \dots, 8\). 
---
### 1. **Game Dynamics:**
- Player update:  
  \( y_{t+1} = y_t + \Delta y_{t} \), where  
  \[
  \Delta y_{t} = \begin{cases} 
  +1 & \text{if } a_{t} = U \\ 
  -1 & \text{if } a_{t} = D \\ 
  0 & \text{if } a_{t} = S 
  \end{cases}, \quad y_{t+1} \in [0,9]
  \]
- Car update rules: 
    For car \(k\) on freeway \(i\), suppose its head is at \(h\), tail is at \(\tau\) at turn \(t_1\), and speed is \(s\). Then at turn \(T > t_1\), the car span becomes:
  - Left-moving: \(\text{Span}(t_1) = [h, \tau] \rightarrow \text{Span}(T) = [h - s (T-t_1), \tau - s (T-t_1)]\)
  - Right-moving: \([\text{Span}(t_1) = [\tau, h] \rightarrow \text{Span}(T) = [\tau + s (T-t_1), h + s (T-t_1)]\)
- Collision occurs at turn \(T\) only if \( 0 \in \text{Span}(T) \) for any car on freeway \(y_T\). 
- Note that if you decide to move to \( y_{T+1}\not= y_{T}\) at turn \(T\), you will **NOT** be considered to be on \(y_{T+1}\) at turn \(T\), thus will **NOT** be collided by cars on \(y_{T+1}\) if \(0 \in \text{Span}(T)\) but \(0 \notin \text{Span}(T+1)\).
---
### 2. **Task (Turn \(t_1\)):**
Find a sequence of actions \(\{a_{t_1 + t\}_{t=1}^{H-1}\) which minimizes \(H\) such that \(y_{t_1 + H - 1} = 9\).  
'''

ACTION_FORMAT_PROMPT = '''
**Answer Format**:

\\boxed{
Turn t_1: a_\{t_1\}
Turn t_1 + 1: a_\{t_1 + 1\}
...
}

Where each action \(a_t \in \{ U, D, S\}\).
---
### 3. **Current State (Turn \(t_1\)):**
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

Where each action \(a_t \in \{ U, D, S\}\).

**(2) Main Thinking Conclusion (one or two sentences):**

A concise summary explaining the main decision strategy behind your chosen sequence. 
---
### 3. **Current State (Turn \(t_1\)):**
'''

FAST_AGENT_PROMPT = '''
You are a player in a freeway game, starting at \((0, y_\{t_0\})\) on a 2D grid (vertical axis \(y = 0, 1, \dots, 9\)). Your goal is to reach \((0, 9)\) while avoiding collisions with cars on freeways \(y = 1, \dots, 8\).
---
### 1. **Game Dynamics:**
- Player update:  
  \( y_{t+1} = y_t + \Delta y_{t} \), where  
  \[
  \Delta y_{t} = \begin{cases} 
  +1 & \text{if } a_{t} = U \\ 
  -1 & \text{if } a_{t} = D \\ 
  0 & \text{if } a_{t} = S 
  \end{cases}, \quad y_{t+1} \in [0,9]
  \]
- Car update rules: 
    For car \(k\) on freeway \(i\), suppose its head is at \(h\), tail is at \(\tau\) at turn \(t_0\), and speed is \(s\). Then at turn \(T > t_0\), the car span becomes:
  - Left-moving: \(\text{Span}(t_0) = [h, \tau] \rightarrow \text{Span}(T) = [h - s (T-t_0), \tau - s (T-t_0)]\)
  - Right-moving: \([\text{Span}(t_0) = [\tau, h] \rightarrow \text{Span}(T) = [\tau + s (T-t_0), h + s (T-t_0)]\)
- Collision occurs at turn \(T\) only if \( 0 \in \text{Span}(T) \) for any car on freeway \(y_T\). 
- Note that if you decide to move to \( y_{T+1}\not= y_{T}\) at turn \(T\), you will **NOT** be considered to be on \(y_{T+1}\) at turn \(T\), thus will **NOT** be collided by cars on \(y_{T+1}\) if \(0 \in \text{Span}(T)\) but \(0 \notin \text{Span}(T+1)\).
---
### 2. **Guidance from a Previous Thinking Model (Turn \(t_1 \leq t_0\)):**  
Sometimes, you have access to a past output from a thinking model, computed at turn \(t_1\) based on then-current observations. This guidance may no longer perfectly match the current situation but can still be valuable for decision-making. You can use this plan as a **strategic reference**, not a mandatory instruction. Consider how much of the original strategy is still valid under the current dynamics.
---
### 3. **Task (Turn \(t_0\)):**
Choose **one** action \(a_{t_0} \in \{U, D, S\}\) for the current turn, with the following considerations:
- **Collision Avoidance:**  
  Ensure the action avoids both immediate and near-future collisions.
- **Strategic Consistency (Optional):**  
  Refer to the thinking model's prior strategy. If the current environment still aligns with its assumptions, you may choose to continue along the same strategic direction. If not, adapt as needed.
**Answer Format**:
\\boxed{
a_{t_0}
}
---
### 4. **Current State (Turn \(t_0\)):**  
'''

DEFAULT_ACTION = "U" 

ALL_ACTIONS = "UDS"