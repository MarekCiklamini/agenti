
### notes 
https://github.com/yfeng997/MadMario

https://huggingface.co/blog/deep-rl-ppo

https://huggingface.co/learn/deep-rl-course/unit3/deep-q-network

https://huggingface.co/learn/deep-rl-course/unit6/advantage-actor-critic 
## Zadání:
Implementuj libovolne prostredi (Pole, Grid, Hra, vlastni ...etc.) pro Reinforcement \
learning a natrenuj libovolneho agenta (Q-table, DQN, REINFORCE, PPO, DPO ... \
etc.). \

## Forma odevzdání:
Vypracovaný úkol odevzdejte ve formě zdrojového kódu. Projekt ideálně nahrajte na
Github a odevzdejte link do Github repositáře. Link odevzdejte v Google Classroom.

ref: https://classroom.google.com/u/1/c/NzY5MjA5NjY3NjMy/a/NzY5MjA5NjY3Njc5/details

## notes

```mermaid
classDiagram
    class TradingEnvironment {
        -df: DataFrame
        -prices: ndarray  
        -dates: Index
        -features: List[str]
        -initial_balance: float
        -commission: float
        -window_size: int
        -max_position_size: float
        -current_step: int
        -balance: float
        -shares_held: int
        -cost_basis: float
        -trades: List
        -portfolio_values: List
        -action_space: Discrete
        -observation_space: Box
        
        +__init__(df, initial_balance, commission, window_size, features, max_position_size, render_mode)
        +reset(seed, options) Tuple[ndarray, Dict]
        +step(action) Tuple[ndarray, float, bool, bool, Dict]
        +render() Optional[ndarray]
        -_execute_trade(action) None
        -_get_observation() ndarray
        -_calculate_reward(prev_value, current_value) float
        -_get_portfolio_value() float
        -_get_info() Dict
    }
    
    class gym.Env {
        <<interface>>
        +reset()
        +step()
        +render()
    }
    
    TradingEnvironment --|> gym.Env : inherits

    note for TradingEnvironment "Actions:\n0: Sell\n1: Hold\n2: Buy"
```
```mermaid
flowchart TD
    A[Initialize Environment] --> B[Reset Episode]
    B --> C[Get Initial Observation]
    C --> D[Agent Selects Action]
    D --> E{Action Type}
    
    E -->|0 - Sell| F[Execute Sell Order]
    E -->|1 - Hold| G[Do Nothing]
    E -->|2 - Buy| H[Execute Buy Order]
    
    F --> I[Update Portfolio State]
    G --> I
    H --> I
    
    I --> J[Calculate Reward]
    J --> K[Get New Observation]
    K --> L{Episode Done?}
    
    L -->|No| D
    L -->|Yes| M[Return Final Results]
    
    subgraph "Observation Components"
        N[Historical Price Data<br/>Normalized Returns]
        O[Portfolio State<br/>Cash/Position Ratios]
        P[Market Indicators<br/>Volatility/Trend]
    end
    
    subgraph "Reward Components"
        Q[Portfolio Returns]
        R[Risk Adjustment<br/>Sharpe Ratio]
        S[Transaction Penalties]
    end
    
    K --> N
    K --> O  
    K --> P
    
    J --> Q
    J --> R
    J --> S
```
