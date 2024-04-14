# project-alphazero

Implementing an chess agent to play at an intermediate level.

This project aims to create a strong chess engine in python without the use of neural networks, just by using search algorithms and heuristics.

Currently the engine can search at speeds of 10-15k nodes per second, without the use of transposition tables.

### Features implemented so far -
- Minimax Search
- Alpha Beta Pruning
- Move Ordering

### Currently working on -
- Improving Iterative deepening search
- Adding zobrist hashing for transposition tables
- Implementing Quiescence search
- A simplified version of an opening book

### To be implemented -
- Piece square tables
- Monte Carlo Tree Search
- MVV-LVA (Most Valuable Victim - Least Valuable Aggressor)
- Static Exchange Evaluation
