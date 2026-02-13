# Lights Out Search

A compact Lights Out search project featuring classic uninformed and informed search algorithms, and a standalone HTML visualizer with interactive input.

## Features
- Algorithms: BFS, DFS, UCS, Greedy Best-First, A*
- Evaluation across all 512 possible 3x3 boards
- Standalone HTML visualizer with clickable grid, board string input, and step playback

## Files
- `LightsOut-Search.py`: CLI runner and search implementations
- `LightsOut-Visualizer.html`: Browser-based visualizer and solver

## Requirements
- Python 3.8+
- A modern browser for the HTML visualizer

## CLI Usage
### Run all 512 boards and show consolidated metrics
```bash
python3 LightsOut-Search.py --all-boards
```

### Solve a specific board
```bash
python3 LightsOut-Search.py --board 101/110/011
```

### Prompt for board input
```bash
python3 LightsOut-Search.py --prompt-board
```

Board formats accepted:
- Slash-separated rows: `101/110/011`
- Flattened digits: `101110011`

## HTML Visualizer
Open `LightsOut-Visualizer.html` in your browser.

How to use:
- Click cells to toggle lights
- Or enter a board string and click **Apply**
- Choose an algorithm and click **Solve**
- Step through actions with Prev/Next/Play

## Notes
- Board size is currently fixed at 3x3.
- Greedy Best-First and A* use the “count of lit tiles” heuristic.
