from __future__ import annotations
import argparse
from collections import deque
from dataclasses import dataclass
import heapq
import time
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

State = Tuple[Tuple[int, ...], ...]
Action = Tuple[int, int]


@dataclass
class SearchResult:
    found: bool
    actions: List[Action]
    states_expanded: int
    max_frontier_size: int
    path_cost: int


class LightsOut:
    def __init__(self, initial: Iterable[Iterable[int]], goal: Optional[Iterable[Iterable[int]]] = None):
        self.initial: State = self._to_state(initial)
        rows = len(self.initial)
        cols = len(self.initial[0]) if rows else 0

        if goal is None:
            self.goal: State = tuple(tuple(0 for _ in range(cols)) for _ in range(rows))
        else:
            self.goal = self._to_state(goal)

        if rows == 0 or cols == 0:
            raise ValueError("Board must be non-empty")

        for row in self.initial:
            if len(row) != cols:
                raise ValueError("Board must be rectangular")

        self.rows = rows
        self.cols = cols
        self.actions: List[Action] = [(r, c) for r in range(rows) for c in range(cols)]

    @staticmethod
    def _to_state(board: Iterable[Iterable[int]]) -> State:
        return tuple(tuple(int(cell) for cell in row) for row in board)

    def is_goal(self, state: State) -> bool:
        return state == self.goal

    def neighbors(self, state: State) -> Iterable[Tuple[Action, State, int]]:
        for action in self.actions:
            yield action, self.apply_action(state, action), 1

    def apply_action(self, state: State, action: Action) -> State:
        r, c = action
        grid = [list(row) for row in state]

        for rr, cc in ((r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < self.rows and 0 <= cc < self.cols:
                grid[rr][cc] ^= 1

        return tuple(tuple(row) for row in grid)


def reconstruct_actions(parent: Dict[State, Tuple[Optional[State], Optional[Action]]], goal_state: State) -> List[Action]:
    actions: List[Action] = []
    cur = goal_state
    while True:
        prev_state, action = parent[cur]
        if prev_state is None:
            break
        actions.append(action)
        cur = prev_state
    actions.reverse()
    return actions


def h_count_lit(state: State, _: Optional[LightsOut] = None) -> int:
    return sum(sum(row) for row in state)


def bfs(problem: LightsOut) -> SearchResult:
    start = problem.initial
    if problem.is_goal(start):
        return SearchResult(True, [], 0, 1, 0)

    queue = deque([start])
    parent: Dict[State, Tuple[Optional[State], Optional[Action]]] = {start: (None, None)}
    visited: Set[State] = {start}
    expanded = 0
    max_frontier = 1

    while queue:
        state = queue.popleft()
        expanded += 1

        for action, nxt, step_cost in problem.neighbors(state):
            if nxt in visited:
                continue
            visited.add(nxt)
            parent[nxt] = (state, action)

            if problem.is_goal(nxt):
                actions = reconstruct_actions(parent, nxt)
                return SearchResult(True, actions, expanded, max_frontier, len(actions) * step_cost)

            queue.append(nxt)
        max_frontier = max(max_frontier, len(queue))

    return SearchResult(False, [], expanded, max_frontier, 0)


def dfs(problem: LightsOut) -> SearchResult:
    start = problem.initial
    if problem.is_goal(start):
        return SearchResult(True, [], 0, 1, 0)

    stack = [start]
    parent: Dict[State, Tuple[Optional[State], Optional[Action]]] = {start: (None, None)}
    visited: Set[State] = set()
    expanded = 0
    max_frontier = 1

    while stack:
        state = stack.pop()
        if state in visited:
            continue

        visited.add(state)
        expanded += 1

        if problem.is_goal(state):
            actions = reconstruct_actions(parent, state)
            return SearchResult(True, actions, expanded, max_frontier, len(actions))

        neighbors = list(problem.neighbors(state))
        for action, nxt, _ in reversed(neighbors):
            if nxt in visited or nxt in parent:
                continue
            parent[nxt] = (state, action)
            stack.append(nxt)
        max_frontier = max(max_frontier, len(stack))

    return SearchResult(False, [], expanded, max_frontier, 0)


def ucs(problem: LightsOut) -> SearchResult:
    start = problem.initial
    if problem.is_goal(start):
        return SearchResult(True, [], 0, 1, 0)

    pq: List[Tuple[int, int, State]] = [(0, 0, start)]
    parent: Dict[State, Tuple[Optional[State], Optional[Action]]] = {start: (None, None)}
    best_cost: Dict[State, int] = {start: 0}
    expanded = 0
    max_frontier = 1
    tie = 1

    while pq:
        g, _, state = heapq.heappop(pq)
        if g > best_cost.get(state, float("inf")):
            continue

        expanded += 1
        if problem.is_goal(state):
            actions = reconstruct_actions(parent, state)
            return SearchResult(True, actions, expanded, max_frontier, g)

        for action, nxt, step_cost in problem.neighbors(state):
            new_g = g + step_cost
            if new_g < best_cost.get(nxt, float("inf")):
                best_cost[nxt] = new_g
                parent[nxt] = (state, action)
                heapq.heappush(pq, (new_g, tie, nxt))
                tie += 1

        max_frontier = max(max_frontier, len(pq))

    return SearchResult(False, [], expanded, max_frontier, 0)


def greedy_best_first(problem: LightsOut, heuristic: Callable[[State, Optional[LightsOut]], int] = h_count_lit) -> SearchResult:
    start = problem.initial
    if problem.is_goal(start):
        return SearchResult(True, [], 0, 1, 0)

    pq: List[Tuple[int, int, State]] = [(heuristic(start, problem), 0, start)]
    parent: Dict[State, Tuple[Optional[State], Optional[Action]]] = {start: (None, None)}
    visited: Set[State] = set()
    depth: Dict[State, int] = {start: 0}
    expanded = 0
    max_frontier = 1
    tie = 1

    while pq:
        _, _, state = heapq.heappop(pq)
        if state in visited:
            continue

        visited.add(state)
        expanded += 1

        if problem.is_goal(state):
            actions = reconstruct_actions(parent, state)
            return SearchResult(True, actions, expanded, max_frontier, depth[state])

        for action, nxt, _ in problem.neighbors(state):
            if nxt in visited:
                continue
            if nxt not in parent:
                parent[nxt] = (state, action)
                depth[nxt] = depth[state] + 1
                heapq.heappush(pq, (heuristic(nxt, problem), tie, nxt))
                tie += 1

        max_frontier = max(max_frontier, len(pq))

    return SearchResult(False, [], expanded, max_frontier, 0)


def a_star(problem: LightsOut, heuristic: Callable[[State, Optional[LightsOut]], int] = h_count_lit) -> SearchResult:
    start = problem.initial
    if problem.is_goal(start):
        return SearchResult(True, [], 0, 1, 0)

    pq: List[Tuple[int, int, State]] = [(heuristic(start, problem), 0, start)]
    parent: Dict[State, Tuple[Optional[State], Optional[Action]]] = {start: (None, None)}
    best_cost: Dict[State, int] = {start: 0}
    expanded = 0
    max_frontier = 1
    tie = 1

    while pq:
        _, _, state = heapq.heappop(pq)
        g = best_cost[state]
        if problem.is_goal(state):
            actions = reconstruct_actions(parent, state)
            return SearchResult(True, actions, expanded, max_frontier, g)

        expanded += 1

        for action, nxt, step_cost in problem.neighbors(state):
            new_g = g + step_cost
            if new_g < best_cost.get(nxt, float("inf")):
                best_cost[nxt] = new_g
                parent[nxt] = (state, action)
                f = new_g + heuristic(nxt, problem)
                heapq.heappush(pq, (f, tie, nxt))
                tie += 1

        max_frontier = max(max_frontier, len(pq))

    return SearchResult(False, [], expanded, max_frontier, 0)


def pretty_state(state: State) -> str:
    return "\n".join(" ".join(str(c) for c in row) for row in state)


def state_after_actions(problem: LightsOut, actions: List[Action]) -> List[State]:
    states = [problem.initial]
    cur = problem.initial
    for action in actions:
        cur = problem.apply_action(cur, action)
        states.append(cur)
    return states


def run_all(problem: LightsOut) -> Dict[str, SearchResult]:
    return {
        "BFS": bfs(problem),
        "DFS": dfs(problem),
        "UCS": ucs(problem),
        "Greedy Best-First": greedy_best_first(problem),
        "A*": a_star(problem),
    }

BOARD_SIZE = 3
TOTAL_BOARDS = 2 ** (BOARD_SIZE * BOARD_SIZE)


def iter_all_boards(size: int = BOARD_SIZE) -> Iterable[List[List[int]]]:
    total = 2 ** (size * size)
    for mask in range(total):
        cells = [(mask >> i) & 1 for i in range(size * size)]
        board = [cells[r * size : (r + 1) * size] for r in range(size)]
        yield board


def parse_board_string(text: str, size: int = BOARD_SIZE) -> List[List[int]]:
    raw = text.strip().replace(" ", "")
    if "/" in raw:
        rows = raw.split("/")
        if any(len(row) != len(rows[0]) for row in rows):
            raise ValueError("Board rows must have the same length.")
        board = [[int(c) for c in row] for row in rows]
    else:
        if len(raw) != size * size:
            raise ValueError(f"Board string must be {size * size} characters of 0/1.")
        board = [[int(raw[r * size + c]) for c in range(size)] for r in range(size)]

    for row in board:
        for cell in row:
            if cell not in (0, 1):
                raise ValueError("Board must contain only 0 or 1 values.")
    return board


def prompt_board_input(size: int = BOARD_SIZE) -> List[List[int]]:
    print(f"Enter a {size}x{size} board as 0/1 rows.")
    rows: List[List[int]] = []
    for r in range(size):
        while True:
            line = input(f"Row {r + 1} (e.g. 101): ").strip().replace(" ", "")
            if len(line) != size or any(c not in "01" for c in line):
                print(f"Row must be exactly {size} characters of 0/1.")
                continue
            rows.append([int(c) for c in line])
            break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Lights Out search (BFS/DFS/UCS/Greedy/A*).")
    parser.add_argument("--all-boards", action="store_true", help="Run all 512 possible 3x3 boards and print consolidated metrics.")
    parser.add_argument("--board", type=str, help="Board string (""101/110/011"" or ""101110011"").")
    parser.add_argument("--prompt-board", action="store_true", help="Prompt for a board in the terminal.")
    parser.add_argument("--delay", type=float, default=0.25, help="Animation delay in seconds (default: 0.25).")
    args = parser.parse_args()

    if args.all_boards:
        start_time = time.perf_counter()
        aggregate: Dict[str, Dict[str, float]] = {}
        counts: Dict[str, int] = {}
        solved: Dict[str, int] = {}
        worst_expanded: Dict[str, int] = {}
        worst_frontier: Dict[str, int] = {}

        for board in iter_all_boards(BOARD_SIZE):
            problem = LightsOut(board)
            results = run_all(problem)
            for name, result in results.items():
                if name not in aggregate:
                    aggregate[name] = {"path_cost": 0.0, "expanded": 0.0, "frontier": 0.0, "actions": 0.0}
                    counts[name] = 0
                    solved[name] = 0
                    worst_expanded[name] = 0
                    worst_frontier[name] = 0

                counts[name] += 1
                if result.found:
                    solved[name] += 1
                    aggregate[name]["path_cost"] += result.path_cost
                    aggregate[name]["actions"] += len(result.actions)
                aggregate[name]["expanded"] += result.states_expanded
                aggregate[name]["frontier"] += result.max_frontier_size
                worst_expanded[name] = max(worst_expanded[name], result.states_expanded)
                worst_frontier[name] = max(worst_frontier[name], result.max_frontier_size)

        elapsed = time.perf_counter() - start_time
        print(f"Ran all {TOTAL_BOARDS} boards in {elapsed:.3f}s")
        print()
        for name in sorted(aggregate.keys()):
            total = counts[name]
            solved_count = solved[name]
            solved_den = max(1, solved_count)
            print(f"{name}:")
            print(f"  Solved: {solved_count}/{total}")
            print(f"  Avg path cost (solved only): {aggregate[name]['path_cost'] / solved_den:.3f}")
            print(f"  Avg actions (solved only): {aggregate[name]['actions'] / solved_den:.3f}")
            print(f"  Avg states expanded: {aggregate[name]['expanded'] / total:.3f}")
            print(f"  Avg max frontier: {aggregate[name]['frontier'] / total:.3f}")
            print(f"  Worst states expanded: {worst_expanded[name]}")
            print(f"  Worst max frontier: {worst_frontier[name]}")
            print()
        return

    if args.prompt_board:
        initial_board = prompt_board_input(BOARD_SIZE)
    elif args.board:
        initial_board = parse_board_string(args.board, BOARD_SIZE)
    else:
        initial_board = [
            [1, 0, 1],
            [1, 1, 0],
            [0, 1, 1],
        ]

    problem = LightsOut(initial_board)
    results = run_all(problem)

    print("Initial state:")
    print(pretty_state(problem.initial))
    print()

    for name, result in results.items():
        print(f"{name}:")
        print(f"  Found solution: {result.found}")
        print(f"  Path cost: {result.path_cost}")
        print(f"  States expanded: {result.states_expanded}")
        print(f"  Max frontier size: {result.max_frontier_size}")
        print(f"  Actions: {result.actions}")
        print()

if __name__ == "__main__":
    main()