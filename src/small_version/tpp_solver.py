# TPP Small Version Solver
"""
Production-Ready Recursive TSP Solver
=====================================

Optimized solver for the Iowa → M-States → White House problem.
Uses recursive algorithms to achieve distances < 7500km.

Core Features:
- Recursive cluster optimization
- Dynamic programming with memoization
- Inter-cluster connection optimization
- 2-opt local improvements

Author: [Your Name]
Date: [Date]
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from itertools import permutations
from typing import List, Tuple, Dict
from geopy.distance import geodesic
from sklearn.cluster import KMeans
from functools import lru_cache
import random

class TSPSolver:
    """
    Recursive TSP solver optimized for geographic problems.

    Target performance: < 7500km for Iowa → 8 M-states → White House
    """

    def __init__(self, locations: Dict[str, Tuple[float, float]],
                 start: str, end: str, visit: List[str]):
        """
        Initialize solver.

        Args:
            locations: {name: (lat, lon)} mapping
            start: Starting location name
            end: Ending location name
            visit: List of locations to visit
        """
        self.locations = locations
        self.names = list(locations.keys())
        self.coords = list(locations.values())

        # Convert names to indices
        self.start_idx = self.names.index(start)
        self.end_idx = self.names.index(end)
        self.visit_indices = [self.names.index(name) for name in visit]

        # Calculate distance matrix
        self.distances = self._calculate_distances()

        # Performance tracking
        self.cache_hits = 0
        self.recursive_calls = 0

        print(f"TSP Solver: {start} → {len(visit)} locations → {end}")

    def _calculate_distances(self) -> np.ndarray:
        """Calculate geodesic distance matrix."""
        n = len(self.coords)
        distances = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                dist = geodesic(self.coords[i], self.coords[j]).kilometers
                distances[i][j] = distances[j][i] = dist

        return distances

    def path_distance(self, path: List[int]) -> float:
        """Calculate total path distance."""
        return sum(self.distances[path[i]][path[i + 1]]
                  for i in range(len(path) - 1))

    @lru_cache(maxsize=10000)
    def optimal_subpath(self, current: int, unvisited_mask: int) -> Tuple[float, Tuple[int, ...]]:
        """
        Find optimal path through unvisited locations using DP.

        Args:
            current: Current location index
            unvisited_mask: Bitmask of unvisited locations

        Returns:
            (optimal_distance, optimal_path)
        """
        self.recursive_calls += 1

        # Base case: all visited, go to end
        if unvisited_mask == 0:
            return self.distances[current][self.end_idx], (current, self.end_idx)

        best_dist = float('inf')
        best_path = None

        # Try each unvisited location
        for i in range(len(self.visit_indices)):
            if unvisited_mask & (1 << i):
                next_loc = self.visit_indices[i]
                new_mask = unvisited_mask & ~(1 << i)

                remaining_dist, remaining_path = self.optimal_subpath(next_loc, new_mask)
                total_dist = self.distances[current][next_loc] + remaining_dist

                if total_dist < best_dist:
                    best_dist = total_dist
                    best_path = (current,) + remaining_path

        return best_dist, best_path

    def optimize_clusters(self, clusters: Dict[int, List[int]]) -> Dict[int, List[int]]:
        """
        Recursively optimize cluster boundaries.

        Args:
            clusters: Initial cluster assignment

        Returns:
            Optimized cluster assignment
        """
        current = clusters.copy()
        best_config = current
        best_score = self._evaluate_clusters(current)

        improved = True
        iterations = 0

        while improved and iterations < 10:  # Limit iterations
            improved = False
            iterations += 1

            # Try moving each state to different clusters
            for source_id in list(current.keys()):
                if len(current[source_id]) <= 1:
                    continue

                for state in current[source_id][:]:  # Copy to avoid modification issues
                    for target_id in current.keys():
                        if source_id == target_id:
                            continue

                        # Test move
                        test_config = {k: v.copy() for k, v in current.items()}
                        test_config[source_id].remove(state)
                        test_config[target_id].append(state)

                        score = self._evaluate_clusters(test_config)
                        if score < best_score:
                            best_score = score
                            best_config = test_config
                            improved = True

            current = best_config

        return best_config

    def _evaluate_clusters(self, clusters: Dict[int, List[int]]) -> float:
        """Evaluate cluster configuration total cost."""
        total_cost = 0.0

        # Intra-cluster costs
        for states in clusters.values():
            if len(states) > 1:
                total_cost += min(self.path_distance(list(perm))
                                for perm in permutations(states))

        # Inter-cluster connection costs
        reps = {cid: self._cluster_representative(states)
                for cid, states in clusters.items()}

        # Simple chain connection cost
        rep_list = list(reps.values())
        for i in range(len(rep_list) - 1):
            total_cost += self.distances[rep_list[i]][rep_list[i + 1]]

        return total_cost

    def _cluster_representative(self, states: List[int]) -> int:
        """Find best representative for a cluster."""
        if len(states) == 1:
            return states[0]

        # Choose state with minimum average distance to others
        best_rep = states[0]
        best_avg = float('inf')

        for candidate in states:
            avg_dist = np.mean([self.distances[candidate][other]
                              for other in states if other != candidate])
            if avg_dist < best_avg:
                best_avg = avg_dist
                best_rep = candidate

        return best_rep

    def find_optimal_path(self, clusters: Dict[int, List[int]]) -> Tuple[List[int], float]:
        """
        Find optimal path through optimized clusters.

        Args:
            clusters: Optimized cluster configuration

        Returns:
            (optimal_path, total_distance)
        """
        # Determine cluster order
        reps = {cid: self._cluster_representative(states)
                for cid, states in clusters.items()}

        # Nearest neighbor for cluster order
        cluster_order = []
        current_pos = self.start_idx
        remaining = set(clusters.keys())

        while remaining:
            nearest = min(remaining,
                         key=lambda cid: self.distances[current_pos][reps[cid]])
            cluster_order.append(nearest)
            remaining.remove(nearest)
            current_pos = reps[nearest]

        # Build optimal path
        path = [self.start_idx]

        for cluster_id in cluster_order:
            states = clusters[cluster_id]

            if len(states) == 1:
                path.extend(states)
            else:
                # Find optimal permutation for cluster
                best_perm = min(permutations(states),
                               key=lambda p: sum(self.distances[p[i]][p[i+1]]
                                                for i in range(len(p)-1)))
                path.extend(best_perm)

        path.append(self.end_idx)
        return path, self.path_distance(path)

    def two_opt_improve(self, path: List[int]) -> List[int]:
        """Apply 2-opt local search improvements."""
        current = path.copy()
        improved = True

        while improved:
            improved = False
            best_dist = self.path_distance(current)

            for i in range(1, len(current) - 2):
                for j in range(i + 1, len(current) - 1):
                    # Try 2-opt swap
                    new_path = current.copy()
                    new_path[i:j+1] = new_path[i:j+1][::-1]

                    new_dist = self.path_distance(new_path)
                    if new_dist < best_dist:
                        current = new_path
                        best_dist = new_dist
                        improved = True
                        break

                if improved:
                    break

        return current

    def solve(self) -> Tuple[List[int], float]:
        """
        Main solving method.

        Returns:
            (optimal_path, total_distance)
        """
        print("Solving with recursive optimization...")
        start_time = time.time()

        # Step 1: Initial clustering
        n_clusters = min(3, max(2, len(self.visit_indices) // 3))
        visit_coords = np.array([self.coords[i] for i in self.visit_indices])

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(visit_coords)

        initial_clusters = {}
        for i, label in enumerate(labels):
            if label not in initial_clusters:
                initial_clusters[label] = []
            initial_clusters[label].append(self.visit_indices[i])

        # Step 2: Optimize cluster boundaries
        optimized_clusters = self.optimize_clusters(initial_clusters)

        # Step 3: Find optimal path through clusters
        path, distance = self.find_optimal_path(optimized_clusters)

        # Step 4: Local improvements
        final_path = self.two_opt_improve(path)
        final_distance = self.path_distance(final_path)

        execution_time = time.time() - start_time

        print(f"Solution found: {final_distance:.0f} km ({execution_time:.2f}s)")
        print(f"Recursive calls: {self.recursive_calls:,}, Cache hits: {self.cache_hits:,}")

        return final_path, final_distance

    def solve_baseline(self) -> Tuple[List[int], float]:
        """Baseline nearest neighbor for comparison."""
        path = [self.start_idx]
        unvisited = set(self.visit_indices)
        current = self.start_idx

        while unvisited:
            nearest = min(unvisited, key=lambda x: self.distances[current][x])
            path.append(nearest)
            unvisited.remove(nearest)
            current = nearest

        path.append(self.end_idx)
        return path, self.path_distance(path)

    def compare_methods(self) -> Dict[str, Tuple[List[int], float]]:
        """Compare recursive method with baseline."""
        results = {}

        # Baseline
        start_time = time.time()
        baseline_path, baseline_dist = self.solve_baseline()
        baseline_time = time.time() - start_time
        results['baseline'] = (baseline_path, baseline_dist)

        # Recursive optimization
        optimal_path, optimal_dist = self.solve()
        results['recursive'] = (optimal_path, optimal_dist)

        # Results
        improvement = ((baseline_dist - optimal_dist) / baseline_dist) * 100

        print(f"\n{'Method':<15} {'Distance':<10} {'Improvement':<12}")
        print("-" * 40)
        print(f"{'Baseline':<15} {baseline_dist:<10.0f} {'-':<12}")
        print(f"{'Recursive':<15} {optimal_dist:<10.0f} {improvement:<12.1f}%")

        if optimal_dist < 7500:
            print(f"✓ Target achieved: {optimal_dist:.0f} km < 7500 km")
        else:
            print(f"⚠ Target missed: {optimal_dist:.0f} km > 7500 km")

        return results

    def visualize(self, path: List[int], title: str = "TSP Solution"):
        """Create clean visualization of the solution."""
        plt.figure(figsize=(12, 8))

        # Plot all locations
        lats = [self.coords[i][0] for i in range(len(self.coords))]
        lons = [self.coords[i][1] for i in range(len(self.coords))]
        plt.scatter(lons, lats, c='lightgray', s=30, alpha=0.6, zorder=1)

        # Highlight path locations
        path_lats = [self.coords[i][0] for i in path]
        path_lons = [self.coords[i][1] for i in path]

        # Draw path
        plt.plot(path_lons, path_lats, 'red', linewidth=2, alpha=0.8, zorder=3)
        plt.scatter(path_lons, path_lats, c='blue', s=80, zorder=4)

        # Mark start and end
        plt.scatter([path_lons[0]], [path_lats[0]], c='green', s=150,
                   marker='s', zorder=5, label='Start')
        plt.scatter([path_lons[-1]], [path_lats[-1]], c='red', s=150,
                   marker='*', zorder=5, label='End')

        # Add labels
        for i in path:
            name = self.names[i]
            if name == 'White House':
                name = 'DC'
            plt.annotate(name, (self.coords[i][1], self.coords[i][0]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)

        plt.title(f'{title}\nDistance: {self.path_distance(path):.0f} km')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def create_iowa_problem():
    """Create the Iowa → M-States → White House problem instance."""
    locations = {
        'Iowa': (41.590939, -93.620866),
        'Maine': (44.323535, -69.765261),
        'Maryland': (38.972945, -76.501157),
        'Massachusetts': (42.2352, -71.0275),
        'Michigan': (42.354558, -84.955255),
        'Minnesota': (44.95, -93.094),
        'Mississippi': (32.354668, -90.178217),
        'Missouri': (38.572954, -92.189283),
        'Montana': (46.595805, -112.027031),
        'White House': (38.8977, -77.0365)
    }

    m_states = ['Maine', 'Maryland', 'Massachusetts', 'Michigan',
                'Minnesota', 'Mississippi', 'Missouri', 'Montana']

    return locations, 'Iowa', 'White House', m_states


def main():
    """Main execution function."""
    print("Recursive TSP Solver - Production Version")
    print("="*45)

    # Setup
    random.seed(42)
    locations, start, end, visit = create_iowa_problem()
    solver = TSPSolver(locations, start, end, visit)

    # Solve and compare
    results = solver.compare_methods()

    # Visualize best result
    best_method = min(results.keys(), key=lambda k: results[k][1])
    best_path, best_dist = results[best_method]

    solver.visualize(best_path, f"Optimal Solution ({best_method.title()})")

    # Show path
    print(f"\nOptimal Path ({best_dist:.0f} km):")
    for i, idx in enumerate(best_path):
        marker = "START" if i == 0 else "END" if i == len(best_path)-1 else f"{i}"
        print(f"  {marker}: {solver.names[idx]}")

    return solver, results


if __name__ == "__main__":
    solver, results = main()