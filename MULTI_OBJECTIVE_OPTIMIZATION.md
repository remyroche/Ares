# Multi-Objective Optimization for HMM Regime Discovery

## 🎯 **Overview**

Multi-objective optimization allows the system to simultaneously optimize multiple conflicting objectives, such as regime quality, computational efficiency, and interpretability, leading to more balanced and practical solutions.

## 🔄 **Multi-Objective Problem Formulation**

### **Primary Objectives**

1. **Regime Quality** (Maximize)
   - Regime differentiation
   - Internal coherence
   - Market condition correlation

2. **Computational Efficiency** (Minimize)
   - Training time
   - Memory usage
   - Inference speed

3. **Interpretability** (Maximize)
   - Regime count (15-20)
   - Regime balance
   - Feature simplicity

4. **Robustness** (Maximize)
   - Cross-validation stability
   - Out-of-sample performance
   - Parameter sensitivity

### **Objective Functions**

```python
@dataclass
class MultiObjectiveMetrics:
    """Container for multi-objective optimization metrics."""

    # Primary objectives
    regime_quality: float          # 0-1, higher is better
    computational_efficiency: float # 0-1, higher is better (lower time/memory)
    interpretability: float        # 0-1, higher is better
    robustness: float             # 0-1, higher is better

    # Derived metrics
    pareto_rank: int = 0
    crowding_distance: float = 0.0
    domination_count: int = 0
```

## 🎛️ **Multi-Objective Optimization Algorithms**

### **1. NSGA-II (Non-dominated Sorting Genetic Algorithm II)**

```python
class NSGAIIOptimizer:
    """NSGA-II optimizer for multi-objective HMM regime optimization."""

    def __init__(self, population_size=100, generations=50):
        self.population_size = population_size
        self.generations = generations
        self.population = []
        self.archive = []  # Pareto front archive

    def optimize(self, data, feature_columns, market_condition_columns):
        """Run NSGA-II optimization."""

        # Initialize population
        self.population = self._initialize_population()

        for generation in range(self.generations):
            # Evaluate objectives
            self._evaluate_population(data, feature_columns, market_condition_columns)

            # Non-dominated sorting
            fronts = self._non_dominated_sort()

            # Crowding distance calculation
            self._calculate_crowding_distance(fronts)

            # Selection
            parents = self._tournament_selection()

            # Crossover and mutation
            offspring = self._generate_offspring(parents)

            # Combine parent and offspring populations
            combined_population = self.population + offspring

            # Environmental selection
            self.population = self._environmental_selection(combined_population)

            # Update archive
            self._update_archive()

        return self.archive

    def _evaluate_objectives(self, individual, data, feature_columns, market_condition_columns):
        """Evaluate all objectives for a single individual."""

        # Generate clusters
        cluster_data = self._generate_clusters(individual, data)

        # Calculate objectives
        regime_quality = self._calculate_regime_quality(cluster_data, market_condition_columns)
        computational_efficiency = self._calculate_computational_efficiency(individual)
        interpretability = self._calculate_interpretability(cluster_data, individual)
        robustness = self._calculate_robustness(cluster_data, data)

        return MultiObjectiveMetrics(
            regime_quality=regime_quality,
            computational_efficiency=computational_efficiency,
            interpretability=interpretability,
            robustness=robustness
        )

    def _non_dominated_sort(self):
        """Perform non-dominated sorting."""

        fronts = [[]]

        for individual in self.population:
            individual.domination_count = 0
            individual.dominated_solutions = []

            for other in self.population:
                if self._dominates(individual, other):
                    individual.dominated_solutions.append(other)
                elif self._dominates(other, individual):
                    individual.domination_count += 1

            if individual.domination_count == 0:
                individual.pareto_rank = 0
                fronts[0].append(individual)

        i = 0
        while fronts[i]:
            next_front = []
            for individual in fronts[i]:
                for dominated in individual.dominated_solutions:
                    dominated.domination_count -= 1
                    if dominated.domination_count == 0:
                        dominated.pareto_rank = i + 1
                        next_front.append(dominated)
            i += 1
            if next_front:
                fronts.append(next_front)

        return fronts

    def _dominates(self, individual1, individual2):
        """Check if individual1 dominates individual2."""

        objectives1 = individual1.objectives
        objectives2 = individual2.objectives

        # Check if individual1 is at least as good in all objectives
        at_least_as_good = all(obj1 >= obj2 for obj1, obj2 in zip(objectives1, objectives2))

        # Check if individual1 is strictly better in at least one objective
        strictly_better = any(obj1 > obj2 for obj1, obj2 in zip(objectives1, objectives2))

        return at_least_as_good and strictly_better
```

### **2. MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition)**

```python
class MOEADOptimizer:
    """MOEA/D optimizer for multi-objective HMM regime optimization."""

    def __init__(self, population_size=100, generations=50, neighborhood_size=20):
        self.population_size = population_size
        self.generations = generations
        self.neighborhood_size = neighborhood_size
        self.weight_vectors = self._generate_weight_vectors()
        self.neighborhoods = self._calculate_neighborhoods()

    def _generate_weight_vectors(self):
        """Generate weight vectors for decomposition."""

        weight_vectors = []

        # Generate evenly distributed weight vectors
        for i in range(self.population_size):
            # Use systematic approach to generate weights
            weights = self._systematic_weights(i, self.population_size, 4)  # 4 objectives
            weight_vectors.append(weights)

        return weight_vectors

    def _systematic_weights(self, index, population_size, num_objectives):
        """Generate systematic weight vectors."""

        # Use Das and Dennis's systematic approach
        H = int(np.sqrt(population_size))  # H parameter for weight generation

        weights = []
        for i in range(num_objectives):
            weight = (H - i) / H
            weights.append(weight)

        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)

        return weights

    def _calculate_neighborhoods(self):
        """Calculate neighborhoods for each weight vector."""

        neighborhoods = []

        for i, weight_i in enumerate(self.weight_vectors):
            distances = []
            for j, weight_j in enumerate(self.weight_vectors):
                if i != j:
                    distance = np.linalg.norm(weight_i - weight_j)
                    distances.append((distance, j))

            # Sort by distance and take top neighbors
            distances.sort()
            neighborhood = [j for _, j in distances[:self.neighborhood_size]]
            neighborhoods.append(neighborhood)

        return neighborhoods

    def _tchebicheff_decomposition(self, individual, weight_vector, reference_point):
        """Calculate Tchebicheff decomposition value."""

        objectives = individual.objectives
        max_value = 0

        for i, (obj, weight) in enumerate(zip(objectives, weight_vector)):
            if weight > 0:
                value = abs(obj - reference_point[i]) / weight
                max_value = max(max_value, value)

        return max_value
```

### **3. SPEA2 (Strength Pareto Evolutionary Algorithm 2)**

```python
class SPEA2Optimizer:
    """SPEA2 optimizer for multi-objective HMM regime optimization."""

    def __init__(self, population_size=100, archive_size=100, generations=50):
        self.population_size = population_size
        self.archive_size = archive_size
        self.generations = generations
        self.population = []
        self.archive = []

    def optimize(self, data, feature_columns, market_condition_columns):
        """Run SPEA2 optimization."""

        # Initialize population
        self.population = self._initialize_population()

        for generation in range(self.generations):
            # Evaluate fitness
            self._evaluate_fitness()

            # Environmental selection
            self._environmental_selection()

            # Mating selection
            parents = self._mating_selection()

            # Variation
            offspring = self._variation(parents)

            # Update population
            self.population = offspring

        return self.archive

    def _evaluate_fitness(self):
        """Evaluate fitness for all individuals."""

        # Calculate strength values
        for individual in self.population + self.archive:
            individual.strength = 0
            individual.raw_fitness = 0

        # Calculate strength (number of dominated solutions)
        for individual in self.population + self.archive:
            for other in self.population + self.archive:
                if self._dominates(individual, other):
                    individual.strength += 1

        # Calculate raw fitness
        for individual in self.population + self.archive:
            for other in self.population + self.archive:
                if self._dominates(other, individual):
                    individual.raw_fitness += other.strength

        # Calculate density estimation
        self._calculate_density()

    def _calculate_density(self):
        """Calculate density estimation using k-nearest neighbor."""

        k = int(np.sqrt(len(self.population) + len(self.archive)))

        for individual in self.population + self.archive:
            distances = []
            for other in self.population + self.archive:
                if individual != other:
                    distance = self._calculate_distance(individual, other)
                    distances.append(distance)

            distances.sort()
            if len(distances) >= k:
                individual.density = 1 / (distances[k-1] + 2)
            else:
                individual.density = 1 / (distances[-1] + 2)

    def _calculate_distance(self, individual1, individual2):
        """Calculate distance between two individuals in objective space."""

        objectives1 = individual1.objectives
        objectives2 = individual2.objectives

        return np.linalg.norm(np.array(objectives1) - np.array(objectives2))
```

## 📊 **Objective Function Definitions**

### **1. Regime Quality Objective**

```python
def calculate_regime_quality_objective(cluster_data, market_condition_columns, params):
    """Calculate regime quality objective."""

    # Calculate individual metrics
    differentiation = calculate_regime_differentiation(cluster_data, market_condition_columns)
    coherence = calculate_internal_coherence(cluster_data, market_condition_columns)
    persistence = calculate_regime_persistence(cluster_data)
    smoothness = calculate_transition_smoothness(cluster_data)

    # Weighted combination
    weights = [0.4, 0.3, 0.2, 0.1]
    metrics = [differentiation, coherence, persistence, smoothness]

    regime_quality = np.average(metrics, weights=weights)

    return regime_quality
```

### **2. Computational Efficiency Objective**

```python
def calculate_computational_efficiency_objective(params, execution_time, memory_usage):
    """Calculate computational efficiency objective."""

    # Normalize execution time (0-1, lower is better)
    max_expected_time = 300  # 5 minutes
    normalized_time = min(execution_time / max_expected_time, 1.0)

    # Normalize memory usage (0-1, lower is better)
    max_expected_memory = 8 * 1024 * 1024 * 1024  # 8GB
    normalized_memory = min(memory_usage / max_expected_memory, 1.0)

    # Parameter complexity penalty
    complexity_penalty = calculate_parameter_complexity(params)

    # Combined efficiency score
    efficiency = (1 - normalized_time) * 0.4 + (1 - normalized_memory) * 0.4 + (1 - complexity_penalty) * 0.2

    return efficiency

def calculate_parameter_complexity(params):
    """Calculate parameter complexity penalty."""

    complexity = 0

    # HMM complexity
    complexity += params.get('n_components', 5) / 10  # Normalize by max components

    # Clustering complexity
    complexity += params.get('n_clusters', 10) / 20  # Normalize by max clusters

    # Feature complexity
    if params.get('use_dimensionality_reduction', False):
        complexity += 0.2

    if params.get('use_feature_interactions', False):
        complexity += 0.3

    return min(complexity, 1.0)
```

### **3. Interpretability Objective**

```python
def calculate_interpretability_objective(cluster_data, params):
    """Calculate interpretability objective."""

    interpretability = 0

    # Regime count penalty (prefer 15-20 regimes)
    n_regimes = len(cluster_data['composite_cluster_id'].unique())
    target_regimes = 18

    if 15 <= n_regimes <= 20:
        regime_count_score = 1.0
    else:
        penalty = abs(n_regimes - target_regimes) / target_regimes
        regime_count_score = max(0, 1 - penalty)

    # Regime balance score
    regime_sizes = cluster_data['composite_cluster_id'].value_counts()
    balance_score = 1.0 / (1.0 + regime_sizes.std() / regime_sizes.mean())

    # Parameter simplicity score
    simplicity_score = calculate_parameter_simplicity(params)

    # Feature simplicity score
    feature_simplicity = calculate_feature_simplicity(params)

    # Weighted combination
    weights = [0.3, 0.3, 0.2, 0.2]
    scores = [regime_count_score, balance_score, simplicity_score, feature_simplicity]

    interpretability = np.average(scores, weights=weights)

    return interpretability

def calculate_parameter_simplicity(params):
    """Calculate parameter simplicity score."""

    simplicity = 1.0

    # Penalize complex covariance types
    if params.get('covariance_type') == 'full':
        simplicity -= 0.2

    # Penalize complex merging methods
    if params.get('merging_method') in ['spectral', 'dbscan']:
        simplicity -= 0.1

    # Penalize high iteration counts
    max_iter = params.get('n_iter', 100)
    simplicity -= min(max_iter / 500, 0.2)

    return max(simplicity, 0.0)
```

### **4. Robustness Objective**

```python
def calculate_robustness_objective(cluster_data, data, params):
    """Calculate robustness objective."""

    robustness = 0

    # Cross-validation stability
    cv_stability = calculate_cv_stability(data, params)

    # Bootstrap confidence
    bootstrap_confidence = calculate_bootstrap_confidence(data, params)

    # Parameter sensitivity
    sensitivity_score = calculate_parameter_sensitivity(data, params)

    # Out-of-sample performance
    oos_performance = calculate_out_of_sample_performance(data, params)

    # Weighted combination
    weights = [0.3, 0.3, 0.2, 0.2]
    scores = [cv_stability, bootstrap_confidence, sensitivity_score, oos_performance]

    robustness = np.average(scores, weights=weights)

    return robustness

def calculate_cv_stability(data, params):
    """Calculate cross-validation stability."""

    cv_scores = []

    # Perform time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)

    for train_idx, val_idx in tscv.split(data):
        train_data = data.iloc[train_idx]
        val_data = data.iloc[val_idx]

        # Generate clusters on training data
        train_clusters = generate_clusters(train_data, params)

        # Evaluate on validation data
        score = evaluate_regime_quality(val_data, train_clusters)
        cv_scores.append(score)

    # Calculate stability as inverse of standard deviation
    stability = 1.0 / (1.0 + np.std(cv_scores))

    return stability
```

## 🎯 **Pareto Front Analysis**

### **1. Pareto Front Visualization**

```python
def visualize_pareto_front(archive, objectives=['regime_quality', 'computational_efficiency', 'interpretability', 'robustness']):
    """Visualize Pareto front."""

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    if len(objectives) == 2:
        # 2D visualization
        fig, ax = plt.subplots(figsize=(10, 8))

        x = [individual.objectives[0] for individual in archive]
        y = [individual.objectives[1] for individual in archive]

        ax.scatter(x, y, c='blue', alpha=0.6, s=50)
        ax.set_xlabel(objectives[0])
        ax.set_ylabel(objectives[1])
        ax.set_title('Pareto Front')
        ax.grid(True, alpha=0.3)

    elif len(objectives) == 3:
        # 3D visualization
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        x = [individual.objectives[0] for individual in archive]
        y = [individual.objectives[1] for individual in archive]
        z = [individual.objectives[2] for individual in archive]

        ax.scatter(x, y, z, c='blue', alpha=0.6, s=50)
        ax.set_xlabel(objectives[0])
        ax.set_ylabel(objectives[1])
        ax.set_zlabel(objectives[2])
        ax.set_title('Pareto Front (3D)')

    plt.tight_layout()
    plt.show()
```

### **2. Pareto Front Analysis**

```python
def analyze_pareto_front(archive):
    """Analyze Pareto front characteristics."""

    analysis = {}

    # Calculate hypervolume
    analysis['hypervolume'] = calculate_hypervolume(archive)

    # Calculate spread
    analysis['spread'] = calculate_spread(archive)

    # Calculate uniformity
    analysis['uniformity'] = calculate_uniformity(archive)

    # Find knee points
    analysis['knee_points'] = find_knee_points(archive)

    # Calculate crowding distance
    analysis['crowding_distances'] = calculate_crowding_distances(archive)

    return analysis

def find_knee_points(archive):
    """Find knee points in Pareto front."""

    knee_points = []

    for i, individual in enumerate(archive):
        # Calculate angle with neighbors
        angles = []

        for j, other in enumerate(archive):
            if i != j:
                angle = calculate_angle(individual, other)
                angles.append(angle)

        # If angle is significantly different from neighbors, it's a knee point
        if angles:
            mean_angle = np.mean(angles)
            std_angle = np.std(angles)

            if abs(angles[0] - mean_angle) > 2 * std_angle:
                knee_points.append(individual)

    return knee_points
```

## 🔧 **Decision Making**

### **1. Weighted Sum Method**

```python
def weighted_sum_decision(archive, weights):
    """Select solution using weighted sum method."""

    best_solution = None
    best_score = -np.inf

    for individual in archive:
        # Calculate weighted sum
        weighted_sum = sum(w * obj for w, obj in zip(weights, individual.objectives))

        if weighted_sum > best_score:
            best_score = weighted_sum
            best_solution = individual

    return best_solution
```

### **2. TOPSIS Method**

```python
def topsis_decision(archive):
    """Select solution using TOPSIS method."""

    # Normalize objectives
    objectives_matrix = np.array([individual.objectives for individual in archive])
    normalized_matrix = objectives_matrix / np.sqrt(np.sum(objectives_matrix**2, axis=0))

    # Determine ideal and anti-ideal solutions
    ideal_solution = np.max(normalized_matrix, axis=0)
    anti_ideal_solution = np.min(normalized_matrix, axis=0)

    # Calculate distances
    distances_to_ideal = np.sqrt(np.sum((normalized_matrix - ideal_solution)**2, axis=1))
    distances_to_anti_ideal = np.sqrt(np.sum((normalized_matrix - anti_ideal_solution)**2, axis=1))

    # Calculate relative closeness
    relative_closeness = distances_to_anti_ideal / (distances_to_ideal + distances_to_anti_ideal)

    # Select solution with maximum relative closeness
    best_index = np.argmax(relative_closeness)

    return archive[best_index]
```

### **3. Interactive Decision Making**

```python
class InteractiveDecisionMaker:
    """Interactive decision making for Pareto front selection."""

    def __init__(self, archive):
        self.archive = archive
        self.selected_solutions = []

    def interactive_selection(self):
        """Interactive Pareto front selection."""

        print("Interactive Pareto Front Selection")
        print("=" * 50)

        while True:
            # Display current solutions
            self._display_solutions()

            # Get user preference
            choice = input("\nEnter solution number to select (or 'q' to quit): ")

            if choice.lower() == 'q':
                break

            try:
                solution_index = int(choice)
                if 0 <= solution_index < len(self.archive):
                    selected_solution = self.archive[solution_index]
                    self.selected_solutions.append(selected_solution)
                    print(f"Selected solution {solution_index}")
                else:
                    print("Invalid solution number")
            except ValueError:
                print("Invalid input")

        return self.selected_solutions

    def _display_solutions(self):
        """Display current solutions."""

        print("\nAvailable Solutions:")
        print("-" * 80)
        print(f"{'Index':<6} {'Regime Quality':<15} {'Efficiency':<12} {'Interpretability':<15} {'Robustness':<12}")
        print("-" * 80)

        for i, individual in enumerate(self.archive):
            objectives = individual.objectives
            print(f"{i:<6} {objectives[0]:<15.4f} {objectives[1]:<12.4f} {objectives[2]:<15.4f} {objectives[3]:<12.4f}")
```

## 📊 **Performance Metrics**

### **1. Hypervolume Indicator**

```python
def calculate_hypervolume(archive, reference_point=None):
    """Calculate hypervolume indicator."""

    if reference_point is None:
        # Use nadir point as reference
        objectives_matrix = np.array([individual.objectives for individual in archive])
        reference_point = np.min(objectives_matrix, axis=0) - 0.1

    # Calculate hypervolume using Monte Carlo method
    n_samples = 10000
    volume = 0

    for _ in range(n_samples):
        # Generate random point
        random_point = np.random.uniform(reference_point, np.max(objectives_matrix, axis=0))

        # Check if point is dominated
        dominated = False
        for individual in archive:
            if all(obj >= ref for obj, ref in zip(individual.objectives, random_point)):
                dominated = True
                break

        if not dominated:
            volume += 1

    hypervolume = volume / n_samples

    return hypervolume
```

### **2. Epsilon Indicator**

```python
def calculate_epsilon_indicator(archive1, archive2):
    """Calculate epsilon indicator between two Pareto fronts."""

    epsilon = 0

    for individual1 in archive1:
        min_epsilon = np.inf

        for individual2 in archive2:
            # Calculate epsilon for this pair
            pair_epsilon = max(obj1 - obj2 for obj1, obj2 in zip(individual1.objectives, individual2.objectives))
            min_epsilon = min(min_epsilon, pair_epsilon)

        epsilon = max(epsilon, min_epsilon)

    return epsilon
```

This comprehensive multi-objective optimization framework ensures that the HMM regime discovery system can balance multiple competing objectives, leading to more practical and robust solutions for trading strategies.