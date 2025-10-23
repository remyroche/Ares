"""
Model Predictive Control (MPC) for Truffle Cultivation
Controls pH, EC, DO, temperature, and flow rates in hydroponic systems
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import logging
from scipy.optimize import minimize
import casadi as ca
from casadi import SX, MX, Function, vertcat, horzcat
import time

logger = logging.getLogger(__name__)

@dataclass
class ControlVariable:
    """Represents a controllable variable in the system"""
    name: str
    min_value: float
    max_value: float
    initial_value: float
    unit: str
    cost_weight: float = 1.0

@dataclass
class StateVariable:
    """Represents a state variable in the system"""
    name: str
    initial_value: float
    setpoint: float
    tolerance: float
    unit: str
    cost_weight: float = 1.0

@dataclass
class Disturbance:
    """Represents a disturbance in the system"""
    name: str
    value: float
    unit: str

class TruffleMPCController:
    """Model Predictive Controller for truffle cultivation systems"""
    
    def __init__(self, prediction_horizon: int = 20, control_horizon: int = 10,
                 dt: float = 1.0):
        """
        Initialize the MPC controller
        
        Args:
            prediction_horizon: Number of prediction steps
            control_horizon: Number of control steps
            dt: Time step size (seconds)
        """
        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon
        self.dt = dt
        
        # Control variables (manipulated variables)
        self.control_vars = {
            'pH_dosing': ControlVariable('pH_dosing', 0.0, 10.0, 0.0, 'mL/min', 1.0),
            'EC_dosing': ControlVariable('EC_dosing', 0.0, 5.0, 0.0, 'mL/min', 1.0),
            'air_flow': ControlVariable('air_flow', 0.0, 100.0, 50.0, 'L/min', 0.5),
            'water_flow': ControlVariable('water_flow', 0.0, 10.0, 2.0, 'L/min', 0.8),
            'heating': ControlVariable('heating', 0.0, 100.0, 0.0, '%', 0.3),
            'cooling': ControlVariable('cooling', 0.0, 100.0, 0.0, '%', 0.3)
        }
        
        # State variables (controlled variables)
        self.state_vars = {
            'pH': StateVariable('pH', 6.0, 6.0, 0.2, 'pH', 10.0),
            'EC': StateVariable('EC', 1.2, 1.2, 0.1, 'mS/cm', 8.0),
            'DO': StateVariable('DO', 8.0, 8.0, 0.5, 'mg/L', 5.0),
            'temperature': StateVariable('temperature', 22.0, 22.0, 1.0, '°C', 3.0),
            'nutrient_N': StateVariable('nutrient_N', 150.0, 150.0, 10.0, 'mg/L', 2.0),
            'nutrient_P': StateVariable('nutrient_P', 50.0, 50.0, 5.0, 'mg/L', 2.0),
            'nutrient_K': StateVariable('nutrient_K', 200.0, 200.0, 15.0, 'mg/L', 2.0)
        }
        
        # Disturbances
        self.disturbances = {
            'plant_uptake': Disturbance('plant_uptake', 0.0, 'mg/L/min'),
            'evaporation': Disturbance('evaporation', 0.0, 'L/min'),
            'ambient_temp': Disturbance('ambient_temp', 20.0, '°C'),
            'ambient_humidity': Disturbance('ambient_humidity', 60.0, '%')
        }
        
        # System model parameters
        self.model_params = self._initialize_model_parameters()
        
        # MPC optimization variables
        self.opti = None
        self.control_sequence = None
        self.state_sequence = None
        
        # History for plotting
        self.history = {
            'time': [],
            'states': {name: [] for name in self.state_vars.keys()},
            'controls': {name: [] for name in self.control_vars.keys()},
            'setpoints': {name: [] for name in self.state_vars.keys()}
        }
        
        logger.info(f"Initialized MPC controller with horizon {prediction_horizon}")
    
    def _initialize_model_parameters(self) -> Dict[str, float]:
        """Initialize system model parameters"""
        return {
            # pH dynamics
            'pH_time_constant': 5.0,  # minutes
            'pH_gain': 0.1,  # pH change per mL/min of acid/base
            
            # EC dynamics
            'EC_time_constant': 10.0,  # minutes
            'EC_gain': 0.2,  # EC change per mL/min of nutrients
            
            # DO dynamics
            'DO_time_constant': 2.0,  # minutes
            'DO_gain': 0.1,  # DO change per L/min of air flow
            
            # Temperature dynamics
            'temp_time_constant': 15.0,  # minutes
            'heating_gain': 0.05,  # °C change per % heating
            'cooling_gain': -0.03,  # °C change per % cooling
            
            # Nutrient dynamics
            'nutrient_time_constant': 30.0,  # minutes
            'nutrient_gain': 0.5,  # mg/L change per mL/min dosing
            
            # Cross-couplings
            'ph_ec_coupling': 0.1,  # pH change per EC change
            'temp_do_coupling': 0.2,  # DO change per °C change
        }
    
    def _create_system_model(self) -> Function:
        """Create the system model using CasADi"""
        # State variables
        states = SX.sym('states', len(self.state_vars))
        controls = SX.sym('controls', len(self.control_vars))
        disturbances = SX.sym('disturbances', len(self.disturbances))
        
        # Extract individual states and controls
        pH, EC, DO, temp, N, P, K = states[0], states[1], states[2], states[3], states[4], states[5], states[6]
        pH_dose, EC_dose, air_flow, water_flow, heating, cooling = controls[0], controls[1], controls[2], controls[3], controls[4], controls[5]
        plant_uptake, evaporation, amb_temp, amb_hum = disturbances[0], disturbances[1], disturbances[2], disturbances[3]
        
        # System dynamics (simplified first-order models with cross-couplings)
        p = self.model_params
        
        # pH dynamics
        pH_dot = (1/p['pH_time_constant']) * (
            p['pH_gain'] * pH_dose - 
            pH + 6.0 +  # Setpoint
            p['ph_ec_coupling'] * (EC - 1.2)  # EC coupling
        )
        
        # EC dynamics
        EC_dot = (1/p['EC_time_constant']) * (
            p['EC_gain'] * EC_dose - 
            EC + 1.2  # Setpoint
        )
        
        # DO dynamics
        DO_dot = (1/p['DO_time_constant']) * (
            p['DO_gain'] * air_flow - 
            DO + 8.0 +  # Setpoint
            p['temp_do_coupling'] * (temp - 22.0)  # Temperature coupling
        )
        
        # Temperature dynamics
        temp_dot = (1/p['temp_time_constant']) * (
            p['heating_gain'] * heating + 
            p['cooling_gain'] * cooling - 
            temp + amb_temp  # Ambient temperature
        )
        
        # Nutrient dynamics (simplified)
        N_dot = (1/p['nutrient_time_constant']) * (
            p['nutrient_gain'] * EC_dose - 
            N + 150.0 -  # Setpoint
            plant_uptake  # Plant uptake
        )
        
        P_dot = (1/p['nutrient_time_constant']) * (
            p['nutrient_gain'] * EC_dose - 
            P + 50.0 -  # Setpoint
            plant_uptake  # Plant uptake
        )
        
        K_dot = (1/p['nutrient_time_constant']) * (
            p['nutrient_gain'] * EC_dose - 
            K + 200.0 -  # Setpoint
            plant_uptake  # Plant uptake
        )
        
        # State derivatives
        state_dot = vertcat(pH_dot, EC_dot, DO_dot, temp_dot, N_dot, P_dot, K_dot)
        
        # Create function
        model = Function('system_model', [states, controls, disturbances], [state_dot])
        
        return model
    
    def _create_mpc_optimization(self):
        """Create the MPC optimization problem"""
        # Create optimization problem
        self.opti = ca.Opti()
        
        # Optimization variables
        N = self.prediction_horizon
        M = self.control_horizon
        
        n_states = len(self.state_vars)
        n_controls = len(self.control_vars)
        n_disturbances = len(self.disturbances)
        
        # State sequence over prediction horizon
        self.state_sequence = self.opti.variable(n_states, N+1)
        
        # Control sequence over control horizon
        self.control_sequence = self.opti.variable(n_controls, M)
        
        # Disturbance sequence (assumed constant over horizon)
        disturbance_sequence = self.opti.parameter(n_disturbances, N)
        
        # Initial state
        initial_state = self.opti.parameter(n_states)
        
        # System model
        model = self._create_system_model()
        
        # Initial condition constraint
        self.opti.subject_to(self.state_sequence[:, 0] == initial_state)
        
        # Dynamics constraints
        for k in range(N):
            # Get current control (use last control if beyond control horizon)
            if k < M:
                u_k = self.control_sequence[:, k]
            else:
                u_k = self.control_sequence[:, -1]
            
            # Get current disturbance
            d_k = disturbance_sequence[:, k]
            
            # State at next time step
            x_k = self.state_sequence[:, k]
            x_next = x_k + self.dt * model(x_k, u_k, d_k)
            
            self.opti.subject_to(self.state_sequence[:, k+1] == x_next)
        
        # Control constraints
        control_names = list(self.control_vars.keys())
        for i, name in enumerate(control_names):
            var = self.control_vars[name]
            self.opti.subject_to(self.opti.bounded(
                var.min_value, 
                self.control_sequence[i, :], 
                var.max_value
            ))
        
        # State constraints (soft constraints)
        state_names = list(self.state_vars.keys())
        slack_vars = []
        
        for i, name in enumerate(state_names):
            var = self.state_vars[name]
            
            # Add slack variables for soft constraints
            slack_pos = self.opti.variable(1, N+1)
            slack_neg = self.opti.variable(1, N+1)
            slack_vars.extend([slack_pos, slack_neg])
            
            # Soft constraints
            self.opti.subject_to(
                self.state_sequence[i, :] <= var.setpoint + var.tolerance + slack_pos
            )
            self.opti.subject_to(
                self.state_sequence[i, :] >= var.setpoint - var.tolerance - slack_neg
            )
            
            # Slack variables must be non-negative
            self.opti.subject_to(slack_pos >= 0)
            self.opti.subject_to(slack_neg >= 0)
        
        # Cost function
        cost = 0
        
        # State tracking cost
        for i, name in enumerate(state_names):
            var = self.state_vars[name]
            weight = var.cost_weight
            
            for k in range(N+1):
                error = self.state_sequence[i, k] - var.setpoint
                cost += weight * error**2
        
        # Control effort cost
        for i, name in enumerate(control_names):
            var = self.control_vars[name]
            weight = var.cost_weight
            
            for k in range(M):
                cost += weight * self.control_sequence[i, k]**2
        
        # Slack variable penalty
        for slack in slack_vars:
            cost += 1000 * ca.sum1(slack)  # Large penalty for constraint violations
        
        # Control rate cost (smooth control changes)
        for i in range(n_controls):
            for k in range(1, M):
                rate = self.control_sequence[i, k] - self.control_sequence[i, k-1]
                cost += 0.1 * rate**2
        
        self.opti.minimize(cost)
        
        # Solver options
        opts = {
            'ipopt.print_level': 0,
            'ipopt.sb': 'yes',
            'print_time': 0
        }
        self.opti.solver('ipopt', opts)
        
        logger.info("Created MPC optimization problem")
    
    def solve_mpc(self, current_state: Dict[str, float], 
                  disturbances: Dict[str, float] = None) -> Dict[str, float]:
        """Solve the MPC optimization problem"""
        if self.opti is None:
            self._create_mpc_optimization()
        
        # Set initial state
        state_names = list(self.state_vars.keys())
        initial_state_vec = np.array([current_state[name] for name in state_names])
        self.opti.set_value(self.opti.parameter(initial_state_vec), initial_state_vec)
        
        # Set disturbances
        if disturbances is None:
            disturbances = {name: 0.0 for name in self.disturbances.keys()}
        
        disturbance_names = list(self.disturbances.keys())
        disturbance_vec = np.array([disturbances[name] for name in disturbance_names])
        
        # Repeat disturbance over prediction horizon
        disturbance_matrix = np.tile(disturbance_vec.reshape(-1, 1), (1, self.prediction_horizon))
        self.opti.set_value(self.opti.parameter(disturbance_matrix), disturbance_matrix)
        
        # Solve optimization problem
        try:
            sol = self.opti.solve()
            
            # Extract optimal control sequence
            control_sequence = sol.value(self.control_sequence)
            control_names = list(self.control_vars.keys())
            
            # Return first control action
            optimal_control = {
                name: control_sequence[i, 0] 
                for i, name in enumerate(control_names)
            }
            
            logger.debug(f"MPC solved successfully. Control: {optimal_control}")
            return optimal_control
            
        except Exception as e:
            logger.error(f"MPC optimization failed: {e}")
            # Return zero control as fallback
            return {name: 0.0 for name in self.control_vars.keys()}
    
    def update_setpoints(self, setpoints: Dict[str, float]):
        """Update setpoints for controlled variables"""
        for name, value in setpoints.items():
            if name in self.state_vars:
                self.state_vars[name].setpoint = value
                logger.info(f"Updated setpoint for {name}: {value}")
    
    def simulate_step(self, current_state: Dict[str, float], 
                     control_action: Dict[str, float],
                     disturbances: Dict[str, float] = None) -> Dict[str, float]:
        """Simulate one step of the system"""
        if disturbances is None:
            disturbances = {name: 0.0 for name in self.disturbances.keys()}
        
        # Create system model
        model = self._create_system_model()
        
        # Prepare inputs
        state_vec = np.array([current_state[name] for name in self.state_vars.keys()])
        control_vec = np.array([control_action[name] for name in self.control_vars.keys()])
        disturbance_vec = np.array([disturbances[name] for name in self.disturbances.keys()])
        
        # Simulate one step
        state_dot = model(state_vec, control_vec, disturbance_vec)
        new_state_vec = state_vec + self.dt * np.array(state_dot).flatten()
        
        # Convert back to dictionary
        new_state = {
            name: new_state_vec[i] 
            for i, name in enumerate(self.state_vars.keys())
        }
        
        return new_state
    
    def run_closed_loop_simulation(self, duration: float, 
                                 setpoint_changes: List[Tuple[float, Dict[str, float]]] = None):
        """Run closed-loop simulation"""
        logger.info(f"Running closed-loop simulation for {duration} minutes")
        
        # Initialize state
        current_state = {
            name: var.initial_value 
            for name, var in self.state_vars.items()
        }
        
        # Initialize time
        t = 0.0
        n_steps = int(duration / self.dt)
        
        # Setpoint changes
        if setpoint_changes is None:
            setpoint_changes = []
        
        setpoint_idx = 0
        
        for step in range(n_steps):
            # Update setpoints if needed
            if (setpoint_idx < len(setpoint_changes) and 
                t >= setpoint_changes[setpoint_idx][0]):
                new_setpoints = setpoint_changes[setpoint_idx][1]
                self.update_setpoints(new_setpoints)
                setpoint_idx += 1
            
            # Get disturbances (simplified - could be more realistic)
            disturbances = {
                'plant_uptake': 0.1 * np.sin(2 * np.pi * t / 60),  # Daily cycle
                'evaporation': 0.05 * (1 + 0.1 * np.sin(2 * np.pi * t / 1440)),  # Daily cycle
                'ambient_temp': 20.0 + 5.0 * np.sin(2 * np.pi * t / 1440),  # Daily cycle
                'ambient_humidity': 60.0 + 20.0 * np.sin(2 * np.pi * t / 1440)  # Daily cycle
            }
            
            # Solve MPC
            control_action = self.solve_mpc(current_state, disturbances)
            
            # Apply control and simulate
            current_state = self.simulate_step(current_state, control_action, disturbances)
            
            # Store history
            self.history['time'].append(t)
            for name, value in current_state.items():
                self.history['states'][name].append(value)
                self.history['setpoints'][name].append(self.state_vars[name].setpoint)
            
            for name, value in control_action.items():
                self.history['controls'][name].append(value)
            
            t += self.dt
            
            if step % 100 == 0:
                logger.info(f"Step {step}/{n_steps}, t={t:.1f} min")
    
    def plot_results(self):
        """Plot simulation results"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # Plot states
        state_names = list(self.state_vars.keys())
        for i, name in enumerate(state_names[:6]):  # Plot first 6 states
            ax = axes[i]
            ax.plot(self.history['time'], self.history['states'][name], 'b-', label='Actual')
            ax.plot(self.history['time'], self.history['setpoints'][name], 'r--', label='Setpoint')
            ax.set_xlabel('Time (min)')
            ax.set_ylabel(f'{name} ({self.state_vars[name].unit})')
            ax.set_title(f'{name} Control')
            ax.legend()
            ax.grid(True)
        
        # Plot controls
        control_names = list(self.control_vars.keys())
        fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))
        axes2 = axes2.flatten()
        
        for i, name in enumerate(control_names):
            ax = axes2[i]
            ax.plot(self.history['time'], self.history['controls'][name], 'g-')
            ax.set_xlabel('Time (min)')
            ax.set_ylabel(f'{name} ({self.control_vars[name].unit})')
            ax.set_title(f'{name} Control Action')
            ax.grid(True)
        
        plt.tight_layout()
        plt.show()

def main():
    """Example usage of the MPC controller"""
    # Create MPC controller
    mpc = TruffleMPCController(prediction_horizon=20, control_horizon=10, dt=1.0)
    
    # Define setpoint changes
    setpoint_changes = [
        (0, {'pH': 6.0, 'EC': 1.2, 'temperature': 22.0}),
        (100, {'pH': 6.2, 'EC': 1.5, 'temperature': 24.0}),
        (200, {'pH': 5.8, 'EC': 1.0, 'temperature': 20.0})
    ]
    
    # Run simulation
    mpc.run_closed_loop_simulation(duration=300, setpoint_changes=setpoint_changes)
    
    # Plot results
    mpc.plot_results()

if __name__ == "__main__":
    main()