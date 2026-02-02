"""
Custom GUI for Robot Navigation Simulation

A graphical interface that embeds the irsim simulation and allows
interactive parameter adjustment.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import threading
import time
import yaml
import os
import tempfile

# Matplotlib embedding in Tkinter
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import torch
from robot_nav.SIM_ENV.sim import SIM
from robot_nav.models.CNNTD3.CNNTD3 import CNNTD3


class SimulationGUI:
    """Main GUI class for the robot navigation simulation."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Robot Navigation Simulation Controller")
        self.root.geometry("1400x900")
        
        # Simulation state
        self.sim = None
        self.model = None
        self.running = False
        self.paused = False
        self.step_count = 0
        self.episode_count = 0
        
        # Parameters (defaults)
        self.params = {
            'max_steps': tk.IntVar(value=300),
            'linear_vel_scale': tk.DoubleVar(value=0.25),  # (action+1)/4 gives 0.25 max
            'world_type': tk.StringVar(value='dynamic'),  # 'static' or 'dynamic'
            'num_obstacles': tk.IntVar(value=4),  # number of obstacles
            'obstacle_speed': tk.DoubleVar(value=0.2),  # max velocity of moving obstacles (dynamic only)
            'model_name': tk.StringVar(value='CNNTD3'),
            'show_lidar': tk.BooleanVar(value=True),
            'show_trajectory': tk.BooleanVar(value=True),
            'simulation_delay': tk.DoubleVar(value=0.05),  # seconds between steps
        }
        
        # Temp world file path
        self.temp_world_file = None
        
        # Statistics
        self.stats = {
            'total_reward': 0.0,
            'collisions': 0,
            'goals': 0,
            'current_distance': 0.0,
        }
        
        self._create_layout()
        self._initialize_model()
        
    def _create_layout(self):
        """Create the GUI layout."""
        # Main container with two columns
        self.main_frame = ttk.Frame(self.root, padding="5")
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Configure grid weights for resizing
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(0, weight=3)  # Simulation view
        self.main_frame.columnconfigure(1, weight=1)  # Control panel
        self.main_frame.rowconfigure(0, weight=1)
        
        # Left side: Simulation visualization
        self._create_simulation_view()
        
        # Right side: Control panel
        self._create_control_panel()
        
    def _create_simulation_view(self):
        """Create the matplotlib simulation view."""
        sim_frame = ttk.LabelFrame(self.main_frame, text="Simulation View", padding="5")
        sim_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        sim_frame.columnconfigure(0, weight=1)
        sim_frame.rowconfigure(0, weight=1)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(8, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)
        self.ax.set_aspect('equal')
        self.ax.set_title("Robot Navigation Environment")
        self.ax.grid(True, alpha=0.3)
        
        # Embed in Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=sim_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
        # Add toolbar
        toolbar_frame = ttk.Frame(sim_frame)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        
    def _create_control_panel(self):
        """Create the control panel with parameters and buttons."""
        control_frame = ttk.LabelFrame(self.main_frame, text="Control Panel", padding="10")
        control_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        row = 0
        
        # === Simulation Controls ===
        ttk.Label(control_frame, text="─── Simulation Controls ───", 
                  font=('Helvetica', 10, 'bold')).grid(row=row, column=0, columnspan=2, pady=10)
        row += 1
        
        # Control buttons
        btn_frame = ttk.Frame(control_frame)
        btn_frame.grid(row=row, column=0, columnspan=2, pady=5)
        
        self.start_btn = ttk.Button(btn_frame, text="▶ Start", command=self._start_simulation, width=10)
        self.start_btn.grid(row=0, column=0, padx=2)
        
        self.pause_btn = ttk.Button(btn_frame, text="⏸ Pause", command=self._toggle_pause, width=10, state='disabled')
        self.pause_btn.grid(row=0, column=1, padx=2)
        
        self.stop_btn = ttk.Button(btn_frame, text="⏹ Stop", command=self._stop_simulation, width=10, state='disabled')
        self.stop_btn.grid(row=0, column=2, padx=2)
        
        self.reset_btn = ttk.Button(btn_frame, text="🔄 Reset", command=self._reset_simulation, width=10)
        self.reset_btn.grid(row=0, column=3, padx=2)
        row += 1
        
        self.step_btn = ttk.Button(btn_frame, text="→ Step", command=self._single_step, width=10)
        self.step_btn.grid(row=1, column=0, columnspan=2, padx=2, pady=5)
        row += 1
        
        ttk.Separator(control_frame, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky='ew', pady=10)
        row += 1
        
        # === Parameters ===
        ttk.Label(control_frame, text="─── Parameters ───", 
                  font=('Helvetica', 10, 'bold')).grid(row=row, column=0, columnspan=2, pady=10)
        row += 1
        
        # Max steps
        ttk.Label(control_frame, text="Max Steps:").grid(row=row, column=0, sticky='w')
        ttk.Entry(control_frame, textvariable=self.params['max_steps'], width=10).grid(row=row, column=1, sticky='w')
        row += 1
        
        # Linear velocity scale
        ttk.Label(control_frame, text="Lin Vel Scale:").grid(row=row, column=0, sticky='w')
        ttk.Scale(control_frame, from_=0.1, to=1.0, variable=self.params['linear_vel_scale'], 
                  orient='horizontal').grid(row=row, column=1, sticky='ew')
        row += 1
        
        # Simulation delay
        ttk.Label(control_frame, text="Sim Delay (s):").grid(row=row, column=0, sticky='w')
        ttk.Scale(control_frame, from_=0.0, to=0.5, variable=self.params['simulation_delay'], 
                  orient='horizontal').grid(row=row, column=1, sticky='ew')
        row += 1
        
        ttk.Separator(control_frame, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky='ew', pady=10)
        row += 1
        
        # === World Settings ===
        ttk.Label(control_frame, text="─── World Settings ───", 
                  font=('Helvetica', 10, 'bold')).grid(row=row, column=0, columnspan=2, pady=10)
        row += 1
        
        # World type selection (Static / Dynamic)
        ttk.Label(control_frame, text="World Type:").grid(row=row, column=0, sticky='w')
        world_type_frame = ttk.Frame(control_frame)
        world_type_frame.grid(row=row, column=1, sticky='w')
        ttk.Radiobutton(world_type_frame, text="Static", variable=self.params['world_type'], 
                        value='static', command=self._update_world_controls).pack(side='left', padx=5)
        ttk.Radiobutton(world_type_frame, text="Dynamic", variable=self.params['world_type'], 
                        value='dynamic', command=self._update_world_controls).pack(side='left', padx=5)
        row += 1
        
        # Number of obstacles
        ttk.Label(control_frame, text="Number of Obstacles:").grid(row=row, column=0, sticky='w')
        ttk.Spinbox(control_frame, from_=1, to=10, textvariable=self.params['num_obstacles'], 
                    width=8).grid(row=row, column=1, sticky='w')
        row += 1
        
        # Obstacle speed (only for dynamic world)
        self.speed_label_text = ttk.Label(control_frame, text="Obstacle Speed:")
        self.speed_label_text.grid(row=row, column=0, sticky='w')
        speed_frame = ttk.Frame(control_frame)
        speed_frame.grid(row=row, column=1, sticky='ew')
        self.speed_scale = ttk.Scale(speed_frame, from_=0.05, to=1.0, variable=self.params['obstacle_speed'], 
                  orient='horizontal')
        self.speed_scale.pack(side='left', fill='x', expand=True)
        self.speed_value_label = ttk.Label(speed_frame, text="0.20", width=5)
        self.speed_value_label.pack(side='right')
        self.params['obstacle_speed'].trace('w', lambda *args: self.speed_value_label.config(
            text=f"{self.params['obstacle_speed'].get():.2f}"))
        self.speed_row = row
        row += 1
        
        ttk.Separator(control_frame, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky='ew', pady=10)
        row += 1
        row += 1
        
        # Model selection
        ttk.Label(control_frame, text="Model:").grid(row=row, column=0, sticky='w')
        model_combo = ttk.Combobox(control_frame, textvariable=self.params['model_name'], width=15)
        model_combo['values'] = ('CNNTD3', 'TD3', 'SAC', 'DDPG', 'PPO')
        model_combo.grid(row=row, column=1, sticky='w')
        row += 1
        
        ttk.Separator(control_frame, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky='ew', pady=10)
        row += 1
        
        # === Display Options ===
        ttk.Label(control_frame, text="─── Display Options ───", 
                  font=('Helvetica', 10, 'bold')).grid(row=row, column=0, columnspan=2, pady=10)
        row += 1
        
        ttk.Checkbutton(control_frame, text="Show LIDAR", 
                        variable=self.params['show_lidar']).grid(row=row, column=0, columnspan=2, sticky='w')
        row += 1
        
        ttk.Checkbutton(control_frame, text="Show Trajectory", 
                        variable=self.params['show_trajectory']).grid(row=row, column=0, columnspan=2, sticky='w')
        row += 1
        
        ttk.Separator(control_frame, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky='ew', pady=10)
        row += 1
        
        # === Statistics ===
        ttk.Label(control_frame, text="─── Statistics ───", 
                  font=('Helvetica', 10, 'bold')).grid(row=row, column=0, columnspan=2, pady=10)
        row += 1
        
        # Stats labels
        self.stats_labels = {}
        for stat_name in ['Episode', 'Step', 'Total Reward', 'Distance to Goal', 'Collisions', 'Goals Reached']:
            ttk.Label(control_frame, text=f"{stat_name}:").grid(row=row, column=0, sticky='w')
            label = ttk.Label(control_frame, text="0")
            label.grid(row=row, column=1, sticky='w')
            self.stats_labels[stat_name] = label
            row += 1
        
        ttk.Separator(control_frame, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky='ew', pady=10)
        row += 1
        
        # Status
        self.status_label = ttk.Label(control_frame, text="Status: Ready", foreground='green')
        self.status_label.grid(row=row, column=0, columnspan=2, pady=10)
        
        # Apply button
        row += 1
        ttk.Button(control_frame, text="Apply & Reload", command=self._apply_and_reload).grid(
            row=row, column=0, columnspan=2, pady=10)
        
        # Initialize world controls visibility
        self._update_world_controls()
        
    def _update_world_controls(self):
        """Show/hide speed controls based on world type."""
        if self.params['world_type'].get() == 'static':
            # Hide speed controls for static world
            self.speed_label_text.config(foreground='gray')
            self.speed_scale.config(state='disabled')
            self.speed_value_label.config(foreground='gray')
        else:
            # Show speed controls for dynamic world
            self.speed_label_text.config(foreground='black')
            self.speed_scale.config(state='normal')
            self.speed_value_label.config(foreground='black')
        
    def _initialize_model(self):
        """Initialize the RL model."""
        try:
            self.status_label.config(text="Status: Loading model...", foreground='orange')
            self.root.update()
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = CNNTD3(
                state_dim=185,
                action_dim=2,
                max_action=1,
                device=device,
                load_model=True,
                model_name=self.params['model_name'].get(),
            )
            self.status_label.config(text="Status: Model loaded", foreground='green')
        except Exception as e:
            self.status_label.config(text=f"Status: Model error - {str(e)[:30]}", foreground='red')
            
    def _create_custom_world_file(self):
        """Create a temporary world file with custom obstacle settings."""
        # Use the robot_world.yaml as base template
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_world_file = os.path.join(script_dir, 'worlds', 'robot_world.yaml')
        
        # Read the base world file
        try:
            with open(base_world_file, 'r') as f:
                world_config = yaml.safe_load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read world file: {e}")
            return None
        
        # Get parameters
        world_type = self.params['world_type'].get()
        num_obstacles = self.params['num_obstacles'].get()
        obstacle_speed = self.params['obstacle_speed'].get()
        
        # Build new obstacle list
        new_obstacles = []
        
        # Static obstacle positions (spread around the world)
        static_positions = [
            [5, 2], [8, 5], [2, 7], [7, 8], [3, 4], 
            [6, 3], [4, 6], [8, 2], [2, 3], [7, 7]
        ]
        
        if world_type == 'static':
            # STATIC WORLD: All obstacles are fixed
            for i in range(min(num_obstacles, len(static_positions))):
                pos = static_positions[i]
                # Vary shapes: circles and rectangles
                if i % 2 == 0:
                    static_obs = {
                        'shape': {'name': 'circle', 'radius': 0.5 + (i % 4) * 0.15},
                        'state': [pos[0], pos[1], 0],
                        'kinematics': {'name': 'static'}
                    }
                else:
                    static_obs = {
                        'shape': {'name': 'rectangle', 'length': 0.6 + (i % 3) * 0.2, 'width': 0.8 + (i % 2) * 0.3},
                        'state': [pos[0], pos[1], i * 0.5],
                        'kinematics': {'name': 'static'}
                    }
                new_obstacles.append(static_obs)
        else:
            # DYNAMIC WORLD: All obstacles move with RVO behavior
            if num_obstacles > 0:
                moving_obs = {
                    'number': num_obstacles,
                    'kinematics': {'name': 'omni'},
                    'distribution': {'name': 'random', 'range_low': [1, 1, -3.14], 'range_high': [9, 9, 3.14]},
                    'behavior': {
                        'name': 'rvo', 
                        'wander': True, 
                        'range_low': [0.5, 0.5, -3.14], 
                        'range_high': [9.5, 9.5, 3.14], 
                        'vxmax': obstacle_speed, 
                        'vymax': obstacle_speed, 
                        'factor': 1.0
                    },
                    'vel_max': [obstacle_speed, obstacle_speed],
                    'vel_min': [-obstacle_speed, -obstacle_speed],
                    'shape': [
                        {'name': 'circle', 'radius': 0.6, 'random_shape': True},
                        {'name': 'polygon', 'random_shape': True, 'avg_radius_range': [0.4, 0.7], 
                         'irregularity_range': [0, 0.3], 'spikeyness_range': [0, 0.3], 'num_vertices_range': [4, 6]}
                    ]
                }
                new_obstacles.append(moving_obs)
        
        # Add boundary walls
        boundary = {
            'shape': {'name': 'linestring', 'vertices': [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]},
            'kinematics': {'name': 'static'},
            'state': [0, 0, 0]
        }
        new_obstacles.append(boundary)
        
        # Update config
        world_config['obstacle'] = new_obstacles
        
        # Create temp file
        temp_dir = tempfile.gettempdir()
        self.temp_world_file = os.path.join(temp_dir, 'custom_robot_world.yaml')
        
        with open(self.temp_world_file, 'w') as f:
            yaml.dump(world_config, f, default_flow_style=False)
        
        return self.temp_world_file
            
    def _initialize_simulation(self):
        """Initialize the simulation environment."""
        try:
            self.status_label.config(text="Status: Loading simulation...", foreground='orange')
            self.root.update()
            
            # Create custom world file with user parameters
            world_file = self._create_custom_world_file()
            if world_file is None:
                world_file = self.params['world_file'].get()
            
            # Create simulation with plotting disabled (we'll handle visualization ourselves)
            self.sim = SIM(
                world_file=world_file,
                disable_plotting=True
            )
            
            self.status_label.config(text="Status: Simulation ready", foreground='green')
            return True
        except Exception as e:
            self.status_label.config(text=f"Status: Sim error - {str(e)[:30]}", foreground='red')
            messagebox.showerror("Error", f"Failed to initialize simulation:\n{str(e)}")
            return False
    
    def _start_simulation(self):
        """Start the simulation loop."""
        if self.sim is None:
            if not self._initialize_simulation():
                return
        
        self.running = True
        self.paused = False
        self.start_btn.config(state='disabled')
        self.pause_btn.config(state='normal')
        self.stop_btn.config(state='normal')
        self.status_label.config(text="Status: Running", foreground='green')
        
        # Start simulation in a separate thread
        self.sim_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.sim_thread.start()
        
    def _toggle_pause(self):
        """Toggle pause state."""
        self.paused = not self.paused
        if self.paused:
            self.pause_btn.config(text="▶ Resume")
            self.status_label.config(text="Status: Paused", foreground='orange')
        else:
            self.pause_btn.config(text="⏸ Pause")
            self.status_label.config(text="Status: Running", foreground='green')
            
    def _stop_simulation(self):
        """Stop the simulation."""
        self.running = False
        self.start_btn.config(state='normal')
        self.pause_btn.config(state='disabled')
        self.stop_btn.config(state='disabled')
        self.status_label.config(text="Status: Stopped", foreground='red')
        
    def _reset_simulation(self):
        """Reset the simulation."""
        self._stop_simulation()
        self.step_count = 0
        self.episode_count = 0
        self.stats = {
            'total_reward': 0.0,
            'collisions': 0,
            'goals': 0,
            'current_distance': 0.0,
        }
        self._update_stats_display()
        
        if self.sim is not None:
            self.sim.reset()
            self._update_visualization()
            
        self.status_label.config(text="Status: Reset", foreground='green')
        
    def _single_step(self):
        """Execute a single simulation step."""
        if self.sim is None:
            if not self._initialize_simulation():
                return
            self.sim.reset()
            
        self._execute_step()
        self._update_visualization()
        
    def _simulation_loop(self):
        """Main simulation loop running in a thread."""
        # Reset environment
        latest_scan, distance, cos, sin, collision, goal, a, reward = self.sim.reset()
        self.episode_count += 1
        
        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue
                
            # Get state and action
            state, terminal = self.model.prepare_state(
                latest_scan, distance, cos, sin, collision, goal, a
            )
            action = self.model.get_action(np.array(state), False)
            
            # Apply velocity scaling
            scale = self.params['linear_vel_scale'].get()
            a_in = [(action[0] + 1) * scale, action[1]]
            
            # Step simulation
            latest_scan, distance, cos, sin, collision, goal, a, reward = self.sim.step(
                lin_velocity=a_in[0], ang_velocity=a_in[1]
            )
            
            # Update stats
            self.step_count += 1
            self.stats['total_reward'] += reward
            self.stats['current_distance'] = distance
            
            if collision:
                self.stats['collisions'] += 1
            if goal:
                self.stats['goals'] += 1
                
            # Update GUI (thread-safe)
            self.root.after(0, self._update_stats_display)
            self.root.after(0, self._update_visualization)
            
            # Check for episode end
            if collision or goal or self.step_count >= self.params['max_steps'].get():
                self.episode_count += 1
                self.step_count = 0
                latest_scan, distance, cos, sin, collision, goal, a, reward = self.sim.reset()
                
            # Delay for visualization
            time.sleep(self.params['simulation_delay'].get())
            
    def _execute_step(self):
        """Execute a single step."""
        if self.model is None or self.sim is None:
            return
            
        # Get current observation
        latest_scan = self.sim.env.get_lidar_scan()["ranges"]
        robot_state = self.sim.env.get_robot_state()
        robot_goal = self.sim.robot_goal
        
        goal_vector = [
            robot_goal[0].item() - robot_state[0].item(),
            robot_goal[1].item() - robot_state[1].item(),
        ]
        distance = np.linalg.norm(goal_vector)
        
        pose_vector = [np.cos(robot_state[2]).item(), np.sin(robot_state[2]).item()]
        cos_val = (pose_vector[0] * goal_vector[0] + pose_vector[1] * goal_vector[1]) / (distance + 1e-8)
        sin_val = (pose_vector[0] * goal_vector[1] - pose_vector[1] * goal_vector[0]) / (distance + 1e-8)
        
        # Get action from model
        state, _ = self.model.prepare_state(
            latest_scan, distance, cos_val, sin_val, False, False, [0, 0]
        )
        action = self.model.get_action(np.array(state), False)
        
        scale = self.params['linear_vel_scale'].get()
        a_in = [(action[0] + 1) * scale, action[1]]
        
        # Step
        latest_scan, distance, cos, sin, collision, goal, a, reward = self.sim.step(
            lin_velocity=a_in[0], ang_velocity=a_in[1]
        )
        
        self.step_count += 1
        self.stats['total_reward'] += reward
        self.stats['current_distance'] = distance
        
        if collision:
            self.stats['collisions'] += 1
        if goal:
            self.stats['goals'] += 1
            
        self._update_stats_display()
        
    def _update_visualization(self):
        """Update the matplotlib visualization."""
        if self.sim is None:
            return
            
        self.ax.clear()
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)
        self.ax.set_aspect('equal')
        self.ax.set_title(f"Episode: {self.episode_count} | Step: {self.step_count}")
        self.ax.grid(True, alpha=0.3)
        
        # Draw obstacles
        for obs in self.sim.env.obstacle_list:
            try:
                # Get vertices - irsim stores as 2xN array (row 0 = x, row 1 = y)
                vertices = obs.vertices
                if vertices is not None and len(vertices) > 0:
                    if isinstance(vertices, np.ndarray):
                        if vertices.shape[0] == 2:  # 2xN format
                            x_coords = vertices[0, :]
                            y_coords = vertices[1, :]
                        else:  # Nx2 format
                            x_coords = vertices[:, 0]
                            y_coords = vertices[:, 1]
                        self.ax.fill(x_coords, y_coords, color='gray', alpha=0.6, edgecolor='black', linewidth=1)
            except Exception:
                # Fallback: try to draw as circle using state and shape
                try:
                    state = obs.state
                    if obs.shape == 'circle' and hasattr(obs, 'radius'):
                        circle = plt.Circle((state[0].item(), state[1].item()), obs.radius, 
                                           color='gray', alpha=0.6, edgecolor='black', linewidth=1)
                        self.ax.add_patch(circle)
                except Exception:
                    pass
        
        # Draw robot
        robot_state = self.sim.env.get_robot_state()
        robot_x, robot_y = robot_state[0].item(), robot_state[1].item()
        robot_theta = robot_state[2].item()
        
        robot_circle = plt.Circle((robot_x, robot_y), 0.2, color='blue', alpha=0.8)
        self.ax.add_patch(robot_circle)
        
        # Robot heading arrow
        arrow_len = 0.4
        self.ax.arrow(robot_x, robot_y, 
                      arrow_len * np.cos(robot_theta), 
                      arrow_len * np.sin(robot_theta),
                      head_width=0.1, head_length=0.05, fc='blue', ec='blue')
        
        # Draw goal
        goal = self.sim.robot_goal
        goal_circle = plt.Circle((goal[0].item(), goal[1].item()), 0.3, 
                                  color='green', alpha=0.5, linestyle='--', fill=False, linewidth=2)
        self.ax.add_patch(goal_circle)
        self.ax.plot(goal[0].item(), goal[1].item(), 'g*', markersize=15)
        
        # Draw LIDAR if enabled
        if self.params['show_lidar'].get():
            scan = self.sim.env.get_lidar_scan()
            ranges = scan["ranges"]
            angles = np.linspace(-np.pi/2, np.pi/2, len(ranges))  # Approximate
            
            for i in range(0, len(ranges), 5):  # Draw every 5th ray
                r = min(ranges[i], 7.0)
                angle = robot_theta + angles[i]
                end_x = robot_x + r * np.cos(angle)
                end_y = robot_y + r * np.sin(angle)
                self.ax.plot([robot_x, end_x], [robot_y, end_y], 'r-', alpha=0.2, linewidth=0.5)
        
        self.canvas.draw()
        
    def _update_stats_display(self):
        """Update the statistics display."""
        self.stats_labels['Episode'].config(text=str(self.episode_count))
        self.stats_labels['Step'].config(text=str(self.step_count))
        self.stats_labels['Total Reward'].config(text=f"{self.stats['total_reward']:.2f}")
        self.stats_labels['Distance to Goal'].config(text=f"{self.stats['current_distance']:.2f}")
        self.stats_labels['Collisions'].config(text=str(self.stats['collisions']))
        self.stats_labels['Goals Reached'].config(text=str(self.stats['goals']))
        
    def _apply_and_reload(self):
        """Apply new parameters and reload simulation."""
        self._stop_simulation()
        self.sim = None
        self._initialize_model()
        if self._initialize_simulation():
            self.sim.reset()
            self._update_visualization()
            messagebox.showinfo("Success", "Settings applied and simulation reloaded!")


def main():
    """Launch the GUI."""
    root = tk.Tk()
    app = SimulationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
