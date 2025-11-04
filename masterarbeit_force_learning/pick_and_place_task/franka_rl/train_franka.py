import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
import os

# ==============================================================================
# CUSTOM FRANKA ENVIRONMENT
# ==============================================================================

class FrankaPushEnv(gym.Env):
    """
    A custom Gymnasium environment for the Franka Panda pushing task.
    """
    
    def __init__(self, model_path="main_push_task.xml", render_mode=None):
        super().__init__()
        
        self.model_path = model_path
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        
        # Action space (7 arm joints) is still the same
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )
        
        # === OBSERVATION SPACE CHANGE ===
        # - 7 joint positions (qpos)
        # - 7 joint velocities (qvel)
        # - 3 bottle position (xpos)
        # - 9 bottle orientation (xmat)
        # - 6 bottle velocity (cvel)
        # - 7 joint torques (actuatorfrc)
        # TOTAL: 7 + 7 + 3 + 9 + 6 + 7 = 39
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(39,), dtype=np.float32
        )
        
        self.render_mode = render_mode
        self.viewer = None
        
        # Get IDs for quick access
        self.bottle_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'bottle')
        self.bottle_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'bottle_geom')
        
        # === SENSOR ID CHANGE ===
        # Get IDs for the 7 new joint torque sensors
        self.torque_sensor_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, f'j{i}_torque')
            for i in range(1, 8)
        ]

        # Define a goal position for the bottle
        self.goal_pos = np.array([0.7, 0.0, 0.88]) # Goal on the task table


    def _get_obs(self):
        # Robot state (first 7 joints)
        qpos = self.data.qpos[:7]
        qvel = self.data.qvel[:7]
        
        # Bottle state
        bottle_pos = self.data.body(self.bottle_body_id).xpos
        bottle_orient_mat = self.data.body(self.bottle_body_id).xmat.flatten() # 9 values
        bottle_vel = self.data.body(self.bottle_body_id).cvel # 6 values
        
        # === SENSOR READING CHANGE ===
        # Read the 7 joint torque sensors
        joint_torques = np.array([
            self.data.sensor(sensor_id).data[0] for sensor_id in self.torque_sensor_ids
        ])
        
        observation = np.concatenate([
            qpos, qvel,
            bottle_pos, bottle_orient_mat, bottle_vel,
            joint_torques
        ])
        
        return observation.astype(np.float32)

    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # === DOMAIN RANDOMIZATION ===
        
        # 1. Randomize bottle mass
        new_mass = self.np_random.uniform(low=0.2, high=1.0) # 200g to 1kg
        self.model.body(self.bottle_body_id).mass[0] = new_mass
        
        # 2. Randomize bottle friction
        new_friction = self.np_random.uniform(low=0.2, high=0.8)
        self.model.geom(self.bottle_geom_id).friction[0] = new_friction
        
        # 3. Randomize bottle start position
        start_x = self.np_random.uniform(low=0.4, high=0.6)
        start_y = self.np_random.uniform(low=-0.2, high=0.2)
        
        # Get the address of the bottle's free joint
        jnt_adr = self.model.body(self.bottle_body_id).jntadr[0]
        # Set the (x, y, z) position in qpos
        self.data.qpos[jnt_adr:jnt_adr+3] = [start_x, start_y, 0.85]
        # Set orientation to default (1, 0, 0, 0) quaternion
        self.data.qpos[jnt_adr+3:jnt_adr+7] = [1, 0, 0, 0]
        
        # Forward kinematics to update the model state
        mujoco.mj_forward(self.model, self.data)
        
        return self._get_obs(), {} # Return obs and info dict

    def step(self, action):
        # Apply the action
        self.data.ctrl[:7] = action
        
        # Step the simulation
        for _ in range(5): 
            mujoco.mj_step(self.model, self.data)
        
        # Get new observation
        obs = self._get_obs()
        
        # --- Calculate Reward ---
        bottle_pos = self.data.body(self.bottle_body_id).xpos
        dist_to_goal = np.linalg.norm(bottle_pos - self.goal_pos)
        
        # Reward 1: Dense reward for getting closer to the goal
        reward = -dist_to_goal
        
        # === REWARD PENALTY CHANGE ===
        # We can remove the old force penalty. The agent will still learn
        # to be gentle, because pushing too hard will make the bottle
        # fly off the table, which is bad for the 'dist_to_goal' reward.
        
        # (Optional) You could add a new penalty based on joint torques:
        # joint_torques = obs[-7:] # Get torques from the observation
        # torque_penalty = -0.001 * np.linalg.norm(joint_torques)
        # reward += torque_penalty
        
        # Check if done
        terminated = dist_to_goal < 0.05 # Success!
        truncated = False
        
        if self.render_mode == "human":
            self.render()
            
        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


# ==============================================================================
# TRAINING PART
# ==============================================================================

if __name__ == "__main__":
    
    # --- This is just to install stable-baselines3 if you haven't ---
    try:
        import stable_baselines3
    except ImportError:
        print("stable_baselines3 not found. Installing...")
        os.system("pip install stable-baselines3")
        import stable_baselines3
    # -----------------------------------------------------------------
    
    print("--- 1. Creating Custom Franka Environment ---")
    
    # We create our custom env, and pass render_mode="human"
    # to watch it train.
    env = FrankaPushEnv(render_mode="human")
    
    # Optional: Check if the environment is valid
    from stable_baselines3.common.env_checker import check_env
    try:
        check_env(env)
        print("--- Environment check passed! ---")
    except Exception as e:
        print(f"--- Environment check failed: {e} ---")
    
    print("--- 2. Creating PPO Agent ---")
    model = PPO("MlpPolicy", env, verbose=1, device='cpu', tensorboard_log="./franka_push_tensorboard/")
    
    print("--- 3. Training Agent ---")
    # This will be slow because of the rendering.
    # For real training, set render_mode=None
    model.learn(total_timesteps=100000) # Train for 100k steps to start
    
    print("--- 4. Saving Trained Model ---")
    model.save("ppo_franka_push")
    
    env.close()

    # --- Load and test the trained model ---
    print("\n--- 5. Loading and Testing Trained Model ---")
    model = PPO.load("ppo_franka_push")
    
    test_env = FrankaPushEnv(render_mode="human")
    obs, info = test_env.reset()
    
    for _ in range(5000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        
        if terminated or truncated:
            print("Episode finished. Resetting.")
            obs, info = test_env.reset()
            
    test_env.close()