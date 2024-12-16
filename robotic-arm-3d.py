import pybullet as p
import pybullet_data
import time
import numpy as np
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces
import cv2
from gymnasium import register

# Register the custom environment
register(
    id='RoboticArmEnv-v0',
    entry_point='__main__:RoboticArmEnv',  # Replace '__main__' with the module name if it's different
)

class RoboticArmEnv(gym.Env):
    def __init__(self, render_mode=False):
        super(RoboticArmEnv, self).__init__()
        self.render_mode = render_mode
        self.physics_client = p.connect(p.GUI if render_mode else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.plane = p.loadURDF("plane.urdf")

        # Load robotic arm with 5 DOF
        self.robot_arm = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)

        # Smaller box dimensions
        self.box = p.loadURDF("cube.urdf", [0.5, 0, 0.1], globalScaling=0.5)

        # Define the action and observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.video_writer = None

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        self.plane = p.loadURDF("plane.urdf")
        self.robot_arm = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
        self.box = p.loadURDF("cube.urdf", [0.5, 0, 0.1], globalScaling=0.5)

        # Randomize box position
        self.box_position = np.random.uniform(0.3, 0.7, size=2)
        p.resetBasePositionAndOrientation(self.box, [self.box_position[0], self.box_position[1], 0.1], [0, 0, 0, 1])

        if self.render_mode:
            self._start_video_recording()

        observation = self._get_observation().astype(np.float32)
        return observation, {}

    def step(self, action):
        # Increase the interpolation factor for faster movements
        speed_factor = 0.3
        for i in range(5):
            current_joint_position = p.getJointState(self.robot_arm, i)[0]
            target_position = current_joint_position + (action[i] - current_joint_position) * speed_factor
            p.setJointMotorControl2(self.robot_arm, i, p.POSITION_CONTROL, targetPosition=target_position)

        p.stepSimulation()
        time.sleep(1. / 480.)  # Increase simulation speed

        observation = self._get_observation().astype(np.float32)
        reward, done = self._calculate_reward()

        terminated = False
        truncated = False

        if done:
            terminated = True

        if self.render_mode:
            self._record_frame()

        return observation, reward, terminated, truncated, {}

    def _get_observation(self):
        joint_positions = [p.getJointState(self.robot_arm, i)[0] for i in range(5)]
        box_position, _ = p.getBasePositionAndOrientation(self.box)

        observation = np.array(joint_positions + list(box_position[:3]))

        if observation.shape[0] < 8:
            observation = np.pad(observation, (0, 8 - observation.shape[0]), 'constant')

        return observation

    def _calculate_reward(self):
        end_effector_pos = p.getLinkState(self.robot_arm, 6)[0]
        box_pos = p.getBasePositionAndOrientation(self.box)[0]
        distance = np.linalg.norm(np.array(end_effector_pos) - np.array(box_pos))

        reward = -distance
        done = distance < 0.1

        if done:
            reward += 10.0

        return reward, done

    def render(self, mode='human'):
        if self.render_mode:
            self._setup_camera()

    def _setup_camera(self):
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=90, cameraPitch=-30, cameraTargetPosition=[0.5, 0, 0.1])

    def close(self):
        if self.render_mode and self.video_writer is not None:
            self.video_writer.release()
        p.disconnect()

    def _start_video_recording(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter('simulation.mp4', fourcc, 20.0, (640, 480))

    def _record_frame(self):
        width, height, rgbImg, _, _ = p.getCameraImage(640, 480)
        frame = np.reshape(rgbImg, (height, width, 4))[:, :, :3]
        self.video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

# Training the model with a higher learning rate and modified hyperparameters
env = gym.make('RoboticArmEnv-v0')
model = PPO('MlpPolicy', env, verbose=1, learning_rate=0.001, n_steps=2048, gamma=0.99, ent_coef=0.01, clip_range=0.2)
model.learn(total_timesteps=20000)

# Testing the trained model
env = gym.make('RoboticArmEnv-v0', render_mode="human")
obs, _ = env.reset()
success_rate = 0
episodes = 10

for episode in range(episodes):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=False)  # Enable more exploration
        obs, rewards, done, truncated, _ = env.step(action)
        if rewards > 0:
            success_rate += 1

print(f"Success Rate: {success_rate}/{episodes}")

env.close()
