import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time


class RoboticArm:
    def __init__(self, num_joints=5):
        self.num_joints = num_joints
        self.joint_angles = np.zeros(num_joints)
        self.success_count = 0
        self.total_episodes = 0

    def forward_kinematics(self):
        """Calculate the position of the end effector."""
        x, y = 0, 0
        for i in range(self.num_joints):
            x += np.cos(np.sum(self.joint_angles[:i]))  # Cumulative angle
            y += np.sin(np.sum(self.joint_angles[:i]))
        return np.array([x, y])

    def reset(self):
        """Reset the arm to its initial state."""
        self.joint_angles = np.zeros(self.num_joints)

    def step(self, box_position):
        """Perform one step in the simulation, adjusting joint angles."""
        # Randomly adjust joint angles for exploration
        self.joint_angles += np.random.uniform(-np.pi / 8, np.pi / 8, self.num_joints)

        # Calculate the end effector position
        end_effector_position = self.forward_kinematics()
        distance_to_box = np.linalg.norm(end_effector_position - box_position)

        # Reward structure
        if distance_to_box < 0.2:  # Close enough to "pick up" the box
            reward = 10
            self.success_count += 1
        else:
            reward = -distance_to_box  # Penalize based on distance

        self.total_episodes += 1
        return end_effector_position, reward

    def get_joint_positions(self):
        """Calculate the positions of all joints."""
        positions = []
        x, y = 0, 0
        for i in range(self.num_joints):
            x += np.cos(np.sum(self.joint_angles[:i]))
            y += np.sin(np.sum(self.joint_angles[:i]))
            positions.append((x, y))
        return positions


def plot_arm_and_box(arm, box_position):
    """Plot the robotic arm and the box."""
    plt.clf()
    joint_positions = arm.get_joint_positions()
    x_vals = [pos[0] for pos in joint_positions]
    y_vals = [pos[1] for pos in joint_positions]

    # Plot arm
    plt.plot(x_vals, y_vals, marker='o', color='b', label='Arm Joints')

    # Plot box
    box = patches.Rectangle((box_position[0] - 0.1, box_position[1] - 0.1), 0.2, 0.2, linewidth=1, edgecolor='r',
                            facecolor='none')
    plt.gca().add_patch(box)

    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid()
    plt.title(f"Success Count: {arm.success_count}, Total Episodes: {arm.total_episodes}")
    plt.pause(0.01)


# Initialize the arm and box
robotic_arm = RoboticArm()
box_position = np.array([2, 0])  # Box position

# Main simulation loop
plt.ion()  # Turn on interactive mode
try:
    while True:
        robotic_arm.step(box_position)
        plot_arm_and_box(robotic_arm, box_position)
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Simulation ended.")
    print(f"Total Successes: {robotic_arm.success_count}, Total Episodes: {robotic_arm.total_episodes}")

finally:
    plt.ioff()
    plt.show()
