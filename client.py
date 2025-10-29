from typing import Dict, List
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Optional, List, Tuple

from openpi_client import image_tools, websocket_client_policy


def delta_quat_to_euler(delta_quat):
    """
    Convert delta rotation from quaternion [qw, qx, qy, qz] to Euler angles [roll, pitch, yaw]
    Args:
        delta_quat: (4,) array of quaternions [qw, qx, qy, qz]
    Returns:
        (3,) array of Euler angles in radians [roll, pitch, yaw]
    """
    # Convert to [x, y, z, w] for scipy
    quat_xyzw = delta_quat[:, [1, 2, 3, 0]]
    rot = R.from_quat(quat_xyzw)
    return rot.as_euler("xyz", degrees=False)


def delta_euler_to_quat(delta_euler):
    """
    Convert delta rotation from Euler angles [roll, pitch, yaw] to quaternion [qw, qx, qy, qz]
    Args:
        delta_euler: (N, 3) array of Euler angles in radians
    Returns:
        (N, 4) array of quaternions [qw, qx, qy, qz]
    """
    rot = R.from_euler("xyz", delta_euler, degrees=False)
    quat_xyzw = rot.as_quat()  # [x, y, z, w]
    return quat_xyzw[:, [3, 0, 1, 2]]  # Convert to [w, x, y, z]

class OpenPIAgent:

    def __init__(
            self,
            host: str = "localhost",
            port: int = 8003,
            image_size: int = 224,
            use_wrist_image: bool = False,
            use_state: bool = False,
    ):
        self.client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        self.image_size = image_size
        self.use_wrist_image = use_wrist_image
        self.use_state = use_state
        self.current_task = None

    def reset_policy(self, task_description: str):
        """Reset the agent with a new task description"""
        self.current_task = task_description

    def get_action(
            self,
            image_list: List[np.ndarray],
            task_description: Optional[str] = None,
            wrist_image: np.ndarray = None,
            state: np.ndarray = None,
    ):
        if len(image_list) == 0:
            raise ValueError("Empty image list")

        img = image_list[-1]  # Use most recent frame

        if not isinstance(img, np.ndarray) or img.dtype != np.uint8:
            raise ValueError("Image must be np.uint8")
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("Image must be (H, W, 3) format")
        
        obs = {
            "observation/image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, self.image_size, self.image_size)),
            "prompt": task_description,
        }

        if self.use_wrist_image:
            assert wrist_image is not None, "wrist_image must be provided if use_wrist_image=True"
            obs["observation/wrist_image"] = self._resize_uint8(wrist_image)

        if self.use_state:
            assert state is not None, "state must be provided if use_state=True"
            obs["observation/state"] = state

        response = self.client.infer(obs)
        action_chunk = response["actions"]  # shape (H, 7)

        # delta_pos = action_chunk[:, :3]
        # delta_euler = action_chunk[:, 3:6]
        # gripper_action = action_chunk[:, 6:7]
        # delta_quat = delta_euler_to_quat(delta_euler)
        # action_chunk = np.concatenate([delta_pos, delta_quat, gripper_action], axis=-1)  # shape (H, 8)

        return action_chunk, None


if __name__ == "__main__":
    import json
    import os
    from collections import deque
    from PIL import Image
    import matplotlib.pyplot as plt
    import re
    import yaml
    from tqdm import tqdm
    
    json_path = "/home/jasonl6/sandeep/Work/lq-exp/exp4/real_data_v4/processed_v5/processed_raw.jsonl"
    image_root = "/home/jasonl6/sandeep/Work/lq-exp/exp4/real_data_v4/processed_v5/images"
    rollout_config_path = "/home/jasonl6/sandeep/Work/lq-exp/exp4/real_data_v4/processed_v5/norm_stats.yaml"
    traj_id = "kitchen_bimanual_v1/traj_0080/"
    traj_data = []
    with open(json_path, "r") as f:
        for line in f:
            data = json.loads(line)
            if traj_id in data["id"]:
                traj_data.append(data)
    traj_data.sort(key=lambda x: x["id"])
    instruction = traj_data[0]["instruction"]
    task_description = re.findall(r'`(.*?)`', instruction)[0]

    with open(rollout_config_path, "r") as f:
        rollout_config = yaml.safe_load(f)

    with open(rollout_config_path, "r") as f:
            rollout_config = yaml.safe_load(f)

    action_norm_stats = {}
    action_norm_stats["mean"] = rollout_config["mean"]
    action_norm_stats["std"] = rollout_config["std"]

    # Initialize OpenVLA agent
    agent = OpenPIAgent(
        host="localhost",
        port=8005,
    )
    agent.reset_policy(task_description)

    # Prepare rollout
    curr_action_sequence = None
    pred_horizon = 14
    curr_action_index = 0
    state_buffer = deque(maxlen=3)

    pred_action_log = []
    real_action_log = []

    for step, data in tqdm(enumerate(traj_data), total=len(traj_data)):
        img_path = os.path.join(image_root, data["image"])
        img = np.array(Image.open(img_path))
        state_buffer.append(img)

        if curr_action_sequence is None or curr_action_index >= pred_horizon:
            curr_action_sequence, _ = agent.get_action([state_buffer[-1]], task_description)
            curr_action_index = 0
            # import ipdb; ipdb.set_trace()

        pred_action = curr_action_sequence[curr_action_index]
        real_action = np.array([float(x) for x in data["raw_action"]])
        real_action = real_action * action_norm_stats["std"] + action_norm_stats["mean"]
        # delta_pos = real_action[:3]
        # delta_quat = real_action[3:7]
        # gripper = real_action[7:]
        # delta_euler = delta_quat_to_euler(delta_quat[None, :])[0]
        # real_action = np.concatenate([delta_pos, delta_euler, gripper], dtype=np.float32)
        # bimanual
        delta_pos1 = real_action[:3]
        delta_quat1 = real_action[3:7]
        gripper1 = real_action[7:8]
        delta_pos2 = real_action[8:11]
        delta_quat2 = real_action[11:15]
        gripper2 = real_action[15:16]
        delta_euler1 = delta_quat_to_euler(delta_quat1[None, :])[0]
        delta_euler2 = delta_quat_to_euler(delta_quat2[None, :])[0]
        real_action = np.concatenate([delta_pos1, delta_euler1, gripper1, delta_pos2, delta_euler2, gripper2], dtype=np.float32)

        pred_action_log.append(pred_action)
        real_action_log.append(real_action)

        curr_action_index += 1

    pred_action_log = np.array(pred_action_log)
    real_action_log = np.array(real_action_log)
    assert pred_action_log.shape == real_action_log.shape, "Predicted and real action logs must have the same shape"
    num_actions = pred_action_log.shape[1]

    # Plot predicted vs real
    fig, axes = plt.subplots(nrows=1, ncols=num_actions, figsize=(5 * num_actions, 5))

    for adim in range(num_actions):
        ax = axes[adim] if num_actions > 1 else axes
        ax.plot(pred_action_log[:, adim], label=f"pred_{adim}", color='r')
        ax.plot(real_action_log[:, adim], label=f"real_{adim}", color='b')
        ax.legend()
        ax.set_title(f"Action Dimension {adim}")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Action Value")

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/openvla_pred_vs_real_actions.png")

    # save action logs
    # os.makedirs("logs", exist_ok=True)
    # np.save("logs/pi0_pred_actions.npy", pred_action_log)
    # np.save("logs/pi0_real_actions.npy", real_action_log)