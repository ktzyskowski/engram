import numpy as np
import torch


class ReplayBuffer:
    def __init__(
        self,
        observation_shape: tuple,
        action_size: int,
        capacity: int,
        dtype: str = "float32",
        action_dtype: str = "uint8",
    ):
        """Construct an new, empty replay buffer.

        Args:
            observation_shape   (tuple): observation shape.
            action_size         (int):   number of action dimensions.
            capacity            (int):   maximum capacity.
            dtype               (str):   observation dtype.
            action_dtype        (str):   action dtype.
        """
        if capacity <= 0:
            raise ValueError("Replay buffer capacity must be greater than zero.")

        self.capacity = capacity
        self.is_full = False
        self.buffer_index = 0  # pointer to the next index to insert a transition

        # internal buffer arrays to store transitions
        self.observations = np.zeros((self.capacity, *observation_shape), dtype=dtype)
        self.actions = np.zeros((self.capacity, action_size), dtype=action_dtype)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=bool)

    def __len__(self) -> int:
        """Get the current number of transitions stored in the buffer."""
        return self.capacity if self.is_full else self.buffer_index

    def add(self, observation, action, reward, done):
        """Add a transition to the replay buffer

        The next observation is not stored, because this is a sequential replay
        buffer, and it can be easily fetched in a batch by incrementing the
        index by one. (Take care to pay attention to episode boundaries.)

        Old transitions are overwritten first if buffer capacity is exceeded.

        Args:
            observation     (*observation_shape): the observation.
            action          (action_size):        the action taken.
            reward          (float):              the observed reward.
            done            (bool):               whether the episode terminated.
        """

        # convert to numpy arrays if tensors are givens
        if torch.is_tensor(observation):
            observation = observation.numpy(force=True)
        if torch.is_tensor(action):
            action = action.numpy(force=True)

        self.observations[self.buffer_index] = observation
        self.actions[self.buffer_index] = action
        self.rewards[self.buffer_index] = reward
        self.dones[self.buffer_index] = done

        # increment buffer index and wrap around if we exceed capacity, overwriting old transitions
        self.buffer_index = (self.buffer_index + 1) % self.capacity
        if self.buffer_index == 0:
            self.is_full = True

    def sample(self, batch_size: int, sequence_length: int) -> dict:
        """Sample from the replay buffer.

        Args:
            batch_size      (int): number of trajectories to sample.
            sequence_length (int): length of each sampled trajectory.
        Returns:
            observations    (N, T, *observation_shape): sampled observations.
            actions         (N, T, action_size):        sampled actions.
            rewards         (N, T):                     sampled rewards.
            dones           (N, T):                     sampled termination flags.
        """
        if len(self) < sequence_length:
            raise ValueError("Buffer does not contain enough transitions.")

        # determine the pool of valid start indices as (base, count):
        # valid starts are (base + 0, base + 1, ..., base + count - 1) mod capacity.
        if not self.is_full:
            # data lives in [0, buffer_index); sequence must fit before the write head
            base = 0
            count = self.buffer_index - sequence_length + 1
        elif self.dones[(self.buffer_index - 1) % self.capacity]:
            # write head sits on an episode boundary, so crossing it is fine
            base = 0
            count = self.capacity
        else:
            # write head does NOT sit on an episode boundary, we cannot cross it mid-sequence.
            base = self.buffer_index
            count = self.capacity - sequence_length + 1

        if count < batch_size:
            raise ValueError("Not enough valid start indices for requested batch size.")

        # sample with replacement
        offsets = np.random.randint(0, count, size=batch_size)
        start_indices = (base + offsets) % self.capacity
        indices = (start_indices[:, None] + np.arange(sequence_length)) % self.capacity

        return {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "dones": self.dones[indices],
        }

    def sample_torch(
        self,
        batch_size: int,
        sequence_length: int,
        device="cpu",
        dtype=torch.float32,
    ) -> dict:
        """Sample PyTorch tensors from the replay buffer.

        Args:
            batch_size      (int): number of trajectories to sample.
            sequence_length (int): length of each sampled trajectory.
            device          (str): torch device.
            dtype           (str): tensor dtype.
        Returns:
            observations    (N, T, *observation_shape): tensor of sampled observations.
            actions         (N, T, action_size):        tensor of sampled actions.
            rewards         (N, T):                     tensor of sampled rewards.
            dones           (N, T):                     tensor of sampled termination flags.
        """
        batch = self.sample(batch_size, sequence_length)
        # cast dtype on CPU before transfer: MPS has had bugs with
        # bool -> float dtype conversion during host->device copy.
        return {
            k: torch.from_numpy(v).to(dtype=dtype).to(device=device)
            for k, v in batch.items()
        }
