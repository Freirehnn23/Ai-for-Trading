import gymnasium as gym
import numpy as np
from gymnasium import spaces


class WindowedObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper untuk mengubah observation flat:
        (features,)

    menjadi sequence:
        (window_size, features)

    Berguna untuk LSTM, CNN-1D, dan Transformer.

    UPDATE:
    - Menambahkan get_action_mask() agar env Week5 bisa meneruskan valid-action mask
      ke PPOSequenceAgent.
    """

    def __init__(self, env, window_size=32):
        super().__init__(env)

        self.window_size = window_size
        self.feature_dim = env.observation_space.shape[0]

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, self.feature_dim),
            dtype=np.float32,
        )

        self.buffer = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.buffer = [obs.copy() for _ in range(self.window_size)]

        return self._get_obs(), info

    def observation(self, obs):
        self.buffer.append(obs.copy())
        self.buffer = self.buffer[-self.window_size:]

        return self._get_obs()

    def _get_obs(self):
        return np.array(self.buffer, dtype=np.float32)

    def get_action_mask(self):
        """
        Ambil action mask dari base environment jika tersedia.

        Return:
            np.ndarray[bool] atau None.
        """
        if hasattr(self.env, "get_action_mask"):
            return self.env.get_action_mask()
        return None
