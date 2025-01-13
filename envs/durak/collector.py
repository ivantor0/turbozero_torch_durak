# envs/durak/collector.py

import torch
import logging
from core.algorithms.evaluator import TrainableEvaluator
from core.train.collector import Collector

class DurakCollector(Collector):
    """
    Collector specialized for Durak with added logging for debugging.
    """

    def __init__(self, evaluator: TrainableEvaluator, episode_memory_device: torch.device):
        super().__init__(evaluator, episode_memory_device)

    def assign_rewards(self, terminated_episodes, terminated):
        """
        Assign rewards to terminated episodes with logging.
        """
        episodes = []
        if terminated.any():
            term_indices = terminated.nonzero(as_tuple=False).flatten()
            final_rews = self.evaluator.env.get_rewards()  # shape [N]

            for i, ep in enumerate(terminated_episodes):
                env_idx = term_indices[i]
                r = final_rews[env_idx].item()
                ep_with_r = []
                for (inputs, visits, legal_actions) in ep:
                    ep_with_r.append(
                        (inputs, visits, torch.tensor(r, dtype=torch.float32, device=inputs.device), legal_actions)
                    )
                episodes.append(ep_with_r)

        return episodes

    def postprocess(self, terminated_episodes):
        """
        Postprocess terminated episodes with logging.
        """
        inputs, probs, rewards, legal_actions = zip(*terminated_episodes)
        return list(zip(inputs, probs, rewards, legal_actions))
