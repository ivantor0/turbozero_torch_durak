# envs/durak/collector.py

import torch
from core.algorithms.evaluator import TrainableEvaluator
from core.train.collector import Collector

class DurakCollector(Collector):
    """
    We override assign_rewards to attach the final environment's outcome
    to each time-step in that environment’s episode.
    """

    def __init__(self, evaluator: TrainableEvaluator, episode_memory_device: torch.device):
        super().__init__(evaluator, episode_memory_device)

    def assign_rewards(self, terminated_episodes, terminated):
        """
        For each environment that terminated, we query final reward from the perspective
        of the last player who moved or from self.evaluator.env.cur_players (like Othello).
        We'll store that final reward for each step in the episode.
        """
        episodes = []

        if terminated.any():
            term_indices = terminated.nonzero(as_tuple=False).flatten()
            # We'll get the final reward for each terminated environment from get_rewards().
            # But note Durak is 2-player. Usually we store the perspective-based reward for the
            # current player. For 2-player zero-sum we might store the reward from the perspective
            # of whichever player actually took the last move.
            # We'll do something simpler: we call get_rewards() with the current environment’s
            # cur_players for each env.
            final_rews = self.evaluator.env.get_rewards()  # shape [N]
            for i, episode in enumerate(terminated_episodes):
                env_idx = term_indices[i]
                r = final_rews[env_idx].item()

                # For each step (inputs, visits, legal_actions) in that episode,
                # we store the same final reward.
                episode_with_rewards = []
                for (inputs, visits, legal_actions) in episode:
                    # shape of visits is [39], etc.
                    # Insert (inputs, visits, reward, legal_actions)
                    episode_with_rewards.append(
                        (inputs, visits, torch.tensor(r, dtype=torch.float32, device=inputs.device), legal_actions)
                    )
                episodes.append(episode_with_rewards)

        return episodes

    def postprocess(self, terminated_episodes):
        """
        If we had any data augmentation, we could do it here.
        Durak doesn’t have obvious symmetries, so we’ll just return as is.
        """
        inputs, probs, rewards, legal_actions = zip(*terminated_episodes)
        return list(zip(inputs, probs, rewards, legal_actions))
