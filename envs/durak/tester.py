# envs/durak/tester.py

import torch
from core.test.tester import TwoPlayerTester

class DurakTester(TwoPlayerTester):
    """
    A tester that logs how often the agent wins during evaluation episodes.
    We do this by checking final reward from the perspective of the current player.
    """

    def add_evaluation_metrics(self, episodes):
        if self.history is not None:
            # We'll compute the fraction of wins across these episodes:
            # final_reward > 0 => win
            wins = 0
            for ep in episodes:
                final_reward = ep[-1][2].item()  # (inputs, visits, reward, legal_actions)
                if final_reward > 0:
                    wins += 1
            total = len(episodes)
            win_rate = wins / total if total > 0 else 0.0
            # We store it in e.g. "durak_win_rate"
            self.history.add_evaluation_data({'durak_win_rate': win_rate}, log=self.log_results)
