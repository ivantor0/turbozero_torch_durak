# envs/durak/tester.py

from core.test.tester import Tester

class DurakTester(Tester):
    """
    Minimal example: after collecting test episodes, we record their final rewards
    (which might be +1 or -1 from the perspective of the environment's current player).
    """

    def add_evaluation_metrics(self, episodes):
        if self.history is not None:
            for ep in episodes:
                final_reward = ep[-1][2].item()  # the last transition’s reward
                self.history.add_evaluation_data({'durak_score': final_reward}, log=self.log_results)
