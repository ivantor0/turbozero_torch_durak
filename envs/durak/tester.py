# envs/durak/tester.py

from core.test.tester import TwoPlayerTester

class DurakTester(TwoPlayerTester):
    """
    Inherits from TwoPlayerTester to handle 2-player head-to-head evaluations
    with baseline algorithms or multiple models.
    """

    def add_evaluation_metrics(self, episodes):
        # Called after test episodes are collected for each baseline
        if self.history is not None:
            for _ in episodes:
                self.history.add_evaluation_data({}, log=self.log_results)
