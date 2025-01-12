# envs/durak/tester.py

from core.test.tester import TwoPlayerTester

class DurakTester(TwoPlayerTester):
    """
    A specialized tester for 2-player Durak.
    Inherits from TwoPlayerTester to handle two-player head-to-head evaluations.
    """

    def add_evaluation_metrics(self, episodes):
        # This gets called for each batch of completed episodes during the test step.
        # You can record stats such as # wins, # losses, etc.
        if self.history is not None:
            for _ in episodes:
                # Example: we won't parse out each final reward here,
                # because TwoPlayerTester automatically handles some stats.
                # But you *could* do e.g. self.history.add_evaluation_data({...})
                self.history.add_evaluation_data({}, log=self.log_results)
