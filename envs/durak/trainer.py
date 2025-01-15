# envs/durak/trainer.py

import torch
import logging
from core.train.trainer import Trainer, TrainerConfig
from core.test.tester import Tester
from envs.durak.collector import DurakCollector
from core.utils.history import TrainingMetrics, Metric


class DurakTrainer(Trainer):
    """
    Trainer specialized for Durak. We add 'episode_reward' and optionally 'episode_win_rate' metrics,
    so that we can see how often we get a positive reward.
    """

    def __init__(
        self,
        config: TrainerConfig,
        collector: DurakCollector,
        tester: Tester,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        raw_train_config: dict,
        raw_env_config: dict,
        history: TrainingMetrics,
        log_results: bool = True,
        interactive: bool = True,
        run_tag: str = 'durak',
        debug: bool = False
    ):
        super().__init__(
            config=config,
            collector=collector,
            tester=tester,
            model=model,
            optimizer=optimizer,
            device=device,
            raw_train_config=raw_train_config,
            raw_env_config=raw_env_config,
            history=history,
            log_results=log_results,
            interactive=interactive,
            run_tag=run_tag,
            debug=debug
        )

        # Ensure 'episode_reward' is recognized as an episode metric
        # so that add_episode_data({'episode_reward': ...}) doesn't cause KeyError.
        if self.history.cur_epoch == 0:
            if 'episode_reward' not in self.history.episode_metrics:
                self.history.episode_metrics['episode_reward'] = Metric(
                    name='episode_reward',
                    xlabel='Episode',
                    ylabel='Reward',
                    maximize=False,
                    alert_on_best=False,
                    proper_name='Episode Reward'
                )

            if 'episode_win_rate' not in self.history.episode_metrics:
                self.history.episode_metrics['episode_win_rate'] = Metric(
                    name='episode_win_rate',
                    xlabel='Episode',
                    ylabel='Win Rate',
                    maximize=True,
                    alert_on_best=False,
                    proper_name='Episode Win Rate'
                )

    def add_collection_metrics(self, episodes):
        # episodes is a list of finished episodes for all parallel envs
        # Each episode is a list of transitions: (inputs, visits, reward, legal_actions)
        # The final transition has the final reward from the perspective of the "current player."
        for ep in episodes:
            final_reward = ep[-1][2].item()
            self.history.add_episode_data({'episode_reward': final_reward}, log=self.log_results)

            # If we want a "win_rate" style metric:
            # 1 for final_reward>0 => "win", else 0
            win_val = 1.0 if final_reward > 0 else 0.0
            self.history.add_episode_data({'episode_win_rate': win_val}, log=self.log_results)

    def add_epoch_metrics(self):
        # Not strictly required. We can gather e.g. an average from the stored episode data if we want.
        pass
