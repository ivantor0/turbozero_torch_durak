# envs/durak/trainer.py

import torch
import logging
from core.train.trainer import Trainer, TrainerConfig
from core.train.collector import Collector
from core.utils.history import TrainingMetrics, Metric
from envs.durak.collector import DurakCollector
from core.test.tester import Tester


class DurakTrainer(Trainer):
    """
    Trainer specialized for Durak.
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

    def add_collection_metrics(self, episodes):
        """
        Called every time we gather new episodes from self-play in an epoch.
        We'll record the final reward for each finished episode (the last state's reward).
        """
        for ep in episodes:
            final_reward = ep[-1][2].item()  # ep[-1] => (inputs, visits, reward, legal_actions)
            self.history.add_episode_data({'episode_reward': final_reward}, log=self.log_results)

    def add_epoch_metrics(self):
        pass
