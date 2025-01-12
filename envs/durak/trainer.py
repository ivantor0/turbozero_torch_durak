# envs/durak/trainer.py

import torch
from core.train.trainer import Trainer, TrainerConfig
from core.utils.history import TrainingMetrics
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

    def add_collection_metrics(self, episodes):
        # For each finished episode, we can record final reward
        for ep in episodes:
            final_reward = ep[-1][2].item()
            self.history.add_episode_data({'episode_reward': final_reward}, log=self.log_results)

    def add_epoch_metrics(self):
        pass
