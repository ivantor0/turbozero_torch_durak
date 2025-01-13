# envs/durak/trainer.py

import torch
import logging
from core.train.trainer import Trainer, TrainerConfig
from core.train.collector import Collector
from core.utils.history import TrainingMetrics
from envs.durak.collector import DurakCollector
from core.test.tester import Tester

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DurakTrainer(Trainer):
    """
    Trainer specialized for Durak with added logging for debugging.
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

    def collect_episodes(self):
        """
        Override the collect_episodes method to add logging.
        """
        logger.info("Starting episode collection...")
        super().collect_episodes()
        logger.info("Episode collection completed.")

    def assign_rewards(self, terminated_episodes, terminated):
        """
        Override to add logging when rewards are assigned.
        """
        logger.info(f"Assigning rewards for {terminated.sum().item()} terminated episodes.")
        episodes = super().assign_rewards(terminated_episodes, terminated)
        logger.info(f"Rewards assigned for {len(episodes)} episodes.")
        return episodes

    def populate_replay_memory(self, episodes):
        """
        Override to add logging when populating replay memory.
        """
        logger.info(f"Populating replay memory with {len(episodes)} episodes.")
        super().populate_replay_memory(episodes)
        logger.info("Replay memory population completed.")

    def train_loop(self):
        """
        Override the train_loop to add logging at each major step.
        """
        logger.info("Training loop started.")
        while not self.should_stop():
            self.collect_episodes()
            episodes = self.collector.get_terminated_episodes()
            if episodes:
                self.assign_rewards(episodes, self.collector.get_terminated_flags())
                self.populate_replay_memory(episodes)
            self.update_model()
            self.evaluate()
            logger.info("Training iteration completed.")
        logger.info("Training loop finished.")

    def print_env_state(self, env_id: int):
        """
        Print the state of a specific environment for debugging.
        """
        if env_id >= self.collector.envs.parallel_envs:
            logger.warning(f"Environment ID {env_id} is out of bounds.")
            return

        env_state = self.collector.envs.get_state(env_id)
        logger.info(f"State of Environment {env_id}: {env_state}")

    def add_collection_metrics(self, episodes):
        # For each finished episode, we can record final reward
        for ep in episodes:
            final_reward = ep[-1][2].item()
            self.history.add_episode_data({'episode_reward': final_reward}, log=self.log_results)

    def add_epoch_metrics(self):
        pass
