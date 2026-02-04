import numpy as np
import time, datetime
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # no dependencies


class MetricLogger:
    def __init__(self, save_dir):
        """Initialize filepaths and data arrays."""
        # Track episodes and steps
        self.curr_step = 0
        self.episode = 0

        # Directory and filepaths
        self.save_log = save_dir / "log"
        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

        # History metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        # Moving averages, added for every call to record()
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        # Current episode metric
        self._init_episode()

        # Timing
        self.record_time = time.time()

    def _init_logfile(self):
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )

    def _init_episode(self):
        """Reset all metrics before a new episode."""
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def log_step(self, reward, loss, q):
        """Update metrics at end of each step."""
        self.curr_step += 1
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        """Update metrics at end of each episode."""
        self.episode += 1
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        self._init_episode()

    def record(self, epsilon, print_fn=print):
        """Print metrics, write to log file, and save plots."""
        # Calculate moving averages
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print_fn(
            f"Episode {self.episode:10d} - "
            f"Step {self.curr_step:15,d} - "
            f"Epsilon {epsilon:.3f} - "
            f"Mean Reward {mean_ep_reward:8.3f} - "
            f"Mean Length {mean_ep_length:8.3f} - "
            f"Mean Loss {mean_ep_loss:10.3f} - "
            f"Mean Q Value {mean_ep_q:8.3f}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{self.episode:10d}{self.curr_step:15d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_lengths", "ep_avg_losses", "ep_avg_qs", "ep_rewards"]:
            plt.clf()
            plt.plot(
                getattr(self, f"moving_avg_{metric}"), label=f"moving_avg_{metric}"
            )
            plt.legend()
            plt.savefig(getattr(self, f"{metric}_plot"))
