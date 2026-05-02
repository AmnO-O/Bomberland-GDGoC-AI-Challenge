import matplotlib.pyplot as plt
import numpy as np

def plot_loss(loss_history, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('DQN Training Loss')
    plt.legend()
    plt.grid()
    if save_path:
        plt.savefig(save_path)
    # plt.show()

def plot_rewards(reward_history, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(reward_history, label='Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DQN Training Rewards')
    plt.legend()
    plt.grid()
    if save_path:
        plt.savefig(save_path)
    # plt.show()

def plot_win_rates(win_history, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(win_history, label='Win Rate')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    plt.title('DQN Training Win Rates')
    plt.legend()
    plt.grid()
    if save_path:
        plt.savefig(save_path)
    # plt.show()

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_moving_average(data, window_size=10, save_path=None):
    ma_data = moving_average(data, window_size)
    plt.figure(figsize=(10, 5))
    plt.plot(ma_data, label=f'Moving Average (window={window_size})')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.title('DQN Training Moving Average')
    plt.legend()
    plt.grid()
    if save_path:
        plt.savefig(save_path)
    # plt.show()