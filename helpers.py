import torch
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt


def plot_losses(losses, filename='', plotName='Loss', show=False):
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(np.arange(len(losses)), losses)
    plt.axhline(y=0.0, color="#999999", linestyle='-')
    plt.ylabel(plotName)
    plt.xlabel("Training Steps")
    if show:
        plt.show()

    if (filename):
        plt.savefig(filename)
    
    if not show:
        plt.cla()
        plt.close(fig)

    if not show:
        fig = plt.figure()
        fig.add_subplot(111)
        plt.plot(np.arange(len(losses[-200:])), losses[-200:])
        plt.axhline(y=0.0, color="#999999", linestyle='-')
        plt.ylabel(plotName)
        plt.xlabel("Training Steps")

        if (filename):
            plt.savefig("trimmed-{}".format(filename))
        
        plt.cla()
        plt.close(fig)


def plot_durations(durations, filename='', plotName='Duration', show=False):
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(np.arange(len(durations)), durations)
    plt.ylabel(plotName)
    plt.xlabel('Episode #')
    if show:
        plt.show()

    if (filename):
        plt.savefig(filename)

    if not show:
        plt.cla()
        plt.close(fig)


def plot_scores(scores, ave_scores, filename='', plotName='Score', show=False):

    # staked_scores = np.stack(scores, axis=1)
    ave_stacked_scores = np.mean(scores, axis=1)

    # staked_ave_scores = np.stack(ave_scores, axis=1)
    ave_staked_ave_scores = np.mean(ave_scores, axis=1)

    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(np.arange(len(ave_stacked_scores)), ave_stacked_scores, color="#d9d9d9")
    plt.plot(np.arange(len(ave_staked_ave_scores)), ave_staked_ave_scores, color="#333333")
    plt.axhline(y=np.amax(ave_staked_ave_scores), color="#8a8a8a", linestyle='-')
    plt.ylabel(plotName)
    plt.xlabel('Episode #')
    if show:
        plt.show()

    if (filename):
        plt.savefig(filename)

    if not show:
        plt.cla()
        plt.close(fig)


def save_model(model, filename):

    state = { 'state_dict': model.state_dict() }
    torch.save(state, '{}'.format(filename))

        

def load_model(model, filename, evalMode=True):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])

    if evalMode:
        model.eval()
    else:
        model.train()

    return model


def worker(model, params, train=True, early_stop_threshold=5., early_stop_target=1.):     # reset the environment

    optimizer = torch.optim.Adam(lr=params['lr'], params=model.parameters())
    replay = []

    highest_score = 0
    early_stop_captures = []

    for epoch in range(params['epochs']):
        if train and len(early_stop_captures) >= early_stop_threshold:
            print("stopped early because net has reached target score")
            print(early_stop_captures)
            break

        final_score, epsilon = run_episode(model, replay, params, epoch, train)
        params['scores'].append(final_score)
        stacked_scores = np.stack(params['scores'], axis=1)
        sliced_scores = [agent_scores[-100:] for agent_scores in stacked_scores]
        average_score = np.mean(sliced_scores, axis=1)
        params['ave_scores'].append(average_score)
        
        if train and final_score.any() >= highest_score:
            highest_score = np.amax(final_score)
            save_model(model, 'actor_critic_checkpoint@highest.pt')

        if train and len(replay) >= params['batch_size']:
            loss, actor_loss, critic_loss = update_params(replay, optimizer, params)

            params['losses'].append(loss.item())
            params['actor_losses'].append(actor_loss.item())
            params['critic_losses'].append(critic_loss.item())

            ave_scores = ' '.join(["{:.3f}".format(s) for s in average_score])
            if  epoch % 100 == 0:
                print("Epoch: {}, Epsilon: {:.3f}, Ave Scores: [{}], Max: {:.4f}".format(epoch + 1, epsilon, ave_scores, np.amax(params['scores'])))
        
            replay = []
            early_stop_compare_array = np.full((len(average_score),), early_stop_target, dtype=float)
            if np.all(np.greater(average_score, early_stop_compare_array)):
                early_stop_captures.append(average_score)


def run_episode(model, replay, params, epoch, train):

    env_info = params['env'].reset(train_mode=train)[params['brain_name']]
    state_ = env_info.vector_observations
    num_agents = len(env_info.agents)
    states = torch.from_numpy(state_).float()
    scores = np.zeros(num_agents)               # initialize the score

    values, logprobs, rewards, mean_entropy = [], [], [], torch.tensor(0.)
    done = False

    epsilon = np.clip((params['end_epsilon'] - params['start_epsilon']) / (params['epochs'] - 0) * epoch + params['start_epsilon'], params['end_epsilon'], params['start_epsilon'])
    step_count = 0
    while (done == False):
        step_count += 1
        actor_mean, value = model(states)
        actor_std = torch.tensor(epsilon)

        actor_mean = actor_mean.t()

        action_dist0 = torch.distributions.Normal(actor_mean[0], actor_std)
        action_dist1 = torch.distributions.Normal(actor_mean[1], actor_std)

        mean_entropy = action_dist0.entropy().mean()

        action0 = torch.clamp(action_dist0.sample(), min=-1, max=1)
        action1 = torch.clamp(action_dist1.sample(), min=-1, max=1)
        logprob0 = action_dist0.log_prob(action0)
        logprob1 = action_dist1.log_prob(action1)

        values.append(value.view(-1))
        logprobs.append([logprob0.view(-1), logprob1.view(-1)])

        action_list = [action0.detach().numpy().squeeze(), action1.detach().numpy().squeeze()]
        action_list = np.stack(action_list, axis=1)
        # send all actions to the environment
        env_info = params['env'].step(action_list)[params['brain_name']]
        # get next state (for each agent)
        state_ = env_info.vector_observations
        # get reward (for each agent)
        reward = env_info.rewards
        # see if episode finished
        done = env_info.local_done[0]

        states = torch.from_numpy(state_).float()
        rewards.append(reward)
        scores += np.array(reward)


    # Update replay buffer for each agent


    stacked_logprob0 = torch.stack([a[0] for a in logprobs], dim=1)
    stacked_logprob1 = torch.stack([a[1] for a in logprobs], dim=1)

    stacked_values = torch.stack(values, dim=1)
    stacked_rewards = np.stack(rewards, axis=1)

    for agent_index in range(len(env_info.agents)):
  
        agent_values = stacked_values[agent_index]
        agent_logprobs = [stacked_logprob0[agent_index], stacked_logprob1[agent_index]]
        agent_rewards = stacked_rewards[agent_index]

        actor_losses, critic_losses, losses = get_trjectory_loss(agent_values, agent_logprobs, agent_rewards, mean_entropy, params)
        replay.append((scores[agent_index], actor_losses, critic_losses, losses))

    return scores, epsilon


def update_params(replay, optimizer, params):
    loss0 = torch.tensor(0.)
    loss1 = torch.tensor(0.)
    actor_loss0 = torch.tensor(0.)
    actor_loss1 = torch.tensor(0.)
    critic_loss = torch.tensor(0.)

    for trajectory in replay:
        rewards_sum, actor_losses, critic_loss, losses = trajectory
        loss0 += losses[0]
        loss1 += losses[1]
        actor_loss0 += actor_losses[0]
        actor_loss1 += actor_losses[1]
        critic_loss += critic_loss
    

    loss0 = loss0 / len(replay)
    loss1 = loss1 / len(replay)
    actor_loss0 = actor_loss0 / len(replay)
    actor_loss1 = actor_loss1 / len(replay)
    critic_loss = critic_loss / len(replay)

    loss_mean = (loss0 + loss1) / 2

    optimizer.zero_grad()
    loss_mean.backward()
    optimizer.step()

    actor_loss_sum = actor_loss0 + actor_loss1

    return loss_mean, actor_loss_sum, critic_loss


def get_trjectory_loss(values, logprobs, rewards, mean_entropy, params):

    [logprob0, logprob1] = logprobs

    values = values.flip(dims=(0,))
    rewards = torch.Tensor(rewards).flip(dims=(0,))
    logprob0 = logprob0.flip(dims=(0,))
    logprob1 = logprob1.flip(dims=(0,))

    Returns = []
    total_return = torch.Tensor([0])
    leadup = 0

    for reward_index in range(len(rewards)):
        if rewards[reward_index].item() > 0:
            leadup = params['reward_leadup']
        if leadup == 0:
            total_return = torch.Tensor([0])
        
        total_return = rewards[reward_index] + total_return * params['gamma']
        Returns.append(total_return)
        leadup = leadup - 1 if leadup > 0 else 0

    Returns = torch.stack(Returns).view(-1)
    Returns = F.normalize(Returns, dim=0)

    actor_loss0 = -logprob0 * (Returns - values.detach())
    actor_loss1 = -logprob1 * (Returns - values.detach())
    critic_loss = torch.pow(values - Returns, 2)

    actor_loss0 = actor_loss0.sum()
    actor_loss1 = actor_loss1.sum()
    critic_loss = critic_loss.sum()

    loss0 = actor_loss0 + params['clc']*critic_loss + params['entropy_bonus'] * mean_entropy
    loss1 = actor_loss1 + params['clc']*critic_loss + params['entropy_bonus'] * mean_entropy

    actor_losses = (actor_loss0, actor_loss1)
    losses = (loss0, loss1)

    return actor_losses, critic_loss, losses