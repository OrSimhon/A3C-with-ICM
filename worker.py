import gym
import numpy as np
import torch as T
from A3C import ActorCritic  # Each thread gets is own actor critic
from ICM import ICM
from memory import Memory
from utils import plot_learning_curve


def worker(name, input_shape, n_actions, global_agent, global_icm, optimizer, icm_optimizer, env_id,
           n_threads, icm=False):
    T_MAX = 20  # max number of steps
    local_agent = ActorCritic(input_shape, n_actions)

    if icm:
        local_icm = ICM(input_shape, n_actions)
        algo = 'ICM'
    else:
        intrinsic_reward = T.zeros(1)
        algo = 'A3C'

    # each agent get his own memory and environment
    memory = Memory()
    env = gym.make(env_id)

    # variables
    total_steps = 0
    max_episodes = 1000
    episode = 0
    scores = []

    # main loop
    while episode < max_episodes:
        obs = env.reset()
        hx = T.zeros(1, 256)  # hidden  state for the Actor Critic
        score, done, episode_steps = 0, False, 0
        while not done:
            # state = T.tensor([obs], dtype=T.float)
            state = T.tensor(obs, dtype=T.float).unsqueeze(0)
            action, value, log_prob, hx = local_agent(state, hx)

            # take action
            obs_, reward, done, info = env.step(action)

            # increment steps and score
            total_steps += 1
            episode_steps += 1
            score += reward

            # reward = 0 # turn off extrinsic rewards
            memory.remember(obs, action, reward, obs_, value, log_prob)
            obs = obs_

            # end the episode and update networks
            if episode_steps % T_MAX == 0 or done:
                states, actions, rewards, new_states, values, log_probs = memory.sample_memory()

                if icm:
                    intrinsic_reward, L_I, L_F = local_icm.calc_loss(states, new_states, actions)

                loss = local_agent.calc_loss(obs, hx, done, rewards, values, log_probs, intrinsic_reward)

                optimizer.zero_grad()
                hx = hx.detach()

                if icm:
                    icm_optimizer.zero_grad()
                    (L_I + L_F).backward()

                loss.backward()

                # Clipping the gradients helps to prevent the gradients from exploding or vanishing
                T.nn.utils.clip_grad_norm_(local_agent.parameters(), 40)

                # takes our gradients from the local agent and upload them to the global agent
                for local_param, global_param in zip(local_agent.parameters(), global_agent.parameters()):
                    global_param._grad = local_param.grad
                optimizer.step()
                local_agent.load_state_dict(global_agent.state_dict())

                if icm:
                    for local_param, global_param in zip(local_icm.parameters(), global_icm.parameters()):
                        global_param._grad = local_param.grad
                    icm_optimizer.step()
                    local_icm.load_state_dict(global_icm.state_dict())

                memory.clear_memory()

        # print debug information for the first agent
        if name == '1':
            scores.append(score)
            avg_score = np.mean(scores[-100:])
            print(f"{algo} episode {episode} thread {name} of {n_threads} steps "
                  f"{total_steps / 1e6:.2f}M score {score:.2f} intrinsic_reward "
                  f"{T.sum(intrinsic_reward):.2f} avg score (100) {avg_score:.1f}")

        episode += 1

    # in the end of all episodes --> plot graphs for the first agent
    if name == '1':
        x = [z for z in range(episode)]
        fname = algo + '_CartPole_no_rewards.png'
        plot_learning_curve(x, scores, fname)
