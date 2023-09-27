"""Data collection script."""

import os
import hydra
import numpy as np
import random
import tqdm

from cliport import tasks
from cliport.dataset import RavensDataset
from cliport.environments.environment import Environment
                
import pickle
import json
import cv2


task_list = [
'assembling-kits-seq-seen-colors',
# 'assembling-kits-seq-unseen-colors',
# 'assembling-kits-seq-full',
'packing-shapes',
'packing-boxes-pairs-seen-colors',
# 'packing-boxes-pairs-unseen-colors',
# 'packing-boxes-pairs-full',
'packing-seen-google-objects-seq',
# 'packing-unseen-google-objects-seq',
# 'packing-seen-google-objects-group',
# 'packing-unseen-google-objects-group',
'put-block-in-bowl-seen-colors',
# 'put-block-in-bowl-unseen-colors',
# 'put-block-in-bowl-full',
'stack-block-pyramid-seq-seen-colors',
# 'stack-block-pyramid-seq-unseen-colors',
# 'stack-block-pyramid-seq-full',
'separating-piles-seen-colors',
# 'separating-piles-unseen-colors',
# 'separating-piles-full',
'towers-of-hanoi-seq-seen-colors',
# 'towers-of-hanoi-seq-unseen-colors',
# 'towers-of-hanoi-seq-full',
# 'align-rope',

]

@hydra.main(config_path='./cfg', config_name='data')
def main(cfg):
    
    save_dir = "/mnt/nfs/share/CLIPORT/emu_data_subtask"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    finish_task = len(os.listdir(save_dir))
    print(f"finish task num : {finish_task}")
    
    for ti,task in tqdm.tqdm(enumerate(task_list[finish_task:]),desc='task'):
        print(task)
        cfg['task'] = task  

        dataset_count = 0
        
        # Initialize environment and task.
        env = Environment(
            cfg['assets_root'],
            disp=cfg['disp'],
            shared_memory=cfg['shared_memory'],
            hz=480,
            record_cfg=cfg['record']
        )
        task = tasks.names[cfg['task']]()
        task.mode = cfg['mode']
        record = cfg['record']['save_video']
        save_data = cfg['save_data']

        # Initialize scripted oracle agent and dataset.
        agent = task.oracle(env)
        # data_path = os.path.join(cfg['data_dir'], "{}-{}".format(cfg['task'], task.mode))
        data_path = os.path.join('/mnt/nfs/share/CLIPORT/data_subtask', "{}-{}".format(cfg['task'], task.mode))
        dataset = RavensDataset(data_path, cfg, n_demos=0, augment=False)
        print(f"Saving to: {data_path}")
        print(f"Mode: {task.mode}")

        # Train seeds are even and val/test seeds are odd. Test seeds are offset by 10000
        seed = dataset.max_seed
        if seed < 0:
            if task.mode == 'train':
                seed = -2
            elif task.mode == 'val': # NOTE: beware of increasing val set to >100
                seed = -1
            elif task.mode == 'test':
                seed = -1 + 10000
            else:
                raise Exception("Invalid mode. Valid options: train, val, test")

        # Collect training data from oracle demonstrations.
        epi_pbar = tqdm.tqdm(total=cfg['n'],desc='episode')
        print(dataset.n_episodes)
        while dataset.n_episodes < cfg['n']:
            episode, total_reward, total_rewards, last_lang_goal  = [], 0, [0.], ''
            seed += 2

            # Set seeds.
            np.random.seed(seed)
            random.seed(seed)

            epi_pbar.update(1)
            print('Oracle demo: {}/{} | Seed: {}'.format(dataset.n_episodes + 1, cfg['n'], seed))

            env.set_task(task)
            task_description = task.task_completed_desc
            obs = env.reset()
            info = env.info
            reward = 0

            # Unlikely, but a safety check to prevent leaks.
            if task.mode == 'val' and seed > (-1 + 10000):
                raise Exception("!!! Seeds for val set will overlap with the test set !!!")

            # Start video recording (NOTE: super slow)
            if record:
                env.start_rec(f'{dataset.n_episodes+1:06d}')

            # Rollout expert policy
            for _ in range(task.max_steps):
                act = agent.act(obs, info)
                episode.append((obs, act, reward, info))
                lang_goal = info['lang_goal']
                obs, reward, done, info = env.step(act)
                total_reward += reward
                total_rewards.append(total_reward)
                print(f'Total Reward: {total_reward:.3f} | Done: {done} | Goal: {lang_goal}')
                if done:
                    break
            episode.append((obs, None, reward, info))

            # End video recording
            if record:
                env.end_rec()

            # Only save completed demonstrations.
            if save_data and total_reward > 0.99:
                dataset.add(seed, episode)
                dataset_count = dum_subtask_dataset(episode, ti, dataset_count, finish_task, total_rewards, task_description)


def dum_subtask_dataset(episode, ti, dataset_count, finish_task, total_rewards, task_description):
    root_path = f'/mnt/nfs/share/CLIPORT/emu_data_subtask/task{ti+finish_task}'
    def dump(data, field, name):
        field_path = os.path.join(root_path, field)
        if not os.path.exists(field_path):
            os.makedirs(field_path)
        fname = f'{name}.pkl'  # -{len(episode):06d}
        with open(os.path.join(field_path, fname), 'wb') as f:
            pickle.dump(data, f)
    
    
    # sort episode through lang_goal
    record_index = 0
    record_list = []
    lang_goals = []
    obs = []
    rewards = []
    for i,epi in enumerate(episode):
        lang_goals.append(epi[3]['lang_goal'])
        if i > 0:
            if epi[3]['lang_goal'] != episode[i-1][3]['lang_goal']:
                record_index += 1
        record_list.append(record_index)
        obs.append((epi[0]['color'][0],epi[0]['depth']))
        # rewards.append(epi[2]+sum(rewards))
        rewards.append(total_rewards[i])
    # lang_goals[-1] = lang_goals[-2]
    # record_index -= 1
    # record_list[-1] = record_list[-2]
    
    if record_index == 0:
        # only one lang_goal
        if rewards[-1] > 0.99:
            dump(np.uint8(obs[0][0]), f'epi_{str(dataset_count)}', 'origin_rgb')
            dump(np.float32(obs[0][1]), f'epi_{str(dataset_count)}', 'origin_dep')
            dump(np.uint8(obs[-1][0]), f'epi_{str(dataset_count)}', 'subtask1_rgb')
            dump(np.float32(obs[-1][1]), f'epi_{str(dataset_count)}', 'subtask1_dep')
            with open(f"{root_path}/epi_{dataset_count}/info.json","w") as f:
                json.dump({'task':task_description,
                           'subtask1':lang_goals[-1]}, 
                          f)
            cv2.imwrite(f"{root_path}/epi_{dataset_count}/origin_rgb.png", np.uint8(obs[0][0][:, :, ::- 1]))
            cv2.imwrite(f"{root_path}/epi_{dataset_count}/subtask1_rgb.png", np.uint8(obs[-1][0][:, :, ::- 1]))

    else:
        # multi lang_goal
        info_dict = {'task':task_description}
        for ri in range(record_index):
            if ri == 0:
                dump(np.uint8(obs[0][0]), f'epi_{str(dataset_count)}', 'origin_rgb')
                dump(np.float32(obs[0][1]), f'epi_{str(dataset_count)}', 'origin_dep')
                cv2.imwrite(f"{root_path}/epi_{dataset_count}/origin_rgb.png", np.uint8(obs[0][0][:, :, ::- 1]))
            where = np.where(np.array(record_list)==ri)[0]
            if rewards[where[-1]+1] - rewards[where[0]] > 0:
                dump(np.uint8(obs[where[-1]+1][0]), f'epi_{str(dataset_count)}', f'subtask{ri+1}_rgb')
                dump(np.float32(obs[where[-1]+1][1]), f'epi_{str(dataset_count)}', f'subtask{ri+1}_dep')
                info_dict[f'subtask{ri+1}'] = lang_goals[where[-1]]
                cv2.imwrite(f"{root_path}/epi_{dataset_count}/subtask{ri+1}_rgb.png", np.uint8(obs[where[-1]+1][0][:, :, ::- 1]))
                
        with open(f"{root_path}/epi_{dataset_count}/info.json","w") as f:
            json.dump(info_dict, f)
            
    return dataset_count+1
     

def dump_dataset(episode, ti, dataset_count, finish_task, total_rewards):
    
    root_path = f'/mnt/nfs/share/CLIPORT/emu_data/task{ti+finish_task}'
    def dump(data, field, name):
        field_path = os.path.join(root_path, field)
        if not os.path.exists(field_path):
            os.makedirs(field_path)
        fname = f'{name}.pkl'  # -{len(episode):06d}
        with open(os.path.join(field_path, fname), 'wb') as f:
            pickle.dump(data, f)
    
    
    # sort episode through lang_goal
    record_index = 0
    record_list = []
    lang_goals = []
    obs = []
    rewards = []
    for i,epi in enumerate(episode):
        lang_goals.append(epi[3]['lang_goal'])
        if i > 0:
            if epi[3]['lang_goal'] != episode[i-1][3]['lang_goal']:
                record_index += 1
        record_list.append(record_index)
        obs.append((epi[0]['color'][0],epi[0]['depth']))
        # rewards.append(epi[2]+sum(rewards))
        rewards.append(total_rewards[i])
    lang_goals[-1] = lang_goals[-2]
    record_index -= 1
    # record_list[-1] = record_list[-2]
    
    if record_index == 0:
        # only one lang_goal
        if rewards[-1] > 0.99:
            dump(np.uint8(obs[0][0]), str(dataset_count), 'origin_rgb')
            dump(np.float32(obs[0][1]), str(dataset_count), 'origin_dep')
            dump(np.uint8(obs[-1][0]), str(dataset_count), 'target_rgb')
            dump(np.float32(obs[-1][1]), str(dataset_count), 'target_dep')
            with open(f"{root_path}/{dataset_count}/info.json","w") as f:
                json.dump({'lang_goal':lang_goals[-1]}, f)
            cv2.imwrite(f"{root_path}/{dataset_count}/origin_rgb.png", np.uint8(obs[0][0][:, :, ::- 1]))
            cv2.imwrite(f"{root_path}/{dataset_count}/target_rgb.png", np.uint8(obs[-1][0][:, :, ::- 1]))
            dataset_count += 1
    else:
        # multi lang_goal
        for ri in range(record_index):
            where = np.where(np.array(record_list)==ri)[0]
            if rewards[where[-1]+1] - rewards[where[0]] > 0:
                dump(np.uint8(obs[where[0]][0]), str(dataset_count), 'origin_rgb')
                dump(np.float32(obs[where[0]][1]), str(dataset_count), 'origin_dep')
                dump(np.uint8(obs[where[-1]+1][0]), str(dataset_count), 'target_rgb')
                dump(np.float32(obs[where[-1]+1][1]), str(dataset_count), 'target_dep')
                with open(f"{root_path}/{dataset_count}/info.json","w") as f:
                    json.dump({'lang_goal':lang_goals[where[-1]]}, f)
                cv2.imwrite(f"{root_path}/{dataset_count}/origin_rgb.png", np.uint8(obs[where[0]][0][:, :, ::- 1]))
                cv2.imwrite(f"{root_path}/{dataset_count}/target_rgb.png", np.uint8(obs[where[-1]+1][0][:, :, ::- 1]))
                dataset_count += 1
     
    return dataset_count
     


if __name__ == '__main__':
    main()
