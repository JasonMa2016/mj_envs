DESC = '''
Helper script to record/examine a rollout's openloop effects (render/ playback/ recover) on an environment\n
  > Examine options:\n
    - Record:   Record an execution. (Useful for kinesthetic demonstrations on hardware)\n
    - Render:   Render back the execution. (sim.forward)\n
    - Playback: Playback the rollout action sequence in openloop (sim.step(a))\n
    - Recover:  Playback actions recovered from the observations \n
  > Render options\n
    - either onscreen, or offscreen, or just rollout without rendering.\n
  > Save options:\n
    - save resulting paths as pickle or as 2D plots\n
USAGE:\n
    $ python examine_rollout.py --env_name door-v0 \n
    $ python examine_rollout.py --env_name door-v0 --rollout_path my_rollouts.pickle --repeat 10 \n
'''

import gym
# from mj_envs.utils.viz_paths import plot_paths as plotnsave_paths
from mj_envs.utils import tensor_utils
import click
import numpy as np
import pickle
import time
import os
import skvideo.io


@click.command(help=DESC)
# @click.option('-e', '--env_name', type=str, help='environment to load', default="FrankaReachFixed_v2d-v0")
# @click.option('-e', '--env_name', type=str, help='environment to load', default="rpFrankaRobotiqData04-v0")
@click.option('-e', '--env_name', type=str, help='environment to load', default="rpFrankaRobotiqDataPenn-v0")

@click.option('-p', '--rollout_path', type=str, help='absolute path of the rollout', default=None)
@click.option('-m', '--mode', type=click.Choice(['record', 'render', 'playback', 'recover']), help='How to examine rollout', default='record')
@click.option('-k', '--keyboard', type=bool, help='use keyboard input', default=False)
@click.option('-h', '--horizon', type=int, help='Rollout horizon, when mode is record', default=50)
@click.option('-s', '--seed', type=int, help='seed for generating environment instances', default=123)
@click.option('-n', '--num_repeat', type=int, help='number of repeats for the rollouts', default=1)
@click.option('-r', '--render', type=click.Choice(['onscreen', 'offscreen', 'none']), help='visualize onscreen or offscreen', default='none')
@click.option('-c', '--camera_name', type=str, default='top_cam', help=('Camera name for rendering'))
@click.option('-o', '--output_dir', type=str, default='/home/jasonyma/Code/robopen_dataset/raw_paths', help=('Directory to save the outputs'))
@click.option('-on', '--output_name', type=str, default="jason", help=('The name to save the outputs as'))
@click.option('-sp', '--save_paths', type=bool, default=True, help=('Save the rollout paths'))
@click.option('-cp', '--compress_paths', type=bool, default=False, help=('compress paths. Remove obs and env_info/state keys'))
@click.option('-pp', '--plot_paths', type=bool, default=False, help=('2D-plot of individual paths'))
@click.option('-ea', '--env_args', type=str, default="{\'is_hardware\':True}", help=('env args. E.g. --env_args "{\'is_hardware\':True}"'))
@click.option('-ns', '--noise_scale', type=float, default=0.0, help=('Noise amplitude in randians}"'))


def main(env_name, rollout_path, mode, keyboard, horizon, seed, num_repeat, render, camera_name, output_dir, output_name, save_paths, compress_paths, plot_paths, env_args, noise_scale):

    # seed and load environments
    np.random.seed(seed)
    env = gym.make(env_name) if env_args==None else gym.make(env_name, **(eval(env_args)))
    env.seed(seed)

    # load paths
    if mode == 'record':
        if keyboard:
            from vtils.keyboard import key_input as keyboard 
            ky = keyboard.Key()
        assert horizon>0, "Rollout horizon must be specified when recording rollout"
        assert output_name is not None, "Specify the name of the recording"
        if save_paths is False:
            print("Warning: Recording is not being saved. Enable save_paths=True to log the recorded path")
        paths = [None,]*num_repeat # empty paths for recordings
        if rollout_path is not None:
            rollout_path = pickle.load(open(rollout_path, 'rb'))
    else:
        assert rollout_path is not None, "Rollout path is required for mode:{} ".format(mode)
        paths = pickle.load(open(rollout_path, 'rb'))
        if output_dir == './': # overide the default
            output_dir = os.path.dirname(rollout_path)
        if output_name is None:
            rollout_name = os.path.split(rollout_path)[-1]
            output_name = os.path.splitext(rollout_name)[0]

    # resolve rendering
    env.mujoco_render_frames = False

    # playback paths
    demo_paths = []
    recover_paths = []

    mode = "record"
    count = 0 

    # for i in range(num_repeat*2):
        # mode = "record" if i % 2 == 0 else "recover"
    while count < num_repeat:
        print(mode, count)
        # initialize buffers
        obs = []
        act = []
        rewards = []
        env_infos = []
        states = []

        # reset all initial states to the initial state of the first demo!
        if len(demo_paths) > 0:
            print("resetting to specified initial state!")
            env.env.robot.robot_config['franka']['robot'].gain_scale = 0.5
            env.env.robot.robot_config['franka']['robot'].reconnect()
            env.reset(reset_qpos=demo_paths[0]['env_infos']['state']['qpos'][0], reset_qvel=demo_paths[0]['env_infos']['state']['qvel'][0])
            if mode == "record":
                env.env.robot.robot_config['franka']['robot'].gain_scale = 0.0
                env.env.robot.robot_config['franka']['robot'].reconnect()
        else:
            if rollout_path is not None:
                print("resetting to specified initial state!")
                assert count == 0 and mode == "record"
                env.env.robot.robot_config['franka']['robot'].gain_scale = 0.5
                env.env.robot.robot_config['franka']['robot'].reconnect()
                env.reset(reset_qpos=rollout_path[0]['env_infos']['state']['qpos'][0], reset_qvel=rollout_path[0]['env_infos']['state']['qvel'][0])
                env.env.robot.robot_config['franka']['robot'].gain_scale = 0.0
                env.env.robot.robot_config['franka']['robot'].reconnect()
            else:
                env.reset()

        # Rollout
        o = env.get_obs()
        sen_last = -1
        path_horizon = horizon 
        for i_step in range(path_horizon):
            print(i_step)
            # Record Execution. Useful for kinesthetic demonstrations on hardware
            # Jason: remember to set gain to 0 in .config file of the environment!
            if mode=='record':
                a = env.action_space.sample() # dummy random sample
                if keyboard:
                    sen = ky.get_sensor()
                    if sen is not None:
                        print(sen, end=", ", flush=True)
                        if sen == 'up' or sen == "b":
                            sen_last = -1
                        elif sen=='down' or sen == "a":
                            sen_last = 1
                a[-1] = sen_last
                onext, r, d, info = env.step(a) # t ==> t+1

            # Recover actions from states
            elif mode=='recover':
                # assumes position controls
                assert len(demo_paths) > 0
                a = demo_paths[-1]['env_infos']['obs_dict']['qp'][i_step]
                if a[-1] > 0.01:
                    a[-1] = 1
                if noise_scale:
                    a = a +  env.env.np_random.uniform(high=noise_scale, low=-noise_scale, size=len(a)).astype(a.dtype)
                if env.normalize_act:
                    a = env.robot.normalize_actions(controls=a)
                onext, r, d, info = env.step(a) # t ==> t+1

            # populate rollout paths
            act.append(a)
            rewards.append(r)
            if compress_paths:
                obs.append([]); o = onext # don't save obs
                del info['state']  # don't save state
            else:
                obs.append(o); o = onext
            env_infos.append(info)

        # Create path
        if mode == "record":
            input("Press any key")
            x = input("Add this demo?")
            if x == "y" or x == "b":
                print("Added demo path", count)
                demo_path = dict(observations=np.array(obs),
                    actions=np.array(act),
                    rewards=np.array(rewards),
                    env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
                    states=states)
                demo_paths.append(demo_path)
                mode = "recover"
            else:
                continue # recollecting demo 
        else:
            input("Press any key")
            x = input("Add this replay?")
            if x == "y" or x == "b":
                print("Added recover path", count)
                recover_path = dict(observations=np.array(obs),
                    actions=np.array(act),
                    rewards=np.array(rewards),
                    env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
                    states=states)
                recover_paths.append(recover_path)
                mode = "record"
                count += 1
            else: 
                continue # replay the same demo

    # reset one last time
    env.env.robot.robot_config['franka']['robot'].gain_scale = 0.5
    env.env.robot.robot_config['franka']['robot'].reconnect()
    env.reset(reset_qpos=demo_paths[0]['env_infos']['state']['qpos'][0], reset_qvel=demo_paths[0]['env_infos']['state']['qvel'][0])
       
    # Save paths
    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    print("Saving Paths!")
    if save_paths:
        # file_name = os.path.join(output_dir, output_name + '{}_paths.pickle'.format(time_stamp))
        # pickle.dump(demo_paths, open(file_name, 'wb'))
        # print("Saved: "+file_name)

        file_name = os.path.join(output_dir, output_name + 'recover_{}_paths.pickle'.format(time_stamp))
        pickle.dump(recover_paths, open(file_name, 'wb'))
        print("Saved: "+file_name)
        
if __name__ == '__main__':
    main()
