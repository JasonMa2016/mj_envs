""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/mj_envs
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import collections
import gym
import numpy as np
import torch 
from PIL import Image  

from mj_envs.envs import env_base


class ReachBaseV0(env_base.MujocoEnv):

    DEFAULT_OBS_KEYS = [
        'qp', 'qv', 'reach_err'
    ]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "reach": -1.0,
        "bonus": 4.0,
        "penalty": -50,
    }


    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):

        # EzPickle.__init__(**locals()) is capturing the input dictionary of the init method of this class.
        # In order to successfully capture all arguments we need to call gym.utils.EzPickle.__init__(**locals())
        # at the leaf level, when we do inheritance like we do here.
        # kwargs is needed at the top level to account for injection of __class__ keyword.
        # Also see: https://github.com/openai/gym/pull/1497
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)

        # This two step construction is required for pickling to work correctly. All arguments to all __init__
        # calls must be pickle friendly. Things like sim / sim_obsd are NOT pickle friendly. Therefore we
        # first construct the inheritance chain, which is just __init__ calls all the way down, with env_base
        # creating the sim / sim_obsd instances. Next we run through "setup"  which relies on sim / sim_obsd
        # created in __init__ to complete the setup.
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed)

        self._setup(**kwargs)


    def _setup(self,
               robot_site_name,
               target_site_name,
               target_xyz_range,
               frame_skip = 40,
               reward_mode = "dense",
               obs_keys=DEFAULT_OBS_KEYS,
               weighted_reward_keys=DEFAULT_RWD_KEYS_AND_WEIGHTS,
               **kwargs,
        ):

        # ids
        self.grasp_sid = self.sim.model.site_name2id(robot_site_name)
        self.target_sid = self.sim.model.site_name2id(target_site_name)
        self.target_xyz_range = target_xyz_range

        super()._setup(obs_keys=obs_keys,
                       weighted_reward_keys=weighted_reward_keys,
                       reward_mode=reward_mode,
                       frame_skip=frame_skip,
                       **kwargs)

        # create goal embedding
        self.goal_embedding = None
        if self.rgb_encoder in ["gofar", "r3m50"]:
            with torch.no_grad():
                # hard-coded goal-embedding
                
                goal_image = "/mnt/tmp_nfs_clientshare/jasonyma/mj_envs/mj_envs/utils/jason_demo/reach_green_plate20220811-220117_paths_left_0-49.png"
                # goal_image = "/mnt/tmp_nfs_clientshare/jasonyma/mj_envs/mj_envs/utils/jason_demo/reach_green_plate20220811-212748_paths_left_0-49.png"
                # goal_image = "/mnt/tmp_nfs_clientshare/jasonyma/mj_envs/mj_envs/utils/jason_demo/close_drawer20220811-211225_paths_top_0-49.png"
                # goal_image = "/mnt/tmp_nfs_clientshare/jasonyma/mj_envs/mj_envs/utils/jason_demo/close_white_cabinet20220811-200446_paths_top_0-38.png"
                # goal_image = "/mnt/tmp_nfs_clientshare/jasonyma/mj_envs/mj_envs/utils/jason_demo/close_white_cabinet20220811-192349_paths_top_0-49.png"
                # goal_image = "/mnt/tmp_nfs_clientshare/jasonyma/mj_envs/mj_envs/utils/jason_demo/close_white_cabinet20220811-190711_paths_top_0-49.png"
                # goal_image = "/mnt/tmp_nfs_clientshare/jasonyma/mj_envs/mj_envs/utils/jason_demo/get_horizontal_init20220811-145643_paths_top_0-42.png" # open
                img = Image.open(goal_image)
                # img = Image.fromarray(img[0].astype(np.uint8))
                rgb_encoded = 255.0 * self.rgb_transform(img).reshape(-1, 3, 224, 224)
                rgb_encoded.to(self.device_encoder)
                rgb_encoded = self.rgb_encoder(rgb_encoded).cpu().numpy()
                self.goal_embedding = np.squeeze(rgb_encoded)
                # print(self.goal_embedding.shape)

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['t'] = np.array([self.sim.data.time])
        obs_dict['qp'] = sim.data.qpos.copy()
        obs_dict['qv'] = sim.data.qvel.copy()
        obs_dict['reach_err'] = sim.data.site_xpos[self.target_sid]-sim.data.site_xpos[self.grasp_sid]
        return obs_dict


    def get_reward_dict(self, obs_dict):
        reach_dist = np.linalg.norm(obs_dict['reach_err'], axis=-1)
        far_th = 1.0

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('reach',   reach_dist),
            ('bonus',   (reach_dist<.1) + (reach_dist<.05)),
            ('penalty', (reach_dist>far_th)),
            # Must keys
            ('sparse',  -1.0*reach_dist),
            ('solved',  reach_dist<.050),
            ('done',    reach_dist > far_th),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict

    def reset(self, reset_qpos=None, reset_qvel=None):
        # Add manual reset positions here!
        # reset_qpos = np.array([-0.318 ,  0.0186,  1.007 , -2.7084, -1.3307, -0.1815, -1.1835,
        # 0.    ,  0.    ])
        # reset_qpos = np.array([0.3757, 1.0852, 0.0092, -1.5276, -1.6599, 0.081, -1.2109, 0., 0.])
        # reset_qpos = np.array([ 0.3362,  1.1408,  0.0097, -1.4155, -1.7049,  0.0144, -1.239,   0.,      0.    ])
        
        # close white cabinet
        # reset_qpos = np.array([0.3326,  1.3868,  0.1608, -0.8501, -1.2601, -0.2747, -1.5374,  0.,      0.,    ])
        # reset_qvel = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])

        # close white drawer
        # reset_qpos = np.array([ 0.9638,  0.961,  -0.1531, -1.5624, -1.0397, -0.2227, -1.424,   0.,      0.    ])
        # reset_qvel = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])

        # reach green plate (left)
        # reset_qpos = np.array([-0.4396, -0.0982,  0.5941, -2.2897,  0.0242,  0.725,  -1.5521,  0.,      0.    ])
        # reset_qvel = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        
        self.sim.model.site_pos[self.target_sid] = self.np_random.uniform(high=self.target_xyz_range['high'], low=self.target_xyz_range['low'])
        self.sim_obsd.model.site_pos[self.target_sid] = self.sim.model.site_pos[self.target_sid]
        obs = super().reset(reset_qpos, reset_qvel)
        return obs
