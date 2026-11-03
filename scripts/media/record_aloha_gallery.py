"""Record four archived ALOHA paths, checking reset hashes and every state/reward."""
import os
os.environ.setdefault('MUJOCO_GL', 'glfw')
import sys
from pathlib import Path
import hashlib
import json
import numpy as np
import imageio.v2 as imageio
import gymnasium as gym
import gym_aloha

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts/analysis'))
from plot_stage2_results import load_policies, audit_aloha_trajectory
from envs.aloha_env import _mujoco_state_sha256, _observation_image_sha256

out = ROOT / 'docs/media/rollouts'
policies = load_policies('aloha')
env = gym.make('gym_aloha/AlohaTransferCube-v0', obs_type='pixels_agent_pos',
               max_episode_steps=400, disable_env_checker=True)
physics = env.unwrapped._env.physics
physics.model.vis.quality.offsamples = 0
records = []
try:
    for start in (10097, 10086):
        for model, exp in [('groot', 'exp05'), ('qwen3vl', 'exp06')]:
            p = policies[3, exp]
            audit_aloha_trajectory(p)
            i = [int(x['initial_state_id']) for x in p['episodes']].index(start)
            ep = p['episodes'][i]
            source = ROOT / 'asset/durable/aloha' / p['trajectory']['path']
            with np.load(source, allow_pickle=False) as z:
                trace = {key: z[f'ep{i:04d}_{key}'].copy()
                         for key in ('state', 'next_state', 'action', 'reward')}
            obs, _ = env.reset(seed=int(ep['environment_seed']))
            assert _mujoco_state_sha256(physics.data) == ep['initial_state_sha256']
            assert _observation_image_sha256(obs['pixels']['top']) == ep['initial_observation_sha256']
            dest = out / f'aloha_seed3_{model}_start{start}.mp4'
            with imageio.get_writer(dest, fps=25, codec='libx264') as writer:
                for t, action in enumerate(trace['action']):
                    assert np.allclose(obs['agent_pos'], trace['state'][t], atol=1e-6, rtol=0)
                    obs, reward, _, _, _ = env.step(np.asarray(action, np.float32))
                    assert int(reward) == int(trace['reward'][t])
                    assert np.allclose(obs['agent_pos'], trace['next_state'][t], atol=1e-6, rtol=0)
                    if t % 2 == 0 or t == len(trace['action']) - 1:
                        writer.append_data(obs['pixels']['top'])
            records.append(dict(dataset='aloha', training_seed=3, backbone=model,
                initial_state_id=start, environment_seed=int(ep['environment_seed']),
                success=bool(ep['is_success']), steps=len(trace['action']),
                reset_hashes_match=True, all_states_rewards_match=True,
                source=str(source.relative_to(ROOT)).replace('\\', '/'),
                file=dest.name, sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                bytes=dest.stat().st_size))
            print(f'Verified and recorded {dest.name}', flush=True)
finally:
    env.close()
(out / 'aloha_manifest.json').write_text(json.dumps(dict(
    scope='Selected archived-action replays; no new policy inference or test cohort.',
    videos=records), indent=2) + '\n', encoding='utf-8')
