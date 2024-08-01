import collections
import functools
import logging
import os
import pathlib
import re
import sys
import warnings
from ruamel.yaml import YAML

try:
  import rich.traceback
  rich.traceback.install()
except ImportError:
  pass

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger().setLevel('ERROR')
warnings.filterwarnings('ignore', '.*box bound precision lowered.*')

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np
import ruamel.yaml as yaml

import agent
import common


def main(logdir=None, config=None):

  print(sys.argv[0])

  if config is None:
    # configs = yaml.safe_load((
    #     pathlib.Path(sys.argv[0]).parent / 'configs.yaml').read_text())
    yaml = YAML(typ='safe', pure=True)
    configs = yaml.load((pathlib.Path(sys.argv[0]).parent / 'configs.yaml').read_text())
    parsed, remaining = common.Flags(configs=['defaults']).parse(known_only=True)
    config = common.Config(configs['defaults'])
    for name in parsed.configs:
      config = config.update(configs[name])
    config = common.Flags(config).parse(remaining)

  if logdir is None:
    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    config.save(logdir / 'config.yaml')
    print(config, '\n')
  print('Logdir', logdir)

  import tensorflow as tf
  tf.config.experimental_run_functions_eagerly(not config.jit)
  message = 'No GPU found. To actually train on CPU remove this assert.'
  assert tf.config.experimental.list_physical_devices('GPU'), message
  for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
  assert config.precision in (16, 32), config.precision
  if config.precision == 16:
    from tensorflow.keras.mixed_precision import experimental as prec
    prec.set_policy(prec.Policy('mixed_float16'))
  #경험 관리용 인스턴스 생성.
  train_replay = common.Replay(logdir / 'train_episodes', **config.replay, config=config)
  eval_replay = common.Replay(logdir / 'eval_episodes', **dict(
      capacity=config.replay.capacity // 10,
      minlen=config.dataset.length,
      maxlen=config.dataset.length))
  #step 관리용.
  step = common.Counter(train_replay.stats['total_steps'])
  
  outputs = [
      common.TerminalOutput(),
      common.JSONLOutput(logdir),
      common.TensorBoardOutput(logdir),
  ]
  logger = common.Logger(step, outputs, multiplier=config.action_repeat)
  metrics = collections.defaultdict(list)
  # 일정 주기마다(예: 매 10 스텝마다) replay buffer에서 경험을 샘플링
  should_train = common.Every(config.train_every) #train_step을 정해주는 거네. 기본 config은 5로 되어있어.

  should_log = common.Every(config.log_every)
  
  should_video_train = common.Every(config.eval_every) #학습 과정을 recording하는 용
  should_video_eval = common.Every(config.eval_every)
  
  should_expl = common.Until(config.expl_until // config.action_repeat)
  #기본적으로 에이전트에게 중요한 경험을 더 자주 샘플링해가지고 학습할 수 있도록하는 것. 여기서는 curiosity를 베이스로 해서 하겠다.
  should_prioritize = lambda step: (config.prioritize_until < 0) or (step < config.prioritize_until)
  should_rescan_priority = common.Every(float(config.rescan_priority_every)) #한번 더 재스캔? 

  def make_env(mode):
    suite, task = config.task.split('_', 1)
    if suite == 'dmc':
      env = common.DMC(
          task, config.action_repeat, config.render_size, config.dmc_camera)
      env = common.NormalizeAction(env)
    elif suite == 'atari':
      env = common.Atari(
          task, config.action_repeat, config.render_size,
          config.atari_grayscale)
      env = common.OneHotAction(env)
    elif suite == 'crafter':
      assert config.action_repeat == 1
      outdir = logdir / 'crafter' if mode == 'train' else None
      reward = bool(['noreward', 'reward'].index(task)) or mode == 'eval'
      env = common.Crafter(outdir, reward)
      env = common.OneHotAction(env)
    elif suite == 'admc':
      from adaptgym.wrapped import ADMC
      env = ADMC(task, action_repeat=config.action_repeat, size=config.render_size, logdir=logdir, mode=mode)
      # from adaptgym.wrapped import AdaptDMC_nonepisodic
      # env = AdaptDMC_nonepisodic(task, action_repeat=config.action_repeat, size=config.render_size, logdir=logdir, mode=mode)
      env = common.NormalizeAction(env)
    else:
      raise NotImplementedError(suite)
    env = common.TimeLimit(env, config.time_limit)
    return env

  def per_episode(ep, mode):
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    print(f'{mode.title()} episode has {length} steps and return {score:.1f}.')
    logger.scalar(f'{mode}_return', score)
    logger.scalar(f'{mode}_length', length)
    for key, value in ep.items():
      if re.match(config.log_keys_sum, key):
        logger.scalar(f'sum_{mode}_{key}', ep[key].sum())
      if re.match(config.log_keys_mean, key):
        logger.scalar(f'mean_{mode}_{key}', ep[key].mean())
      if re.match(config.log_keys_max, key):
        logger.scalar(f'max_{mode}_{key}', ep[key].max(0).mean())
    should = {'train': should_video_train, 'eval': should_video_eval}[mode]
    if should(step):
      for key in config.log_keys_video:
        logger.video(f'{mode}_policy_{key}', ep[key])
    replay = dict(train=train_replay, eval=eval_replay)[mode]
    logger.add(replay.stats, prefix=mode)
    logger.write()

# 환경만들어서 탐색 축적하는 파트.

  print('Create envs.')
  num_eval_envs = min(config.envs, config.eval_eps)
  if config.envs_parallel == 'none':
    train_envs = [make_env('train') for _ in range(config.envs)]
    eval_envs = [make_env('eval') for _ in range(num_eval_envs)]
  else:
    make_async_env = lambda mode: common.Async(
        functools.partial(make_env, mode), config.envs_parallel)
    train_envs = [make_async_env('train') for _ in range(config.envs)]
    eval_envs = [make_async_env('eval') for _ in range(eval_envs)]
  act_space = train_envs[0].act_space #{'action': Box(0.0, 1.0, (17,), float32)}이면 0~1 사이의 continuous한 17가지 action
  obs_space = train_envs[0].obs_space
  train_driver = common.Driver(train_envs)
  # Driver로 에이전트의 학습을 관리, 에피소드와 스텝을 관리하는데. 이걸로 world model에서 시뮬레이션을 뽑아냄.
  train_driver.on_episode(lambda ep: per_episode(ep, mode='train'))
  train_driver.on_step(lambda tran, worker: step.increment())
  train_driver.on_step(train_replay.add_step)
  train_driver.on_reset(train_replay.add_step)
  eval_driver = common.Driver(eval_envs)
  eval_driver.on_episode(lambda ep: per_episode(ep, mode='eval'))
  eval_driver.on_episode(eval_replay.add_episode)
  
  #이건 그냥 initial policy 포인트를 잡기 위해서 대충 아무런 액션이나 해보고, 환경에서 피드백 얻은다음에 그걸 바탕으로 학습. weight init같은 느낌인듯.
  prefill = max(0, config.prefill - train_replay.stats['total_steps'])
  if prefill:
    print(f'Prefill dataset ({prefill} steps).')
    random_agent = common.RandomAgent(act_space)
    train_driver(random_agent, steps=prefill, episodes=1)
    eval_driver(random_agent, episodes=1)
    train_driver.reset()
    eval_driver.reset()

  print('Create agent.')
  #replay buffer에서 config.datasets에 정의된 {'batch': 16, 'length': 50} 이거대로 가지고 와라.
  train_dataset = iter(train_replay.dataset(**config.dataset))
  #이건 성능평가용.
  report_dataset = iter(train_replay.dataset(**config.dataset))
  eval_dataset = iter(eval_replay.dataset(**config.dataset))
  agnt = agent.Agent(config, obs_space, act_space, step)
  train_agent = common.CarryOverState(agnt.train) #에이전트가 다음 행동을 결정하는데 필요한 정보들을 가지고 오기. 
  
  train_agent(next(train_dataset)) #샘플링된 데이터를 바탕으로 배치를 가져와사 그만큼 학습하겠다.
  
  if (logdir / 'variables.pkl').exists():
    agnt.load(logdir / 'variables.pkl')
  else:
    print('Pretrain agent.')
    for _ in range(config.pretrain):
      train_agent(next(train_dataset))
  #train_policy가 에이전트가 학습 모드와 탐험 모드에서 행동을 선택하는 정책을 정의, 탐색 모드와 학습 모드 사이를 전환하면서 policy를 결정하는거 같은데.. 
  #사실 agent class를 보면 train, explore가 별 차이가 없는데..?
  train_policy = lambda *args: agnt.policy(
      *args, mode='explore' if should_expl(step) else 'train')
  eval_policy = lambda *args: agnt.policy(*args, mode='eval')

  print('Train steps: ', config.train_steps)

  def train_step(tran, worker):
    if should_train(step):
      for _ in range(config.train_steps):
        if should_prioritize(step):
          #이거 하면 우선순위 버퍼들을 배치로 가지고 와서 업데이트하겠다.
          get_batch = lambda: train_replay.get_prioritized_batch(config.dataset.batch,
                                                           config.dataset.length,
                                                           config.policy_priority_weight)
        else:
          get_batch = lambda: (next(train_dataset), None)
        batch, batch_eps = get_batch()
        mets, outs = train_agent(batch)
        if should_prioritize(step):
          #우선순위 버퍼를 가지고 왔으면 이 작업을 해줌. 아 각경험의 중요도라는게 변할 수 있으니깐 train_agent(batch)로 해당 배치를 한번 업데이트 시켜주고나서,
          #그 배치를 가지고 다시 학습을 돌려보겠다.
          train_replay.update_batch_prioritization(outs, batch_eps)
        [metrics[key].append(value) for key, value in mets.items()]
    
    if should_log(step):
      for name, values in metrics.items():
        logger.scalar(name, np.array(values, np.float64).mean())
        metrics[name].clear()
      logger.add(agnt.report(next(report_dataset)), prefix='train')
      logger.write(fps=True)
    
    #이건 optional
    
    if should_rescan_priority(step):
      if should_prioritize(step):
        train_replay.update_priority_entire_buffer(agnt, config.dataset.batch, config.dataset.length)
        train_replay.save_priority(suffix=f'_rescan_{step.value}')

  train_driver.on_step(train_step)

  while step < config.steps:
    logger.write()
    print('Start evaluation.')
    logger.add(agnt.report(next(eval_dataset)), prefix='eval')
    eval_driver(eval_policy, episodes=config.eval_eps)
    print('Start training.')
    train_driver(train_policy, steps=config.eval_every)
    agnt.save(logdir / 'variables.pkl')
    if should_prioritize(step):
      train_replay.save_priority(suffix=f'_{step.value}')
      train_replay.save_priority(suffix=None)  # Also overwrite the default file.
  for env in train_envs + eval_envs:
    try:
      env.close()
    except Exception:
      pass


if __name__ == '__main__':
  main()
