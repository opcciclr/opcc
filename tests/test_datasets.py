import os
import opcc
import pytest

DATASET_ENV_PAIRS = []
DATASET_ENV_PAIRS += [(env_name, dataset_name)
                     for dataset_name in ['1k', '10k', '100k', '1m']
                     for env_name in ['d4rl:maze2d-open-dense-v0',
                                      'd4rl:maze2d-umaze-dense-v1',
                                      'd4rl:maze2d-medium-dense-v1',
                                      'd4rl:maze2d-large-dense-v1']]
DATASET_ENV_PAIRS += [(env_name, dataset_name)
                      for dataset_name in ['random',
                                           'random-1k',
                                           'random-10k',
                                           'random-100k',
                                           'medium',
                                           'medium-1k',
                                           'medium-10k',
                                           'medium-100k',
                                           'medium-replay',
                                           'medium-replay-1k',
                                           'medium-replay-10k',
                                           'medium-replay-100k',
                                           'expert', 'expert-1k', 'expert-10k',
                                           'expert-100k',
                                           'medium-expert', 'medium-expert-1k',
                                           'medium-expert-10k',
                                           'medium-expert-100k']
                      for env_name in ['Hopper-v2',
                                       'Walker2d-v2',
                                       'HalfCheetah-v2']]


@pytest.mark.parametrize('env_name,dataset_name', DATASET_ENV_PAIRS)
@pytest.mark.skipif(os.environ.get('SKIP_Q_LEARNING_DATASET_TEST', '0') == '1',
                    reason="forcefully skipped by user")
def test_get_qlearning_dataset(env_name, dataset_name):
    dataset = opcc.get_qlearning_dataset(env_name, dataset_name)


@pytest.mark.parametrize('env_name,dataset_name', DATASET_ENV_PAIRS)
@pytest.mark.skipif(os.environ.get('SKIP_SEQUENCE_DATASET_TEST', '0') == '1',
                    reason="forcefully skipped by user")
def test_get_sequence_dataset(env_name, dataset_name):
    dataset = opcc.get_sequence_dataset(env_name, dataset_name)