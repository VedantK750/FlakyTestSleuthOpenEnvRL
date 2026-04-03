from env.environment import FlakySleuthEnv
from env.models import FlakySleuthAction


def test_reset_and_step_smoke():
    env = FlakySleuthEnv(dataset_path="dataset/py_tasks.csv")
    obs = env.reset()

    assert obs.test_name
    assert obs.task_type in {"classify", "root_cause", "fix_proposal"}

    action = FlakySleuthAction(action_type="search_code", argument="random")
    next_obs, reward, done, info = env.step(action)

    assert isinstance(next_obs.file_tree, list)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
