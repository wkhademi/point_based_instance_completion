import train.object_completion
import train.scene_completion
import test.scene_completion


TRAIN_FNCS = {
    "SeedFormer": train.object_completion.train,
    "RobustSeedFormer": train.object_completion.train,
    "SCCNet": train.scene_completion.train,
}

TEST_FNCS = {
    "SCCNet": test.scene_completion.test,
}


def get_train_fnc(model_type):
    run_fnc = TRAIN_FNCS[model_type]

    return run_fnc


def get_test_fnc(model_type):
    run_fnc = TEST_FNCS[model_type]

    return run_fnc