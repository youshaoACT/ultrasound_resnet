import ml_collections

def config_dict(**kwargs):
    return ml_collections.ConfigDict(kwargs)

def get_config():
    return config_dict(
        seed=0,
        dataset=config_dict(
            seed=0,
            dataset=config_dict(
                name="cifar10",
            )
        )
    )