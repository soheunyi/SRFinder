# ad hoc script to run counting test
import click
import yaml


@click.command()
@click.option("--config", "-c", type=str, multiple=True)
def main(config):
    cfg_list = config
    config_dicts = []
    for cfg in cfg_list:
        with open(cfg, "r") as ymlfile:
            config_dicts.append(yaml.safe_load(ymlfile))

    for i, config in enumerate(config_dicts):
        print(f"Running config {i+1} of {len(config_dicts)}")
        # print the config in a pretty way
        print(yaml.dump(config, default_flow_style=False))

        assert "step" in config, f"step is not in the config: {config}"
        step = config["step"]

        if step == 1:
            from step_1_base_fvt_training import routine
        elif step == 2:
            from step_2_smeared_fvt_training import routine
        elif step == 3:
            from step_3_define_CR_and_train_fvt import routine
        else:
            raise ValueError(f"Step {step} not supported")

        routine(config)


if __name__ == "__main__":
    main()
