# ad hoc script to run counting test
import click
import yaml


@click.command()
@click.option("--config", type=str)
def main(config):
    print("config: ", config)

    with open(config, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    assert "step" in config, "step is not in the config"
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
