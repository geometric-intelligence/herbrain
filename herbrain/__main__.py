import logging
from enum import Enum
from typing import List, Optional

import typer
from hydra import compose, initialize

app = typer.Typer(pretty_exceptions_short=False) # turn this to False if you want to see more the error path.


class PregnancyDataOptions(str, Enum):
    maternal = "maternal"
    multiple = "multiple"


def _launch_app(my_app, overrides, config_path, config_name, logging_level, **kwargs):
    logging.basicConfig(level=logging_level)

    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=config_name, overrides=overrides)

    my_app(cfg, **kwargs)


@app.command()
def pregnancy_app(
    data: PregnancyDataOptions = PregnancyDataOptions.multiple,
    overrides: Optional[List[str]] = typer.Argument(None),
    config_path: str = "config/pregnancy",
    config_name: str = "config",
    logging_level: int = 20,
    gpt: bool = False,
):
    """Launch pregnancy app."""
    from herbrain.pregnancy.app import my_app

    if overrides is None:
        overrides = []

    return _launch_app(
        my_app,
        overrides,
        config_path,
        config_name,
        logging_level,
        data=data.value,
        gpt=gpt,
    )


@app.command()
def menstrual_app(
    overrides: Optional[List[str]] = typer.Argument(None),
    config_path: str = "config/menstrual",
    config_name: str = "config",
    logging_level: int = 20,
):
    """Launch menstrual app."""
    from herbrain.menstrual.app import my_app

    return _launch_app(my_app, overrides, config_path, config_name, logging_level)


if __name__ == "__main__":
    app()
