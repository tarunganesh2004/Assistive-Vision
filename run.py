from src.pipeline import Pipeline
from src.utils import load_config, setup_logging


def main():
    setup_logging()
    config = load_config()
    pipeline = Pipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
