"""
The `oai-analysis` command.

    oai-analysis IMAGE OUTPUT_DIR left|right [--config FILE]   process one image (see pipeline_cli.py)
    oai-analysis config dump [--config FILE]                   print the full pipeline config
"""
import argparse
import sys

COMMANDS = ("config",)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in COMMANDS:  # the original command line: IMAGE OUTPUT_DIR LATERALITY
        from oai_analysis import pipeline_cli
        return pipeline_cli.main(argv)

    parser = argparse.ArgumentParser(prog="oai-analysis")
    commands = parser.add_subparsers(dest="command", required=True)
    config_parser = commands.add_parser("config", help="Pipeline configuration")
    config_commands = config_parser.add_subparsers(dest="config_command", required=True)
    dump_parser = config_commands.add_parser(
        "dump", help="Print the full config: the defaults, with the values from --config applied")
    dump_parser.add_argument("--config", help="YAML file with the settings to apply")
    args = parser.parse_args(argv)

    if args.command == "config" and args.config_command == "dump":
        from oai_analysis import config
        from oai_analysis.pipeline import OAPipeline
        pipeline = OAPipeline.from_config(args.config) if args.config else OAPipeline()
        print(config.dump(pipeline.config()), end="")


if __name__ == "__main__":
    main()
