"""
The `oai-analysis` command.

    oai-analysis IMAGE OUTPUT_DIR left|right [--config FILE]   process one image (see pipeline_cli.py)
    oai-analysis config dump [--config FILE]                   print the full pipeline config
    oai-analysis init PROJECT --manifest CSV --data-root DIR   create a project (see project.py)
    oai-analysis run PROJECT                                   process the cases that are not up to date
    oai-analysis status PROJECT                                show where every case is
"""
import argparse
import csv
import sys

COMMANDS = ("config", "init", "run", "status")
STEP_COLUMNS = {"preprocess": "prep", "segment": "seg", "register": "reg", "mesh": "mesh", "thickness": "thick",
                "atlas_map": "atlas", "biomarkers": "bio"}
SYMBOLS = {"done": "✓", "review": "!", "failed": "✗", "outdated": "~", "missing": ""}


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

    init_parser = commands.add_parser("init", help="Create a project folder")
    init_parser.add_argument("project")
    init_parser.add_argument("--manifest", help="knee-sarg manifest CSV (patient_id, study_uid, vol_path, month, "
                                                "study_description); without it, cases.csv starts empty")
    init_parser.add_argument("--data-root", help="Folder the manifest's paths are relative to, e.g. O:/OAI")
    init_parser.add_argument("--patients", help="Comma-separated patient IDs to include (default: all)")
    init_parser.add_argument("--months", help="Comma-separated visit months to include, e.g. 0,96 (default: all)")
    init_parser.add_argument("--laterality", choices=["left", "right"], help="Only this knee (default: both)")
    init_parser.add_argument("--config", help="YAML file with pipeline settings (see `oai-analysis config dump`)")

    selectors = argparse.ArgumentParser(add_help=False)
    selectors.add_argument("project")
    selectors.add_argument("--cases", help="Comma-separated case IDs (default: all)")
    selectors.add_argument("--failed", action="store_true", help="Only cases that failed")
    selectors.add_argument("--stale", action="store_true", help="Only cases that are outdated by a config change")
    selectors.add_argument("--failed-at", metavar="STEP", help="Only cases that failed at this step, e.g. mesh")
    selectors.add_argument("--needs-review", action="store_true", help="Only cases that a sanity check flagged")

    run_parser = commands.add_parser("run", parents=[selectors], help="Process the cases that are not up to date")
    run_parser.add_argument("--from", dest="from_step", metavar="STEP",
                            help="Redo this step, and the steps that use its outputs, even if up to date")
    run_parser.add_argument("--batch-size", type=int, metavar="N",
                            help="Cases whose images go through segmentation and registration together "
                                 "(default: batch_size in project.yaml)")
    status_parser = commands.add_parser("status", parents=[selectors], help="Show the status of every case")
    status_parser.add_argument("--csv", help="Also write the table to this CSV file")
    args = parser.parse_args(argv)

    if args.command == "config" and args.config_command == "dump":
        from oai_analysis import config
        from oai_analysis.pipeline import OAPipeline
        pipeline = OAPipeline.from_config(args.config) if args.config else OAPipeline()
        print(config.dump(pipeline.config()), end="")
    elif args.command == "init":
        return init(args)
    elif args.command == "run":
        return run(args)
    elif args.command == "status":
        return status(args)


def init(args):
    from oai_analysis.project import Project

    if args.manifest:
        if not args.data_root:
            sys.exit("--manifest needs --data-root, the folder its paths are relative to")
        project = Project.from_manifest(args.project, args.manifest, args.data_root, _list(args.patients),
                                        _list(args.months), args.laterality, args.config)
    else:
        project = Project.create(args.project, pipeline_config=args.config, data_root=args.data_root)
    print(f"Created {project.path} with {len(project.cases)} cases. Settings are in project.yaml, cases in cases.csv.")


def run(args):
    from oai_analysis.project import FAILED, Project

    project = Project(args.project)
    case_ids = _selected(project, args, project.pipeline())
    if args.batch_size is not None and args.batch_size < 1:
        sys.exit("--batch-size must be at least 1")
    statuses = project.run([s.case_id for s in case_ids], args.from_step, batch_size=args.batch_size)
    return 1 if any(s.status == FAILED for s in statuses) else 0


def status(args):
    from oai_analysis.project import Project

    _utf8_output()
    project = Project(args.project)
    pipeline = project.pipeline()
    selected = _selected(project, args, pipeline)
    project.write_index(project.status(pipeline))
    counts = {}
    for s in selected:
        counts[s.status] = counts.get(s.status, 0) + 1
    print(f"{len(selected)} {'case' if len(selected) == 1 else 'cases'}: " + ", ".join(f"{n} {name}" for name, n in counts.items()))
    if not selected:
        return
    rows = [[s.case_id, s.status, *[SYMBOLS[state] for state in _step_columns(s).values()], s.note]
            for s in selected]
    header = ["case", "status", *_step_columns(selected[0]).keys(), "note"]
    widths = [max(len(str(row[i])) for row in [header, *rows]) for i in range(len(header) - 1)]
    print()
    for row in [header, *rows]:
        print("  ".join(str(value).ljust(width) for value, width in zip(row, widths)) + "  " + row[-1])
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["case_id", "status", "failed_step", "note", *selected[0].steps, "review"])
            for s in selected:
                review = "; ".join(message for messages in s.review.values() for message in messages)
                writer.writerow([s.case_id, s.status, s.failed_step or "", s.note, *s.steps.values(), review])
        print(f"\nWrote {args.csv}")


def _selected(project, args, pipeline):
    case_ids = _list(args.cases)
    unknown = [c for c in case_ids or [] if c not in project.cases]
    if unknown:
        sys.exit(f"Not in cases.csv: {', '.join(unknown)}")
    return project.select(project.status(pipeline, case_ids), args.failed, args.stale, args.failed_at,
                          args.needs_review)


def _step_columns(case_status) -> dict[str, str]:
    """One column per step, with the biomarker steps combined: their worst state. "review": done, but flagged."""
    columns = {}
    for name, state in case_status.steps.items():
        if state == "done" and name in case_status.review:
            state = "review"
        column = STEP_COLUMNS["biomarkers" if name.startswith("biomarkers/") else name]
        order = ["failed", "outdated", "missing", "review", "done"]
        if column not in columns or order.index(state) < order.index(columns[column]):
            columns[column] = state
    return columns


def _list(text):
    return [item.strip() for item in text.split(",") if item.strip()] if text else None


def _utf8_output():
    """Print the status symbols even when the output is redirected on Windows."""
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(encoding="utf-8")


if __name__ == "__main__":
    main()
