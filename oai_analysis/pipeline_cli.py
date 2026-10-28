# Usage
# oai-analysis IMAGE OUTPUT_DIR left|right [--config FILE]
# python -m oai_analysis IMAGE OUTPUT_DIR left|right
# python oai_analysis/pipeline_cli.py IMAGE OUTPUT_DIR left|right  (the package must be installed)

import argparse

from oai_analysis.pipeline import OAPipeline, analysis_pipeline
from oai_analysis.stages import CaseInfo


def main(argv=None):
    parser = argparse.ArgumentParser(prog='oai-analysis', description='OAI Analysis CLI',
                                     epilog='Other commands: oai-analysis config dump')
    parser.add_argument('input_path', type=str, help='Path to image file or directory containing DICOM series')
    parser.add_argument('output_dir', type=str, help='Directory to make output files')
    parser.add_argument('laterality', type=str, help='Knee: left or right')
    parser.add_argument(
        '--no_intermediates',
        action='store_true',
        help='Do not write files representing intermediate steps to the output directory',
    )
    parser.add_argument('--config', type=str, help='YAML file choosing the stages and their parameters')

    args = parser.parse_args(argv)

    if args.config:
        OAPipeline.from_config(args.config).run_case(
            CaseInfo(args.input_path, args.laterality), args.output_dir, not args.no_intermediates)
    else:
        analysis_pipeline(args.input_path, args.output_dir, args.laterality, not args.no_intermediates)


if __name__ == '__main__':
    main()
