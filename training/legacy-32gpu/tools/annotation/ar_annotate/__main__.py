import argparse
import json
from pathlib import Path

from .common import ROOT, read_json


def main():
    p = argparse.ArgumentParser(description='Qwen3.8 offline video evidence annotation for AR-RLSD')
    sub = p.add_subparsers(dest='command', required=True)
    exclusions = sub.add_parser('exclusions', help='Refresh evaluation video-ID exclusion asset')
    exclusions.add_argument('--eval-root', required=True)
    exclusions.add_argument('--out', required=True)
    prep = sub.add_parser('prepare', help='Strict video matching and evaluation exclusions; CPU only')
    prep.add_argument('--config', default=str(ROOT / 'config.server.json'))
    prep.add_argument('--out', required=True)
    prep.add_argument('--retain-evaluation-overlaps', action='store_true',
                      help='Keep matching videos in the configured evaluation sets and record membership')
    combined = sub.add_parser('prepare-combined', help='Reuse SSv2/K710 candidates and add existing HMDB51 train videos')
    combined.add_argument('--config', default=str(ROOT / 'config.server.json'))
    combined.add_argument('--existing-candidates', required=True)
    combined.add_argument('--out', required=True)
    increment = sub.add_parser('prepare-overlap-incremental',
                               help='Recover excluded overlaps and skip all records from a completed full run; CPU only')
    increment.add_argument('--config', default=str(ROOT / 'config.server.json'))
    increment.add_argument('--existing-candidates', required=True)
    increment.add_argument('--existing-run', required=True)
    increment.add_argument('--out', required=True)
    missing = sub.add_parser('prepare-incremental',
                             help='Recover omitted exact video matches and skip all records from a completed full run; CPU only')
    missing.add_argument('--config', default=str(ROOT / 'config.server.json'))
    missing.add_argument('--existing-candidates', required=True)
    missing.add_argument('--existing-run', required=True)
    missing.add_argument('--out', required=True)
    missing.add_argument('--scope', choices=['all-missing', 'evaluation-overlaps'], default='all-missing',
                         help='Default: include all omitted matching videos, including evaluation overlaps')
    ann = sub.add_parser('annotate', help='Two-pass annotation; resumes identical runs')
    ann.add_argument('--config', default=str(ROOT / 'config.server.json'))
    ann.add_argument('--candidates', required=True)
    ann.add_argument('--out', required=True)
    ann.add_argument('--model')
    ann.add_argument('--sample-id', action='append', help='Select an existing candidate ID for isolated debugging; repeatable')
    ann.add_argument('--limit-per-dataset', type=int, default=0, help='0 = all; pilot and full runs need separate output directories')
    ann.add_argument('--num-shards', type=int, default=1)
    ann.add_argument('--shard-index', type=int, default=0)
    ann.add_argument('--retry-errors', action='store_true')
    ann.add_argument('--rollouts', type=int, help='Fixed number of independent draws per video')
    ann.add_argument('--min-accepted-rollouts', type=int, help='Default: at least one verified draw')
    ann.add_argument('--seed', type=int, help='Base seed for deterministic per-sample/per-draw seeds')
    ann.add_argument('--prompt-profile', choices=['legacy', 'temporal_v1', 'tree_v1'])
    exp = sub.add_parser('export', help='Export paired RLSD/SFT files without overwriting base data')
    exp.add_argument('--config', default=str(ROOT / 'config.server.json'))
    exp.add_argument('--candidates', required=True)
    exp.add_argument('--run', required=True)
    exp.add_argument('--out', required=True)
    exp.add_argument('--base')
    exp.add_argument('--pool', action='store_true', help='Export only this annotated pool; do not append old HMDB51/base rows')
    exp.add_argument('--student-model')
    exp.add_argument('--max-reference-tokens', type=int, default=768)
    exp.add_argument('--allow-partial', action='store_true')
    report = sub.add_parser('summarize', help='Read global coverage across both nodes without loading a model')
    report.add_argument('--run', required=True)
    report.add_argument('--candidates', required=True)
    check = sub.add_parser('doctor', help='Inspect dependencies, CUDA, model files and video processor (no weights loaded)')
    check.add_argument('--config', default=str(ROOT / 'config.server.json'))
    args = p.parse_args()
    if args.command == 'exclusions':
        from .exclusions import build
        if Path(args.out).exists():
            raise FileExistsError('Choose a new exclusion file')
        build(args.eval_root, args.out)
    elif args.command == 'prepare':
        from .prepare import prepare
        prepare(args.config, args.out, retain_evaluation_overlaps=args.retain_evaluation_overlaps)
    elif args.command == 'prepare-combined':
        from .hmdb51 import combine
        combine(args.config, args.existing_candidates, args.out)
    elif args.command == 'prepare-overlap-incremental':
        from .incremental import prepare_overlap_incremental
        prepare_overlap_incremental(args.config, args.existing_candidates, args.existing_run, args.out)
    elif args.command == 'prepare-incremental':
        from .incremental import prepare_incremental
        prepare_incremental(args.config, args.existing_candidates, args.existing_run, args.out, scope=args.scope)
    elif args.command == 'annotate':
        from .annotate import annotate
        annotate(args)
    elif args.command == 'export':
        if args.pool and args.base:
            p.error('--pool cannot be combined with --base')
        from transformers import AutoTokenizer
        from .export import export_data
        cfg = read_json(args.config)
        tokenizer = AutoTokenizer.from_pretrained(args.student_model or cfg['student_model'],
                        local_files_only=True, trust_remote_code=False)
        export_data(args.candidates, args.run, None if args.pool else (args.base or cfg['base_rlsd']), args.out,
                    lambda s: len(tokenizer.encode(s, add_special_tokens=False)),
                    args.max_reference_tokens, args.allow_partial, pool_only=args.pool)
    elif args.command == 'summarize':
        from .report import summarize
        print(json.dumps(summarize(args.run, args.candidates), ensure_ascii=False, indent=2))
    else:
        import torch
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
        from .backend import environment, model_identity
        cfg = read_json(args.config)
        identity = model_identity(cfg['annotation']['model'])
        processor = AutoProcessor.from_pretrained(identity['path'], local_files_only=True, trust_remote_code=False)
        if not getattr(processor, 'video_processor', None):
            raise ValueError('No video processor')
        print(json.dumps(dict(versions=environment(), cuda=torch.version.cuda,
                  devices=[dict(name=torch.cuda.get_device_name(i),
                                memory_GiB=round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1))
                           for i in range(torch.cuda.device_count())],
                  processor=type(processor).__name__, patch=processor.video_processor.patch_size,
                  model=identity, actual_video_inference_tested=False), indent=2))
        if not torch.cuda.is_available():
            raise RuntimeError('No CUDA available')


if __name__ == '__main__':
    main()
