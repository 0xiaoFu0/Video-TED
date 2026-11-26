"""Exercise both comparison groups without Ray, GPUs, SSH, or training weights."""
import json
import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'examples/action_recognition/run_data_comparison.sh'
HOSTS = {
    'GROUP_A_HEAD': '10.20.0.1', 'GROUP_A_WORKER': '10.20.0.2',
    'GROUP_B_HEAD': '10.20.0.3', 'GROUP_B_WORKER': '10.20.0.4',
}
FULL_REFERENCE_POLICY = 'all_nonerror_generated_trees_with_reference_loss'


def write_comparison_rows(directory, values):
    (directory / 'train.rlsd.think.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in values))
    sft = [dict(messages=[dict(role='user', content=row['prompt']),
                          dict(role='assistant', content=row['reference_response'])],
                answer=row['answer'], videos=row['videos'], metadata=row['metadata']) for row in values]
    (directory / 'train.sft.think.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in sft))


def update_manifest(data_root, **updates):
    path = data_root / 'manifest.json'
    value = json.loads(path.read_text())
    value.update(updates)
    path.write_text(json.dumps(value))


def refresh_hashes(data_root):
    paths = sorted(data_root.glob('*/train.*.think.jsonl'))
    update_manifest(data_root, output_sha256={str(path.relative_to(data_root)):
                                            hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})


@pytest.fixture
def comparison(tmp_path):
    data_root = tmp_path / 'data root with spaces'
    accepted = dict(prompt='<video> Which action?', answer='run', videos=['/video/accepted.mp4'],
                    teacher_context='Generated tree evidence for run.',
                    reference_response='<think>Action: run\n  Motion: runs forward</think><answer>run</answer>',
                    reference_status='unreviewed', metadata=dict(sample_id='accepted', annotation_status='accepted',
                                                                 reference_tokens_including_eos=1200))
    needs_review = dict(prompt='<video> Which action?', answer='jump', videos=['/video/review.mp4'],
                        teacher_context='Generated tree evidence for jump.',
                        reference_response='<think>Action: jump\n  Motion: jumps upward</think><answer>jump</answer>',
                        reference_status='unreviewed',
                        metadata=dict(sample_id='needs_review', annotation_status='needs_review',
                                      reference_tokens_including_eos=1800))
    for variant in ('qualified', 'full'):
        directory = data_root / variant
        directory.mkdir(parents=True)
        write_comparison_rows(directory, [accepted] if variant == 'qualified' else [accepted, needs_review])
    (data_root / 'manifest.json').write_text(json.dumps({
        'merge_schema_version': 1,
        'planned': 3, 'qualified_rows': 1, 'full_rows': 2,
        'statuses': dict(accepted=1, needs_review=1, error=1),
        'source_counts': {
            'old': dict(planned=2, qualified_rows=1, full_rows=2,
                        statuses=dict(accepted=1, needs_review=1, error=0)),
            'incremental': dict(planned=1, qualified_rows=0, full_rows=0,
                                statuses=dict(accepted=0, needs_review=0, error=1)),
        },
        'expected_counts': dict(planned=3, qualified=1, full=2),
        'reference_answer_policy': 'original_qa_label', 'error_policy': 'omit_without_retry',
        'reference_max_length': 1800, 'max_observed_reference_tokens': 1800,
        'full_reference_policy': FULL_REFERENCE_POLICY}))
    refresh_hashes(data_root)
    bindir = tmp_path / 'env/bin'
    bindir.mkdir(parents=True)
    (bindir / 'activate').write_text('export PATH="' + str(bindir) + ':$PATH"\n')
    recorder = '#!' + sys.executable + '''
import json, os, sys
with open(os.environ['AR_TEST_CALLS'], 'a') as stream:
    stream.write(json.dumps({'tool': sys.argv[0], 'args': sys.argv[1:],
        'env': {key: os.getenv(key) for key in ('TRAIN_DATA', 'SFT_DATA', 'MODEL_PATH',
            'OUTPUT_ROOT', 'RUN_NAME', 'HEAD_HOST', 'WORKER_HOST', 'RAY_PORT',
            'AR_NCCL_PORT', 'RAY_ADDRESS', 'CUDA_VISIBLE_DEVICES')}}) + '\\n')
sys.exit(7 if os.getenv('AR_TEST_FAIL_CLUSTER') == '1' and 'verl.ar.check_cluster' in sys.argv else 0)
'''
    for name in ('python', 'ray'):
        (bindir / name).write_text(recorder)
        (bindir / name).chmod(0o755)
    (bindir / 'hostname').write_text('#!/bin/sh\nprintf "%s\\n" "$AR_TEST_NODE_IP"\n')
    (bindir / 'hostname').chmod(0o755)
    calls_path = tmp_path / 'calls.jsonl'
    env = {**os.environ, **HOSTS,
           'TRAIN_ENV_ROOT': str(bindir.parent), 'EXPERIMENT_DATA_ROOT': str(data_root),
           'OUTPUT_ROOT': str(tmp_path / 'outputs'), 'MODEL_PATH': '/models/base with spaces',
           'TRAIN_DATA': '/old/unrelated.jsonl', 'SFT_DATA': '/old/unrelated.sft.jsonl',
           'HEAD_HOST': '1.1.1.1', 'WORKER_HOST': '1.1.1.2',
           'RAY_PORT': '6379', 'AR_NCCL_PORT': '29511', 'DRY_RUN': '0',
           'AR_EXPECT_OLD_PLANNED': '2', 'AR_EXPECT_INCREMENTAL_PLANNED': '1',
           'AR_EXPECT_QUALIFIED_ROWS': '1', 'AR_EXPECT_FULL_ROWS': '2',
           'RUN_NAME': 'comparison_test', 'AR_TEST_NODE_IP': HOSTS['GROUP_A_HEAD'],
           'AR_TEST_CALLS': str(calls_path)}

    def run(variant, mode, *args, **overrides):
        if calls_path.exists():
            calls_path.unlink()
        result = subprocess.run(['bash', str(SCRIPT), variant, mode, *args],
                                env={**env, **overrides}, cwd=tmp_path,
                                capture_output=True, text=True, timeout=15)
        calls = [json.loads(line) for line in calls_path.read_text().splitlines()] if calls_path.exists() else []
        return result, calls

    return run, data_root


def dry_environment(stdout):
    tokens = shlex.split(stdout)
    assert tokens[0] == 'env'
    end = tokens.index('bash')
    return dict(token.split('=', 1) for token in tokens[1:end]), tokens[end:]


@pytest.mark.parametrize('variant,head,worker,ray,nccl', [
    ('qualified', HOSTS['GROUP_A_HEAD'], HOSTS['GROUP_A_WORKER'], '6383', '29523'),
    ('full', HOSTS['GROUP_B_HEAD'], HOSTS['GROUP_B_WORKER'], '6384', '29524'),
])
@pytest.mark.parametrize('mode', ['head', 'worker', 'cluster', 'smoke', 'train', 'check-video'])
def test_all_dry_modes_are_quoted_and_do_not_launch(comparison, variant, head, worker, ray, nccl, mode):
    run, data_root = comparison
    result, calls = run(variant, mode, DRY_RUN='1', TRAIN_ENV_ROOT='/missing/env')
    assert result.returncode == 0, result.stderr
    assert not calls
    env, command = dry_environment(result.stdout)
    assert env['TRAIN_DATA'] == str(data_root / variant / 'train.rlsd.think.jsonl')
    assert env['MODEL_PATH'] == '/models/base with spaces'
    assert env['HEAD_HOST'] == head and env['WORKER_HOST'] == worker
    assert env['RAY_PORT'] == ray and env['AR_NCCL_PORT'] == nccl
    assert Path(env['OUTPUT_ROOT']).name == variant
    if mode in ('smoke', 'train'):
        assert command[-3:] == ['opsd.ar_reference_max_length=1800',
                                'opsd.ar_allow_unreviewed_reference=true', 'opsd.ar_reference_coef=0.1']


@pytest.mark.parametrize('overrides,message', [
    ({'GROUP_B_WORKER': ''}, 'Missing GROUP_B_WORKER'),
    ({'GROUP_B_WORKER': HOSTS['GROUP_A_HEAD']}, 'must be distinct'),
    ({'GROUP_B_WORKER': '10.20.0.256'}, 'invalid IPv4 address'),
    ({'GROUP_B_WORKER': '10.020.0.4'}, 'must be an IPv4 address'),
    ({'DRY_RUN': 'yes'}, 'DRY_RUN must'),
])
def test_invalid_groups_fail_before_launch(comparison, overrides, message):
    run, _ = comparison
    result, calls = run('qualified', 'head', **overrides)
    assert result.returncode == 2 and message in result.stderr
    assert not calls


def test_both_train_from_same_model_and_configuration_with_distinct_outputs(comparison):
    run, data_root = comparison
    drivers = []
    for variant, node in [('qualified', HOSTS['GROUP_A_HEAD']), ('full', HOSTS['GROUP_B_HEAD'])]:
        result, calls = run(variant, 'train', AR_TEST_NODE_IP=node)
        assert result.returncode == 0, result.stderr
        driver = [call for call in calls if 'verl.trainer.opsd_main' in call['args']]
        assert len(driver) == 1
        drivers.append(driver[0])
        args = driver[0]['args']
        assert 'worker.actor.model.model_path=/models/base with spaces' in args
        assert 'trainer.nnodes=2' in args and 'trainer.n_gpus_per_node=8' in args
        assert 'opsd.ar_reference_max_length=1800' in args
        assert 'opsd.ar_allow_unreviewed_reference=true' in args
        assert 'opsd.ar_reference_coef=0.1' in args
        assert f'data.train_files={data_root}/{variant}/train.rlsd.think.jsonl' in args
        assert not any('trainer.max_steps=' in arg for arg in args)
    assert drivers[0]['env']['OUTPUT_ROOT'] != drivers[1]['env']['OUTPUT_ROOT']
    assert drivers[0]['env']['RAY_ADDRESS'] == f"{HOSTS['GROUP_A_HEAD']}:6383"
    assert drivers[1]['env']['RAY_ADDRESS'] == f"{HOSTS['GROUP_B_HEAD']}:6384"
    # Only input/output identities differ; optimizer, seed, objectives and epoch
    # settings all come from the same existing config and shell launcher.
    common = lambda args: [arg for arg in args if not arg.startswith((
        'data.train_files=', 'trainer.save_checkpoint_path='))]
    assert common(drivers[0]['args']) == common(drivers[1]['args'])


def test_failed_cluster_check_blocks_driver(comparison):
    run, _ = comparison
    result, calls = run('full', 'train', AR_TEST_NODE_IP=HOSTS['GROUP_B_HEAD'], AR_TEST_FAIL_CLUSTER='1')
    assert result.returncode != 0
    assert not any('verl.trainer.opsd_main' in call['args'] for call in calls)


def test_full_video_probe_uses_its_generated_cot_sft_for_backward(comparison):
    run, data_root = comparison
    assert (data_root / 'full/train.sft.think.jsonl').exists()
    result, calls = run('full', 'check-video', CHECK_SAMPLES='3')
    assert result.returncode == 0, result.stderr
    assert len(calls) == 1
    args = calls[0]['args']
    assert '--backward' in args
    assert args[args.index('--data') + 1] == str(data_root / 'full/train.rlsd.think.jsonl')
    assert args[args.index('--sft-data') + 1] == str(data_root / 'full/train.sft.think.jsonl')
    assert args[args.index('--samples') + 1] == '3'


def test_qualified_video_probe_keeps_backward_check(comparison):
    run, data_root = comparison
    result, calls = run('qualified', 'check-video')
    assert result.returncode == 0, result.stderr
    assert '--backward' in calls[0]['args']
    assert calls[0]['args'][calls[0]['args'].index('--sft-data') + 1] == str(
        data_root / 'qualified/train.sft.think.jsonl')


@pytest.mark.parametrize('value', [0, -1, 1024.0, True, '1024'])
def test_invalid_shared_reference_limit_is_rejected(comparison, value):
    run, data_root = comparison
    update_manifest(data_root, reference_max_length=value)
    result, calls = run('qualified', 'train', DRY_RUN='1')
    assert result.returncode == 2 and not calls


def test_missing_manifest_and_missing_data_fail_early(comparison):
    run, data_root = comparison
    (data_root / 'manifest.json').unlink()
    result, calls = run('full', 'train', DRY_RUN='1')
    assert result.returncode == 2 and not calls
    (data_root / 'full/train.rlsd.think.jsonl').unlink()
    result, calls = run('full', 'cluster')
    assert result.returncode == 2 and not calls


@pytest.mark.parametrize('variant', ['qualified', 'full'])
@pytest.mark.parametrize('mode', ['smoke', 'train', 'check-video'])
@pytest.mark.parametrize('policy', [None, 'accepted_tree_else_label_hint_with_zero_reference_loss'])
def test_old_full_reference_policy_is_rejected_even_in_dry_run(comparison, variant, mode, policy):
    run, data_root = comparison
    update_manifest(data_root, full_reference_policy=policy)
    result, calls = run(variant, mode, DRY_RUN='1')
    assert result.returncode == 2 and not calls
    assert 'full_reference_policy' in result.stderr


@pytest.mark.parametrize('variant', ['qualified', 'full'])
@pytest.mark.parametrize('mode', ['smoke', 'train', 'check-video'])
@pytest.mark.parametrize('failure,message', [
    ('empty_reference', 'missing generated CoT reference'),
    ('rejected_reference', 'ineligible for CoT supervision'),
    ('empty_context', 'missing CoT teacher context'),
    ('mismatched_sft', 'RLSD/SFT pair mismatch'),
    ('mismatched_ids', 'RLSD/SFT sample IDs differ'),
    ('missing_sft', 'No such file'),
])
def test_reference_eligibility_and_sft_pair_checks_precede_gpu_work(comparison, variant, mode, failure, message):
    run, data_root = comparison
    directory = data_root / variant
    values = [json.loads(line) for line in (directory / 'train.rlsd.think.jsonl').read_text().splitlines()]
    # In full, exercise the needs_review row rather than only the accepted row.
    row = values[-1]
    if failure == 'empty_reference':
        row['reference_response'] = '  '
    elif failure == 'rejected_reference':
        row['reference_status'] = 'rejected'
    elif failure == 'empty_context':
        row['teacher_context'] = ''
    write_comparison_rows(directory, values)
    if failure in ('mismatched_sft', 'mismatched_ids'):
        sft_path = directory / 'train.sft.think.jsonl'
        sft_values = [json.loads(line) for line in sft_path.read_text().splitlines()]
        if failure == 'mismatched_sft':
            sft_values[-1]['messages'][1]['content'] = 'unrelated old CoT'
        else:
            sft_values[-1]['metadata']['sample_id'] = 'unrelated_sample'
        sft_path.write_text(''.join(json.dumps(value) + '\n' for value in sft_values))
    elif failure == 'missing_sft':
        (directory / 'train.sft.think.jsonl').unlink()
    # Keep provenance valid so these cases exercise semantic checks, rather
    # than only the independent changed-file hash check below.
    if failure != 'missing_sft':
        refresh_hashes(data_root)
    result, calls = run(variant, mode, DRY_RUN='1', TRAIN_ENV_ROOT='/missing/env')
    assert result.returncode == 2 and not calls
    assert message in result.stderr


@pytest.mark.parametrize('variant', ['qualified', 'full'])
@pytest.mark.parametrize('mode', ['smoke', 'train'])
def test_comparison_keeps_generated_reference_supervision_enabled(comparison, variant, mode):
    run, _ = comparison
    result, calls = run(variant, mode, 'opsd.ar_reference_coef=0',
                        'opsd.ar_allow_unreviewed_reference=false', DRY_RUN='1')
    assert result.returncode == 0, result.stderr
    assert not calls
    _, command = dry_environment(result.stdout)
    assert command[-2:] == ['opsd.ar_allow_unreviewed_reference=true', 'opsd.ar_reference_coef=0.1']


@pytest.mark.parametrize('schema', [None, 0, 2, True, '1'])
@pytest.mark.parametrize('mode', ['head', 'worker', 'cluster', 'check-video', 'smoke', 'train'])
def test_every_mode_rejects_an_unmerged_or_wrong_version_pool(comparison, schema, mode):
    run, data_root = comparison
    update_manifest(data_root, merge_schema_version=schema)
    result, calls = run('full', mode, DRY_RUN='1', TRAIN_ENV_ROOT='/missing/env')
    assert result.returncode == 2 and not calls
    assert 'merge_schema_version' in result.stderr


@pytest.mark.parametrize('failure,message', [
    ('old_planned', 'source planned/status count mismatch: old'),
    ('incremental_planned', 'source planned/status count mismatch: incremental'),
    ('source_full', 'source qualified/full count mismatch: old'),
    ('qualified_rows', 'merged planned/qualified/full/status count mismatch'),
    ('full_rows', 'merged planned/qualified/full/status count mismatch'),
    ('expected_counts', 'manifest expected_counts mismatch: full'),
    ('missing_source', 'exactly old and incremental'),
])
def test_manifest_source_and_combined_counts_are_locked(comparison, failure, message):
    run, data_root = comparison
    manifest_path = data_root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if failure == 'old_planned':
        manifest['source_counts']['old']['planned'] += 1
    elif failure == 'incremental_planned':
        manifest['source_counts']['incremental']['planned'] += 1
    elif failure == 'source_full':
        manifest['source_counts']['old']['full_rows'] -= 1
    elif failure in ('qualified_rows', 'full_rows'):
        manifest[failure] += 1
    elif failure == 'expected_counts':
        manifest['expected_counts']['full'] += 1
    else:
        del manifest['source_counts']['incremental']
    manifest_path.write_text(json.dumps(manifest))
    result, calls = run('qualified', 'train', DRY_RUN='1')
    assert result.returncode == 2 and not calls
    assert message in result.stderr


@pytest.mark.parametrize('variant', ['qualified', 'full'])
def test_actual_file_hash_must_match_the_merge_manifest(comparison, variant):
    run, data_root = comparison
    path = data_root / variant / 'train.rlsd.think.jsonl'
    with path.open('a') as stream:
        stream.write('\n')
    result, calls = run('qualified', 'head', DRY_RUN='1')
    assert result.returncode == 2 and not calls
    assert f'output SHA256 mismatch: {variant}/train.rlsd.think.jsonl' in result.stderr


def test_all_four_training_payload_hashes_are_required(comparison):
    run, data_root = comparison
    manifest = json.loads((data_root / 'manifest.json').read_text())
    del manifest['output_sha256']['full/train.sft.think.jsonl']
    update_manifest(data_root, output_sha256=manifest['output_sha256'])
    result, calls = run('qualified', 'check-video', DRY_RUN='1')
    assert result.returncode == 2 and not calls
    assert 'output_sha256 is missing training files' in result.stderr


def test_actual_row_count_is_checked_after_valid_hashes(comparison):
    run, data_root = comparison
    directory = data_root / 'full'
    values = [json.loads(line) for line in (directory / 'train.rlsd.think.jsonl').read_text().splitlines()]
    write_comparison_rows(directory, values[:1])
    refresh_hashes(data_root)
    result, calls = run('full', 'smoke', DRY_RUN='1')
    assert result.returncode == 2 and not calls
    assert 'actual RLSD/SFT row count mismatch: full' in result.stderr


@pytest.mark.parametrize('variant,status', [('qualified', 'needs_review'), ('full', 'error')])
def test_group_annotation_status_cannot_be_replaced(comparison, variant, status):
    run, data_root = comparison
    directory = data_root / variant
    values = [json.loads(line) for line in (directory / 'train.rlsd.think.jsonl').read_text().splitlines()]
    values[-1]['metadata']['annotation_status'] = status
    write_comparison_rows(directory, values)
    refresh_hashes(data_root)
    result, calls = run(variant, 'train', DRY_RUN='1')
    assert result.returncode == 2 and not calls
    assert f'invalid {variant} annotation_status' in result.stderr


def test_both_groups_use_the_actual_longest_full_reference(comparison):
    run, data_root = comparison
    directory = data_root / 'full'
    values = [json.loads(line) for line in (directory / 'train.rlsd.think.jsonl').read_text().splitlines()]
    values[-1]['metadata']['reference_tokens_including_eos'] = 2200
    write_comparison_rows(directory, values)
    refresh_hashes(data_root)
    result, calls = run('qualified', 'train', DRY_RUN='1')
    assert result.returncode == 2 and not calls
    assert 'shared reference_max_length' in result.stderr
    update_manifest(data_root, reference_max_length=2200, max_observed_reference_tokens=2200)
    for variant in ('qualified', 'full'):
        result, calls = run(variant, 'train', DRY_RUN='1')
        assert result.returncode == 0 and not calls, result.stderr
        _, command = dry_environment(result.stdout)
        assert 'opsd.ar_reference_max_length=2200' in command


def test_default_output_and_run_name_are_v3_specific(comparison):
    run, _ = comparison
    result, calls = run('qualified', 'train', DRY_RUN='1', OUTPUT_ROOT='', RUN_NAME='')
    assert result.returncode == 0 and not calls, result.stderr
    env, _ = dry_environment(result.stdout)
    assert env['OUTPUT_ROOT'] == '/gemini-3/space/zwj/AR-RLSD/outputs/data_comparison_v3/qualified'
    assert env['RUN_NAME'].startswith('ar_qwen25_qualified_v3_train_')


def test_sourced_node_configuration_sets_all_v3_defaults():
    names = ('GROUP_A_HEAD', 'GROUP_A_WORKER', 'GROUP_B_HEAD', 'GROUP_B_WORKER',
             'EXPERIMENT_DATA_ROOT', 'OUTPUT_ROOT', 'MODEL_PATH', 'TRAIN_ENV_ROOT',
             'AR_EXPECT_OLD_PLANNED', 'AR_EXPECT_INCREMENTAL_PLANNED',
             'AR_EXPECT_QUALIFIED_ROWS', 'AR_EXPECT_FULL_ROWS')
    command = 'source "$1"; ' + '; '.join('printf "%s\\n" "${' + name + '}"' for name in names)
    result = subprocess.run(['bash', '-c', command, 'v3-env',
                             str(ROOT / 'examples/action_recognition/comparison_nodes.env')],
                            capture_output=True, text=True, check=True)
    values = dict(zip(names, result.stdout.splitlines()))
    assert [values[name] for name in names[:4]] == [
        '192.0.2.12', '192.0.2.13', '192.0.2.10', '192.0.2.11']
    assert values['EXPERIMENT_DATA_ROOT'] == str(ROOT / 'data/data_comparison_v3')
    assert values['OUTPUT_ROOT'].endswith('/outputs/data_comparison_v3')
    assert values['MODEL_PATH'].endswith('/Qwen2.5-VL-7B-Instruct')
    assert [int(values[name]) for name in names[-4:]] == [11964, 7698, 9014, 19621]


@pytest.mark.parametrize('variant,mode,node,expected', [
    ('qualified', 'head', HOSTS['GROUP_A_HEAD'], '--port=6383'),
    ('qualified', 'worker', HOSTS['GROUP_A_WORKER'], f"--address={HOSTS['GROUP_A_HEAD']}:6383"),
    ('full', 'head', HOSTS['GROUP_B_HEAD'], '--port=6384'),
    ('full', 'worker', HOSTS['GROUP_B_WORKER'], f"--address={HOSTS['GROUP_B_HEAD']}:6384"),
])
def test_real_ray_launch_chain_uses_group_role_and_port(comparison, variant, mode, node, expected):
    run, _ = comparison
    result, calls = run(variant, mode, AR_TEST_NODE_IP=node)
    assert result.returncode == 0, result.stderr
    assert len(calls) == 1 and expected in calls[0]['args']
    assert '--num-gpus=8' in calls[0]['args']


@pytest.mark.parametrize('variant,mode', [('qualified', 'head'), ('qualified', 'worker'),
                                         ('full', 'head'), ('full', 'worker'), ('full', 'train')])
def test_wrong_local_node_blocks_ray_or_training(comparison, variant, mode):
    run, _ = comparison
    result, calls = run(variant, mode, AR_TEST_NODE_IP='10.99.99.99')
    assert result.returncode != 0 and not calls
    assert 'does not belong to this node' in result.stderr


def test_real_merged_export_is_accepted_by_both_launcher_dry_runs():
    # Use the production merger and its fake annotation backend in an isolated
    # interpreter. This checks the actual manifest/hash/metadata contract rather
    # than only fixtures that mirror the shell preflight implementation.
    annotation = Path(os.environ.get('AR_ANNOTATION_SOURCE', str(ROOT / 'tools/annotation')))
    if not (annotation / 'tests/test_merged_comparison.py').is_file():
        pytest.skip('Optional legacy annotation pipeline: set AR_ANNOTATION_SOURCE to its source directory')
    program = '''
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
sys.path[:0] = [sys.argv[1], str(Path(sys.argv[1]) / 'tests')]
from test_merged_comparison import MergedComparisonTest
test = MergedComparisonTest('test_two_pools_keep_all_nonerror_trees_and_pair_sft_rlsd')
test.setUp()
try:
    with contextlib.redirect_stdout(io.StringIO()):
        pools = test.fixture()
        output = test.root / 'merged_launcher_data'
        test.merge(pools, output, expect_qualified=2, expect_full=4, expect_planned=6)
    env = dict(os.environ, DRY_RUN='1', TRAIN_ENV_ROOT='/missing/env',
        GROUP_A_HEAD='192.0.2.12', GROUP_A_WORKER='192.0.2.13',
        GROUP_B_HEAD='192.0.2.10', GROUP_B_WORKER='192.0.2.11',
        EXPERIMENT_DATA_ROOT=str(output), AR_EXPECT_OLD_PLANNED='3',
        AR_EXPECT_INCREMENTAL_PLANNED='3', AR_EXPECT_QUALIFIED_ROWS='2', AR_EXPECT_FULL_ROWS='4')
    for variant in ('qualified', 'full'):
        result = subprocess.run(['bash', sys.argv[2], variant, 'train'],
            env=env, text=True, capture_output=True, timeout=15)
        assert result.returncode == 0, result.stderr
        assert 'opsd.ar_reference_max_length=3001' in result.stdout
        print(variant + ': merged export dry run passed')
finally:
    test.tearDown()
'''
    result = subprocess.run([sys.executable, '-c', program, str(annotation), str(SCRIPT)],
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'qualified: merged export dry run passed' in result.stdout
    assert 'full: merged export dry run passed' in result.stdout
