"""Qwen3 adapter checks; no model weights or GPUs are required."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import qwen3_video as adapter
from protocol import messages, official, registry, load_dataset


@pytest.fixture
def legacy():
    path = ROOT / "upstream/src/qwen-vl-utils/src/qwen_vl_utils/vision_process.py"
    spec = importlib.util.spec_from_file_location("pinned_v6_vision_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("total,fps", [(2,30.), (3,30.), (9,29.97), (120,30.), (3000,29.97)])
def test_same_frames_as_original_decord(legacy, monkeypatch, total, fps):
    import numpy as np
    class Reader:
        def __init__(self, _): pass
        def __len__(self): return total
        def get_avg_fps(self): return fps
        def get_batch(self, indices):
            data = np.broadcast_to(np.array(indices)[:,None,None,None], (len(indices),32,64,3)).copy()
            return SimpleNamespace(asnumpy=lambda: data)
    monkeypatch.setitem(sys.modules, "decord", SimpleNamespace(VideoReader=Reader))
    monkeypatch.setattr(legacy, "get_video_reader_backend", lambda: "decord")
    old, sampled_fps = legacy._read_video_decord({"video":"test.avi"})
    new, metadata = adapter.decode_video("test.avi", legacy)
    assert old.equal(new)
    assert metadata["fps"] == fps  # NEVER replace by sampled_fps
    assert metadata["frames_indices"] == old[:,0,0,0].tolist()
    assert metadata["total_num_frames"] == total
    assert len(new) <= 16
    if total == 3000:
        assert sampled_fps != fps


def test_torchvision_fallback_preserves_timing(legacy, monkeypatch):
    import torch
    frames = torch.arange(90).reshape(90,1,1,1).expand(90,3,32,64)
    def fail(_): raise RuntimeError("test decode failure")
    monkeypatch.setitem(sys.modules, "decord", SimpleNamespace(VideoReader=fail))
    monkeypatch.setattr(legacy, "get_video_reader_backend", lambda: "decord")
    monkeypatch.setattr(legacy.io, "read_video", lambda *a, **k: (frames,None,{"video_fps":29.97}))
    with pytest.warns(UserWarning, match="decord failed"):
        decoded, metadata = adapter.decode_video("test.webm", legacy)
    assert metadata["video_backend"] == "torchvision"
    assert metadata["frames_indices"] == decoded[:,0,0,0].tolist()
    assert metadata["fps"] == 29.97


@pytest.mark.parametrize("size", [(240,320), (720,1280), (256,256)])
def test_native_grid_and_unchanged_pixel_budget(legacy, size):
    import torch
    processor = SimpleNamespace(video_processor=SimpleNamespace(patch_size=16,merge_size=2))
    frames = torch.ones(4,3,*size,dtype=torch.uint8)
    resized = adapter.resize_video(frames,processor,legacy)
    expected = legacy.smart_resize(*size,factor=32,min_pixels=100352,max_pixels=int(100352*1.05))
    assert tuple(resized.shape) == (4,3,*expected)
    assert all(n % 32 == 0 for n in resized.shape[-2:])
    assert resized.dtype == torch.float32 and resized.min() == resized.max() == 1


def test_vllm_payload_and_prompt_are_separate(legacy, monkeypatch):
    import torch
    from transformers.video_utils import VideoMetadata
    frames = torch.zeros(4,3,224,448)
    meta = dict(fps=29.97,frames_indices=[0,29,60,89],total_num_frames=90,video_backend="decord")
    monkeypatch.setattr(adapter,"decode_video",lambda *a:(frames,meta))
    monkeypatch.setattr(adapter,"resize_video",lambda *a:frames)
    rendered = []
    def template(msg,**kwargs):
        rendered.append(msg)
        return "<|vision_start|><|video_pad|><|vision_end|>" + msg[0]["content"][1]["text"]
    proc = SimpleNamespace(apply_chat_template=template)
    api = official("ceval2oss",{"0":"brush_hair: unused explanation"},"/PRIVATE","/POSE")
    batch = [messages({"path":"brush_hair/hidden-label.avi"},api)]
    value = adapter.multimodal_inputs(batch,proc,legacy)[0]
    assert "/PRIVATE" not in value["prompt"] and "hidden-label" not in value["prompt"]
    assert "unused explanation" not in value["prompt"]
    assert rendered[0][0]["content"][0] == {"type":"video"}
    assert value["mm_processor_kwargs"] == {"do_resize":False,"do_sample_frames":False}
    assert len(value["multi_modal_data"]["video"]) == 1
    video, metadata = value["multi_modal_data"]["video"][0]
    assert video is frames and metadata == meta
    # The exact dict shape accepted by vLLM 0.11's VideoMetadata constructor.
    assert VideoMetadata(**metadata).fps == 29.97


def test_preprocessing_config_is_part_of_resume_signature(tmp_path):
    import evaluate
    model = tmp_path / "model"
    model.mkdir()
    (model/"config.json").write_text(json.dumps({"model_type":"qwen3_vl"}))
    (model/"model.safetensors").touch()
    processor = model/"video_preprocessor_config.json"
    processor.write_text('{"patch_size":16}')
    spec = registry()["datasets"]["hmdb51-split1"]
    rows,actions,count = load_dataset(spec,1)
    selected = ("hmdb51-split1",spec,rows,actions,count,tmp_path,tmp_path)
    args = evaluate.parser().parse_args(["run","--model",str(model),"--run-name","test"])
    first = evaluate.signature_for(args,selected,evaluate.model_identity(model))
    assert first["model_input"]["resize_factor"] == 32
    processor.write_text('{"patch_size":14}')
    second = evaluate.signature_for(args,selected,evaluate.model_identity(model))
    assert first != second


@pytest.mark.parametrize("mode,gpus,limit", [("smoke","0","2"),("run","0,1,2,3,4,5,6,7",None)])
def test_launcher_exact_paths_and_smoke_is_separate(tmp_path, mode, gpus, limit):
    capture = tmp_path/"python"
    capture.write_text("#!"+sys.executable+"\nimport sys,json\nprint(json.dumps(sys.argv[1:]))\n")
    capture.chmod(0o755)
    env = dict(os.environ,EVAL_PYTHON=str(capture))
    env.pop("QWEN3_MODEL",None); env.pop("QWEN3_RESULTS_ROOT",None)
    result = subprocess.check_output(["bash",str(ROOT/"run_qwen3.sh"),mode],env=env,text=True)
    args = json.loads(result)
    assert args[args.index("--model")+1] == "/path/to/workspace/models/Qwen/Qwen3-VL-8B-Instruct"
    assert args[args.index("--gpus")+1] == gpus
    assert args[args.index("--table-dir")+1].endswith("VideoSTAR-Eval-results-v6-qwen3")
    if limit:
        assert args[args.index("--limit")+1] == limit
        assert "smoke" in args[args.index("--run-name")+1]
    else:
        assert "--limit" not in args


@pytest.mark.skipif(not os.environ.get("QWEN3_TEST_PROCESSOR"), reason="Set QWEN3_TEST_PROCESSOR to official local processor assets")
def test_real_video_and_official_qwen3_processor(tmp_path, legacy, monkeypatch):
    import av
    import numpy as np
    from transformers import AutoProcessor
    from transformers.video_utils import VideoMetadata
    from protocol import EVAL_SPLITS
    path = tmp_path/"private_video_filename.mp4"
    with av.open(str(path),"w") as container:
        stream = container.add_stream("mpeg4",rate=4)
        stream.width,stream.height,stream.pix_fmt = 96,64,"yuv420p"
        for i in range(16):
            frame = av.VideoFrame.from_ndarray(np.full((64,96,3),i*12,dtype=np.uint8),format="rgb24")
            for packet in stream.encode(frame): container.mux(packet)
        for packet in stream.encode(): container.mux(packet)
    monkeypatch.setattr(legacy,"get_video_reader_backend",lambda:"torchvision")
    sampled,meta = adapter.decode_video(str(path),legacy)
    original,_ = legacy._read_video_torchvision({"video":str(path)})
    assert sampled.equal(original)
    assert meta["fps"] == 4 and meta["total_num_frames"] == 16
    processor = AutoProcessor.from_pretrained(os.environ["QWEN3_TEST_PROCESSOR"],local_files_only=True)
    # All twelve real prompts, candidate spaces and official native chat template.
    for key in EVAL_SPLITS:
        spec = registry()["datasets"][key]
        _,classes,_ = load_dataset(spec,1)
        api = official(spec["entry"],classes,tmp_path,tmp_path)
        msg = messages({"path":path.name},api)
        value = adapter.multimodal_inputs([msg],processor,legacy)[0]
        assert path.name not in value["prompt"] and str(tmp_path) not in value["prompt"]
        frames,metadata = value["multi_modal_data"]["video"][0]
        inputs = processor(text=[value["prompt"]],videos=[frames],
            video_metadata=[VideoMetadata(**metadata)],return_tensors="pt",**value["mm_processor_kwargs"])
        assert inputs["input_ids"].shape[1] < 16384
        assert inputs["video_grid_thw"].tolist() == [[len(frames)//2,frames.shape[2]//16,frames.shape[3]//16]]
