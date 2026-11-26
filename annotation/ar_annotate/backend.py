import hashlib
import importlib.metadata
import math
import tempfile
from pathlib import Path

from .common import ROOT, file_hash, read_json
from .decoder_pool import DecoderProcess


def model_identity(path):
    path = Path(path).resolve()
    config = read_json(path / 'config.json')
    if config.get('model_type') != 'qwen3_5' or not config.get('vision_config'):
        raise ValueError('Expected Qwen3.8 multimodal checkpoint with model_type=qwen3_5 and vision_config')
    weights = sorted(path.glob('*.safetensors'))
    if not weights:
        raise FileNotFoundError('No safetensors weights in ' + str(path))
    index = path / 'model.safetensors.index.json'
    if index.exists():
        required = set(read_json(index)['weight_map'].values())
        if required - {p.name for p in weights}:
            raise FileNotFoundError('Missing weight shards')
    return dict(path=str(path), config_sha256=file_hash(path / 'config.json'),
                auxiliaries={p.name: file_hash(p) for p in sorted(path.glob('*.json'))},
                templates={p.name: file_hash(p) for p in sorted(path.glob('*.jinja'))},
                weights=[dict(name=p.name, size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in weights])


def environment():
    return {name: importlib.metadata.version(name) for name in
            ('torch', 'torchvision', 'transformers', 'accelerate', 'qwen-vl-utils', 'decord')}


class QwenBackend:
    def __init__(self, cfg):
        import torch
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
        self.torch, self.cfg = torch, cfg
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for real annotation; CPU tests do not load the model')
        model_identity(cfg['model'])
        self.processor = AutoProcessor.from_pretrained(cfg['model'], local_files_only=True, trust_remote_code=False)
        if not getattr(self.processor, 'video_processor', None):
            raise ValueError('Checkpoint has no video processor')
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            cfg['model'], local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16, device_map='auto', attn_implementation='sdpa')
        self.model.eval()
        if any(str(v) in ('cpu', 'disk') for v in getattr(self.model, 'hf_device_map', {}).values()):
            raise RuntimeError('Model offloaded to CPU/disk; expose more GPU memory for this BF16 run')
        self.device = self.model.get_input_embeddings().weight.device
        self.patch = int(self.processor.video_processor.patch_size)
        self.decoder = DecoderProcess()

    def load_video(self, path):
        with tempfile.TemporaryDirectory(prefix='ar_decode_') as tmp:
            target = Path(tmp) / 'frames.pt'
            self.decoder.request(dict(video=path, out=str(target), patch=self.patch,
                fps=self.cfg['fps'], max_frames=self.cfg['max_frames'], pixels=self.cfg['pixels']),
                timeout=self.cfg['decode_timeout'])
            data = self.torch.load(target, map_location='cpu', weights_only=True)
        frames, metadata = data['frames'], data['metadata']
        if not 2 <= len(frames) <= self.cfg['max_frames']:
            raise ValueError('Decoded frame count outside configured range')
        fps = float(metadata['fps'])
        indices = [int(x) for x in metadata['frames_indices']]
        if not math.isfinite(fps) or fps <= 0 or len(indices) != len(frames):
            raise ValueError('Invalid video timing metadata')
        data['audit'] = dict(frame_indices=indices, timestamps=[i / fps for i in indices],
                             original_fps=fps, sample_fps=data['sample_fps'], shape=list(frames.shape),
                             frame_sha256=hashlib.sha256(frames.contiguous().numpy().tobytes()).hexdigest())
        return data

    def metrics(self):
        return {'gpu_peak_allocated_GiB': [round(self.torch.cuda.max_memory_allocated(i) / 2**30, 3)
                                          for i in range(self.torch.cuda.device_count())],
                'device_map': {k: str(v) for k, v in getattr(self.model, 'hf_device_map', {}).items()}}

    def close(self):
        self.decoder.close()

    def generate(self, video, question, seed):
        from transformers import set_seed
        set_seed(seed)
        # apply_chat_template only renders tokens here; no real path/class name enters text.
        messages = [{'role': 'user', 'content': [{'type': 'video', 'video': 'video.mp4'},
                                                {'type': 'text', 'text': question}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False, preserve_thinking=False)
        inputs = self.processor(text=[prompt], videos=[video['frames']],
                    video_metadata=[video['metadata']], do_resize=False, do_sample_frames=False,
                    add_special_tokens=False, return_tensors='pt').to(self.device)
        n = inputs['input_ids'].shape[-1]
        if n + self.cfg['max_new_tokens'] > self.cfg['max_context']:
            raise ValueError('Input exceeds configured context budget; no silent truncation')
        with self.torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=self.cfg['max_new_tokens'],
                        do_sample=True, temperature=self.cfg.get('temperature', 0.7),
                        top_p=self.cfg.get('top_p', 0.8), top_k=self.cfg.get('top_k', 20),
                        repetition_penalty=1.0, use_cache=True)
        new = outputs[0, n:]
        raw = self.processor.decode(new, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if len(new) >= self.cfg['max_new_tokens']:
            raise ValueError('Generation hit token limit; refusing possibly truncated annotation')
        return raw
