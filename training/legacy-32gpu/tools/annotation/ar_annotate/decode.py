"""Isolated video decoder: the parent can terminate a stuck codec safely."""
import argparse
import os


def decode_to_file(job):
    os.environ['FORCE_QWENVL_VIDEO_READER'] = 'decord'
    import torch
    from qwen_vl_utils.vision_process import fetch_video
    (frames, metadata), sample_fps = fetch_video(
        dict(video=job['video'], fps=job['fps'], max_frames=job['max_frames'],
             min_pixels=job['pixels'], max_pixels=job['pixels']),
        image_patch_size=job['patch'], return_video_sample_fps=True, return_video_metadata=True)
    torch.save(dict(frames=frames, metadata=metadata, sample_fps=float(sample_fps)), job['out'])


def worker(connection):
    """Persistent CPU process, retaining imports without inheriting a CUDA context."""
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    try:
        while True:
            job = connection.recv()
            if job is None:
                return
            try:
                decode_to_file(job)
                connection.send({'ok': True})
            except Exception as e:
                connection.send({'ok': False, 'error': f'{type(e).__name__}: {e}'})
    except EOFError:
        return
    finally:
        connection.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--video', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--patch', type=int, required=True)
    p.add_argument('--fps', type=float, required=True)
    p.add_argument('--max-frames', type=int, required=True)
    p.add_argument('--pixels', type=int, required=True)
    decode_to_file(vars(p.parse_args()))


if __name__ == '__main__':
    main()
