"""Build inline previews from the curated rollout manifest."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import imageio_ffmpeg

root = Path(__file__).resolve().parents[2]
folder = root / 'docs/media/rollouts'
manifest_path = folder / (sys.argv[1] if len(sys.argv) > 1 else 'manifest.json')
manifest = json.loads(manifest_path.read_text())
for video in manifest['videos']:
    src = folder / video['file']
    dst = src.with_suffix('.gif')
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error',
                    '-i', str(src), '-filter_complex',
                    'fps=8,scale=320:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse',
                    '-loop', '0', str(dst)], check=True)
    video['preview'] = dict(file=dst.name, sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),
                            bytes=dst.stat().st_size, fps=8, width=320, loop=True)
manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
