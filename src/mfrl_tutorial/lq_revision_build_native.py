"""Build the optional local C backend with an already installed compiler.

No package installation or system configuration change is performed. NumPy is
the default backend and does not need this command.
"""
from pathlib import Path
import os
import platform
import shlex
import shutil
import subprocess
import sys
from .lq_revision_common import digest, write_json


def main():
    if sys.platform not in ('darwin', 'linux'):
        raise SystemExit('Native builder supports macOS/Linux; use the NumPy backend here')
    compiler = shlex.split(os.environ['CC']) if 'CC' in os.environ else [shutil.which('clang') or shutil.which('cc') or '']
    if not compiler[0]:
        raise SystemExit('No existing C compiler found; use NumPy. Nothing was installed.')
    here = Path(__file__).resolve().parent/'native'
    suffix = '.dylib' if sys.platform == 'darwin' else '.so'
    output = here/('rollout_fused'+suffix)
    flags = ['-arch', platform.machine(), '-dynamiclib'] if sys.platform == 'darwin' else ['-shared', '-fPIC']
    command = compiler+flags+['-O3', '-ffp-contract=off', str(here/'rollout_fused.c'), '-o', str(output), '-lm']
    subprocess.run(command, check=True, timeout=60)
    write_json(here/'BUILD.json', dict(command=command, compiler=subprocess.check_output(compiler+['--version'], text=True),
        architecture=platform.machine(), platform=platform.platform(), source_sha256=digest(here/'rollout_fused.c'),
        library_sha256=digest(output)))
    print('Built local optional backend: '+str(output))


if __name__ == '__main__':main()
