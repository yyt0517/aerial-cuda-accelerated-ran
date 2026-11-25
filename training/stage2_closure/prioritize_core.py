"""Temporarily suspend this campaign's non-core native children, then resume.

Native and Python communicate over blocking Unix sockets with no read deadline.
Only the native child is suspended; all scientific clocks advance in logical TTI.
No trajectory, algorithm, source hash, seed, or measurement window is changed.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import time

def main(args):
    root=Path(args.source).resolve();protocol=json.loads((root/'protocol.json').read_text())
    def key(j):return f"{j['scene']}_r{j['rate']:g}_{j['arm']}_s{j['seed']}"
    allkeys={key(j) for j in protocol['jobs']}
    core={key(j) for j in protocol['jobs'] if j['rate']<=1 and j['arm'] in ('pfq','deficit','mlwdf_pf')}
    assert len(core)==60
    paused={};events=[];out=root.parent/'execution_priority.json';complete=False
    def record(kind,pid,path):
        events.append(dict(action=kind,pid=pid,trajectory=Path(path).name,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        temp=out.with_suffix('.tmp');temp.write_text(json.dumps(dict(completed=complete,scope='core first; suspend only native children with established blocking sockets',
            scientific_protocol_changed=False,events=events,paused_pids=paused),ensure_ascii=False,indent=2)+'\n');temp.replace(out)
        print(kind,pid,Path(path).name,flush=True)
    def terminate(_signum,_frame):raise SystemExit(0)
    signal.signal(signal.SIGTERM,terminate)
    try:
        while True:
            done={p.parent.name for p in root.glob('*/done.json')}
            if core<=done:complete=True;break
            progress_file=root/'progress_campaign.json'
            if progress_file.exists():
                failures=json.loads(progress_file.read_text())['failed']
                if any(key(f['job']) in core for f in failures):raise RuntimeError('A core trajectory failed; resume extensions for diagnosis')
            for entry in Path('/proc').iterdir():
                if not entry.name.isdigit():continue
                pid=int(entry.name)
                if pid in paused:continue
                try:
                    if (entry/'comm').read_text().strip()!='simulator':continue
                    cwd=Path(os.readlink(entry/'cwd'))
                    if cwd.parent!=root or cwd.name not in allkeys or cwd.name in core:continue
                    progress=cwd/'progress.json'
                    if not progress.exists() or json.loads(progress.read_text())['tti']<100:continue
                    os.kill(pid,signal.SIGSTOP);paused[pid]=str(cwd);record('SUSPEND_EXTENSION',pid,cwd)
                except (FileNotFoundError,ProcessLookupError,PermissionError):continue
            time.sleep(2)
    finally:
        for pid,path in list(paused.items()):
            try:
                if os.readlink(f'/proc/{pid}/cwd')==path:os.kill(pid,signal.SIGCONT)
            except (FileNotFoundError,ProcessLookupError,PermissionError):pass
            del paused[pid];record('RESUME_EXTENSION',pid,path)
        record('CORE_COMPLETE' if complete else 'CONTROLLER_STOPPED',os.getpid(),root)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);main(p.parse_args())
