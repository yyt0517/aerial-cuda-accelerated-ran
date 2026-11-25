"""Make completion-index links readable from both host and container mounts."""
import os,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parents[1]
EXP=ROOT/'deliverables/stage2_generalization_20261106'
changes=[]
while True:
    for link in (EXP/'campaign').glob('*'):
        if not link.is_symlink():continue
        target=os.readlink(link)
        if not target.startswith('/opt/nvidia/cuBB/'):continue
        host=ROOT/target[len('/opt/nvidia/cuBB/'):]
        relative=os.path.relpath(host,link.parent)
        temporary=link.with_name(link.name+'_relative_index_link')
        temporary.symlink_to(relative,target_is_directory=True);os.replace(temporary,link)
        changes.append(dict(index=str(link),container_target=target,relative_target=relative))
        (OUT/'核验/运行路径映射.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2)+'\n')
        print('PORTABLE INDEX',link.name,flush=True)
    if len(list((EXP/'campaign').glob('*/done.json')))==60:break
    time.sleep(10)
