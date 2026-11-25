"""Isolated scenario-only extension of the report's frozen native simulator."""
import json,hashlib,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
ARCHIVE=ROOT/'.worktrees/pfq-independent-reuse-actor'
FORMAL=ARCHIVE/'output/pfq_independent_reuse/formal'
OLD=FORMAL/'boundary08_steady_state_20261021_v1'
PHY=FORMAL/'boundary08_corrected_physics_20261020_v1'
OUT=ROOT/'deliverables/stage2_generalization_20261106/native'
BUILD=ARCHIVE/'build.corrected_uniform_v1/cuMAC/examples/multiCellSchedulerUeSelection'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((OLD/'native_build_manifest.json').read_text())
    assert sha(OLD/'simulator')==manifest['simulator_sha256']
    source=(OLD/'instrumented_main.cpp').read_text()
    needle='    trafficFlowSpecs = buildTrafficGranularityProfile(\n        trafficProfile, traffic_num_flows, packetSizeBytes, trafficArrivalRate);'
    assert source.count(needle)==1
    extension='''    const char* flowFile = std::getenv("CUMAC_GENERALIZATION_FLOW_CSV");
    if (flowFile && flowFile[0]) {
      std::ifstream input(flowFile);
      if (!input) throw std::invalid_argument("Cannot open per-UE traffic configuration");
      std::string line; std::getline(input,line);
      int expectedUe=0;
      while (std::getline(input,line)) {
        if (line.empty()) continue;
        int ue=-1,size=0; double rate=-1;
        if (std::sscanf(line.c_str(),"%d,%d,%lf",&ue,&size,&rate)!=3 || ue!=expectedUe || size<=0 || !std::isfinite(rate) || rate<=0)
          throw std::invalid_argument("Invalid per-UE traffic row");
        trafficFlowSpecs.push_back({static_cast<float>(size),static_cast<float>(rate),size<3000?"high_frequency_short":size>3000?"low_frequency_large":"homogeneous"});
        ++expectedUe;
      }
      if (expectedUe != traffic_num_flows) throw std::invalid_argument("Per-UE traffic row count mismatch");
    } else {
'''+needle+'\n    }'
    source=source.replace(needle,extension)
    (OUT/'main.cpp').write_text(source)
    network=(PHY/'native_network_source.cu').read_text()
    needle='''    const UePlacementConfig placementCfg = getUePlacementConfig();'''
    assert network.count(needle)==1
    network=network.replace(needle,needle+'''
    const char* mixedEnv = std::getenv("CUMAC_GENERALIZATION_MIXED_PLACEMENT");
    const bool mixedPlacement = mixedEnv && std::strcmp(mixedEnv,"1")==0;
    if (mixedPlacement && (netData->numCoorCell!=3 || netData->numActiveUesPerCell!=12 || placementCfg.mode!=UePlacementMode::CooperativeBoundary))
      throw std::runtime_error("Generalization placement requires the frozen 3 x 12 boundary setup");
    if (mixedPlacement) printf("Generalization placement: first six UEs per nominal cell uniform, last six boundary; strongest association unchanged\\n");
''')
    needle='''                } else if (placementCfg.mode == UePlacementMode::CooperativeBoundary) {'''
    assert network.count(needle)==1
    network=network.replace(needle,'''                } else if (placementCfg.mode == UePlacementMode::CooperativeBoundary && (!mixedPlacement || uIdx>=6)) {''')
    (OUT/'network.cu').write_text(network)
    (OUT/'network.h').write_bytes((PHY/'native_network_source.h').read_bytes())
    cm=list(manifest['compile_main']);cm[cm.index('-c')+1]=str(OUT/'main.cpp');cm[cm.index('-o')+1]=str(OUT/'main.o');cm.insert(1,'-I'+str(OUT))
    cn=json.loads((PHY/'native_network_build.json').read_text())['command'];cn[cn.index('-c')+1]=str(OUT/'network.cu');cn[cn.index('-o')+1]=str(OUT/'network.o')
    cn.insert(1,'-I'+str(OUT));cn.insert(2,'-I'+str(ARCHIVE/'cuMAC/examples'))
    commands=[cm,cn]
    for c in commands:subprocess.run(c,cwd=BUILD,check=True)
    link=list(manifest['link_command'])
    link=[str(OUT/'main.o') if x==str(OLD/'native_main.o') else str(OUT/'network.o') if x==str(PHY/'native_network.o') else x for x in link]
    link[link.index('-o')+1]=str(OUT/'simulator');subprocess.run(link,cwd=BUILD,check=True);commands.append(link)
    assert sha(OLD/'simulator')==manifest['simulator_sha256']
    assert sha(ARCHIVE/'build.corrected_uniform_v1/cuMAC/src/libcumac.so')==manifest['libcumac_sha256']
    out=dict(simulator_sha256=sha(OUT/'simulator'),parent_simulator_sha256=manifest['simulator_sha256'],libcumac_sha256=manifest['libcumac_sha256'],scope='Only mixed geometry branch and per-UE traffic input; old physics/scheduler/continuous channel unchanged',commands=commands,sources={str(p):sha(p) for p in [OUT/'main.cpp',OUT/'network.cu',OUT/'network.h',Path(__file__)]})
    (OUT/'build_manifest.json').write_text(json.dumps(out,indent=2)+'\n');print('BUILD COMPLETE',out['simulator_sha256'],flush=True)
if __name__=='__main__':main()
