"""Update native Word fields and render PDF through local LibreOffice."""
import argparse
import sys
sys.path.insert(0,'/tmp/stage2_report_python')
from zipfile import ZipFile,ZIP_DEFLATED
from lxml import etree
from copy import deepcopy
from pathlib import Path
import time
import subprocess
import os
import uno
from com.sun.star.beans import PropertyValue

def prop(name,value):
    p=PropertyValue();p.Name=name;p.Value=value;return p

def main(args):
    runtime=Path('/tmp/report_generalization_runtime');runtime.mkdir(exist_ok=True)
    env=dict(os.environ,XDG_RUNTIME_DIR=str(runtime),GSETTINGS_BACKEND='memory',SAL_USE_VCLPLUGIN='svp')
    office=subprocess.Popen(['libreoffice','-env:UserInstallation=file:///tmp/report_generalization_lo_20261106','--headless','--accept=pipe,name=report_generalization_20261106;urp;StarOffice.ServiceManager','--norestore','--nodefault','--nofirststartwizard'],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    original_xml=ZipFile(Path(args.path)).read('word/document.xml')
    context=uno.getComponentContext()
    resolver=context.ServiceManager.createInstanceWithContext('com.sun.star.bridge.UnoUrlResolver',context)
    for attempt in range(100):
        try:
            remote=resolver.resolve('uno:pipe,name=report_generalization_20261106;urp;StarOffice.ComponentContext');break
        except Exception:
            if office.poll() is not None:raise RuntimeError(office.stderr.read().decode())
            time.sleep(.2)
    else:raise RuntimeError('Document renderer did not become ready')
    desktop=remote.ServiceManager.createInstanceWithContext('com.sun.star.frame.Desktop',remote)
    path=Path(args.path).resolve();url=uno.systemPathToFileUrl(str(path))
    doc=desktop.loadComponentFromURL(url,'_blank',0,(prop('Hidden',True),prop('UpdateDocMode',3)))
    try:
        doc.refresh();indexes=doc.getDocumentIndexes();print('indexes',indexes.getCount(),flush=True)
        for _ in range(2):
            for i in range(indexes.getCount()):indexes.getByIndex(i).update()
            doc.refresh()
        doc.storeAsURL(url,(prop('FilterName','Office Open XML Text'),prop('Overwrite',True)))
        if args.word_only:
            print(path,flush=True)
            return
        pdf=path.with_suffix('.pdf')
        doc.storeToURL(uno.systemPathToFileUrl(str(pdf)),(prop('FilterName','writer_pdf_Export'),prop('Overwrite',True)))
        print(pdf,flush=True)
    finally:
        doc.close(True)
        desktop.terminate()
        office.wait(timeout=15)
    # Keep every native Word formula exactly as supplied; LO otherwise rewrites OMML.
    ns={'m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
    original=etree.fromstring(original_xml).xpath('//m:oMath',namespaces=ns)
    with ZipFile(path) as z:entries={n:z.read(n) for n in z.namelist()}
    tree=etree.fromstring(entries['word/document.xml'])
    converted=tree.xpath('//m:oMath',namespaces=ns)
    assert len(original)==len(converted)
    for before,after in zip(original,converted):after.getparent().replace(after,deepcopy(before))
    entries['word/document.xml']=etree.tostring(tree,xml_declaration=True,encoding='UTF-8',standalone=True)
    with ZipFile(path,'w',ZIP_DEFLATED) as z:
        for n,data in entries.items():z.writestr(n,data)
    print('Preserved native Word formulas:',len(original),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('path');p.add_argument('--word-only',action='store_true');main(p.parse_args())
