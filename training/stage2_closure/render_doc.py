"""Update native Word fields and render PDF through local LibreOffice."""
import argparse
from pathlib import Path
import time
import uno
from com.sun.star.beans import PropertyValue

def prop(name,value):
    p=PropertyValue();p.Name=name;p.Value=value;return p

def main(args):
    context=uno.getComponentContext()
    resolver=context.ServiceManager.createInstanceWithContext('com.sun.star.bridge.UnoUrlResolver',context)
    remote=resolver.resolve('uno:pipe,name=stage2_doc_render;urp;StarOffice.ComponentContext')
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
    finally:doc.close(True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('path');p.add_argument('--word-only',action='store_true');main(p.parse_args())
