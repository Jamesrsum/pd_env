#!/usr/bin/env python3
"""Fetch pinned public PDFs and extract their full text into PRIVATE storage.

One reviewer then uses docs/literature/AGENT_PROTOCOL.md to create original
method cards. Full copyrighted PDFs/text never enter the Git checkout.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request
REPO=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',required=True,type=Path)
    p.add_argument('--manifest',type=Path,default=REPO/'docs/literature/pdf_manifest.json')
    p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    root=a.root.resolve()
    if root==REPO or REPO in root.parents:raise ValueError('Full literature PDFs/text require storage outside Git')
    os.umask(0o077);root.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(a.manifest.read_text());results=[]
    for key,entry in manifest.items():
        if Path(key).name!=key:raise ValueError('Unsafe paper identifier')
        path=root/f'{key}.pdf'
        if not path.exists():
            if a.verify_only:raise FileNotFoundError(path)
            request=urllib.request.Request(entry['url'],headers={'User-Agent':'Mozilla/5.0 research-methods-archive'})
            with urllib.request.urlopen(request,timeout=60) as response:data=response.read()
            if not data.startswith(b'%PDF'):raise ValueError(f'{key}: source did not return a PDF')
            temporary=path.with_suffix('.part');temporary.write_bytes(data);temporary.replace(path)
        checksum=hashlib.sha256(path.read_bytes()).hexdigest()
        if checksum!=entry['sha256']:raise ValueError(f'{key}: PDF differs from reviewed source; do not silently refresh methodology')
        if not a.verify_only:
            binary=shutil.which('pdftotext')
            if not binary:raise RuntimeError('Install pdftotext to extract full PDFs')
            subprocess.run([binary,'-layout',str(path),str(root/f'{key}.txt')],check=True)
        results.append({'paper':key,'sha256_verified':True,'bytes':path.stat().st_size})
    print(json.dumps({'papers_verified':len(results),'private_archive':str(root),'results':results},indent=2))


if __name__=='__main__':main()
