"""HTTPS range downloader; validates each byte range, retries without truncation."""
import argparse,concurrent.futures,os,time
from pathlib import Path
import requests

def download(url,dest,workers=16,chunk_mb=16):
    dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    r=requests.get(url,headers={'Range':'bytes=0-0'},stream=True,timeout=(20,60));r.raise_for_status()
    if r.status_code!=206 or not r.headers.get('Content-Range','').startswith('bytes 0-0/'):
        raise RuntimeError('Server does not support validated range requests')
    size=int(r.headers['Content-Range'].split('/')[-1]);r.close()
    fd=os.open(dest,os.O_CREAT|os.O_TRUNC|os.O_WRONLY,0o600);os.ftruncate(fd,size)
    chunk=chunk_mb*1024*1024
    def part(start):
        end=min(start+chunk,size)-1
        for attempt in range(5):
            try:
                with requests.get(url,headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(20,60)) as r:
                    r.raise_for_status()
                    if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}':
                        raise RuntimeError('Incorrect range response')
                    offset=start
                    for data in r.iter_content(256*1024):
                        if offset+len(data)>end+1:raise RuntimeError('Too many bytes')
                        written=os.pwrite(fd,data,offset)
                        if written!=len(data):raise RuntimeError('Short write')
                        offset+=written
                    if offset!=end+1:raise RuntimeError('Incomplete range')
                return end-start+1
            except Exception:
                if attempt==4:raise
                time.sleep(min(2**attempt,10))
    try:
        done=0
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            for result in ex.map(part,range(0,size,chunk)):
                done+=result;print(f'{dest.name}: {done}/{size}',flush=True)
        os.fsync(fd)
    finally:os.close(fd)
    return size

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('url');p.add_argument('dest');p.add_argument('--workers',type=int,default=16);a=p.parse_args();download(a.url,a.dest,a.workers)
