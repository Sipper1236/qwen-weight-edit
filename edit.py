"""Reproducible, bounded-memory structural pruning; originals are read-only."""
import hashlib,json,math,re,shutil,struct
from pathlib import Path
src=Path('source'); dst=Path('output/pruned-28'); dst.mkdir(parents=True,exist_ok=True)
def digest(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
p=src/'model.safetensors'
expected=next(x['lfs']['sha256'] for x in json.load(open('source-metadata.json'))['siblings'] if x['rfilename']==p.name)
assert digest(p)==expected,'Source SHA mismatch'
with p.open('rb') as f:
 n=struct.unpack('<Q',f.read(8))[0]; header=json.loads(f.read(n)); base=8+n
removed=set(range(24,28)); mapping={old:new for new,old in enumerate(i for i in range(32) if i not in removed)}
new={}; copies=[]; offset=0; params=0
for key,v in header.items():
 if key=='__metadata__':new[key]=v;continue
 match=re.match(r'model.language_model.layers\.(\d+)\.',key)
 target=key
 if match:
  old=int(match[1])
  if old in removed:continue
  target=key[:match.start(1)]+str(mapping[old])+key[match.end(1):]
 a,b=v['data_offsets']; size=b-a
 assert size==math.prod(v['shape'])*{'BF16':2,'F32':4,'F16':2}[v['dtype']]
 new[target]={**v,'data_offsets':[offset,offset+size]};copies.append((key,target,a,size));offset+=size;params+=math.prod(v['shape'])
encoded=json.dumps(new,separators=(',',':')).encode();encoded+=b' '*((-len(encoded))%8)
out=dst/'model.safetensors'; tmp=out.with_suffix('.part'); hashes={}
with p.open('rb') as inp,tmp.open('wb') as f:
 f.write(struct.pack('<Q',len(encoded)));f.write(encoded)
 for key,target,a,size in copies:
  inp.seek(base+a);h=hashlib.sha256()
  while size:
   block=inp.read(min(size,4*1024*1024));assert block;f.write(block);h.update(block);size-=len(block)
  hashes[target]=h.hexdigest()
tmp.rename(out)
for file in src.iterdir():
 if file.is_file() and file.suffix!='.safetensors':shutil.copy2(file,dst/file.name)
cfg=json.load(open(src/'config.json'));cfg['text_config']['num_hidden_layers']=28;cfg['text_config']['layer_types']=[t for i,t in enumerate(cfg['text_config']['layer_types']) if i not in removed]
(dst/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
with out.open('rb') as f:
 n=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(n));start=8+n
 for key,v in h.items():
  if key=='__metadata__':continue
  a,b=v['data_offsets'];f.seek(start+a);d=hashlib.sha256();remaining=b-a
  while remaining:
   block=f.read(min(remaining,4*1024*1024));assert block;d.update(block);remaining-=len(block)
  assert d.hexdigest()==hashes[key],key
 assert out.stat().st_size==start+offset
assert sorted({int(re.search(r'layers\.(\d+)',k)[1]) for k in new if k.startswith('model.language_model.layers.')})==list(range(28))
report=dict(source_sha256=expected,output_sha256=digest(out),removed_layers=sorted(removed),layer_mapping=mapping,parameters=params,bytes=out.stat().st_size,removed_bytes=p.stat().st_size-out.stat().st_size,tensors_verified=len(copies),inference_tested=False)
(dst/'edit-manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
