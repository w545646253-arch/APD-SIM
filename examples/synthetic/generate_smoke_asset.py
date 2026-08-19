from pathlib import Path
import hashlib, json, numpy as np
root=Path(__file__).resolve().parent
y,x=np.mgrid[-1:1:64j,-1:1:64j]
gt=(np.exp(-((x-0.25)**2+(y+0.1)**2)/0.05)+0.6*np.exp(-((x+0.3)**2+(y-0.25)**2)/0.02)).astype(np.float32)
gt/=gt.max(); path=root/'smoke_input.npz'; np.savez_compressed(path,gt=gt)
receipt={'generator':'generate_smoke_asset.py','shape':[64,64],'dtype':'float32','array_sha256':hashlib.sha256(gt.tobytes()).hexdigest(),'restricted_data':False}
(root/'smoke_asset_receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n',encoding='utf-8')
