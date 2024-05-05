# Project FOD

### NOTE
- [x] Python 3.10
- [x] Data training in Folder FOD
- [x] Weights Model in Folder Weights
- [x] Screenshoot in Folder Detect


### Clone Project
```
git clone https://github.com/ASNProject/project-fod.git
```

and intall requirements

```
pip install -r requirements.txt
```

### Run Project
1. Realtime camera
```
python detect2.py --weights best.pt --source 0
```
2. Single Image
```
python detect2.py --weights best.pt --source [name image].jpg
```
3. Video
```
python detect2.py --weights best.pt --source [name video].mp4
```
### Screenshot<br/>

Open in Detect Folder to show screenshot detection
