```bash
cd ~/Bureau/
recordmydesktop --width 600 --height 420 -o ~/Bureau/test.ogv
mplayer -ao null test.ogv  -vo jpeg:outdir=images
convert ~/Bureau/images/* test.gif
```

