import subprocess
from PIL import Image
import os
import sys

def set_background(pil_img):
	if pil_img.mode != 'RGB':
		pil_img = pil_img.convert('RGB')
	pil_img.save(os.path.expanduser('~/AppData/Roaming/Microsoft/Windows/Themes/TranscodedWallpaper'), format='PNG')
	subprocess.run(['rundll32.exe', 'user32.dll,', 'UpdatePerUserSystemParameters'])
	subprocess.run(['rundll32.exe', 'user32.dll,', 'UpdatePerUserSystemParameters'])

def set_background_file(filepath):
	set_background(Image.open(filepath))

if len(sys.argv) > 1:
	set_background_file(sys.argv[1])