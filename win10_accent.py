import winreg
from winreg import *

def get_accent_rgb():
	key = OpenKey(HKEY_CURRENT_USER, r'SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Accent', 0, KEY_ALL_ACCESS)
	val = QueryValueEx(key, "AccentColorMenu")
	rgbVal = intDwordColorToRGB(val[0])
	return rgbVal

print(get_accent_rgb())