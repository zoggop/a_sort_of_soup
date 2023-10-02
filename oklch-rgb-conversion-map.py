import numpy as np
import os
import coloraide

rgb2lch = np.zeros((256, 256, 256, 3), dtype=np.single)
cMax = 0
for ri in range(255):
	r = ri / 255
	print(ri, r)
	for gi in range(255):
		g = gi / 255
		for bi in range(255):
			b = bi / 255
			color = coloraide.Color('srgb', [r, g, b]).convert('oklch')
			if color.c > cMax:
				cMax = color.c
			l = max(0, min(1, color.l))
			c = max(0, min(0.32, color.c))
			h = max(0, min(360, color.h))
			rgb2lch[ri, gi, bi] = [l, c, h]
np.save(os.path.expanduser('~/rgb2lch.npy'), rgb2lch)
print(cMax, "maximum chroma")

# cMax = 0.322

lResolution = 300
cResolution = 300
lch2rgb = np.zeros((lResolution, cResolution, 361, 3), dtype=np.uint8)
lDivisor = lResolution - 1
cDivisor = (cResolution - 1) * (1 / cMax)
for li in range(lResolution):
	l = li / lDivisor
	print(li, l, "lightness")
	for ci in range(cResolution):
		c = ci / cDivisor
		# print(ci, c, "chroma")
		for h in range(361):
			color = coloraide.Color('oklch', [l, c, h]).convert('srgb')
			r = round(max(0, min(1, color.r)) * 255)
			g = round(max(0, min(1, color.g)) * 255)
			b = round(max(0, min(1, color.b)) * 255)
			lch2rgb[li, ci, h] = [r, g, b]
np.save(os.path.expanduser('~/lch2rgb.npy'), lch2rgb)
