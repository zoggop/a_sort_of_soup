import numpy as np
import os
import coloraide
import random

# rgb2lch = np.zeros((256, 256, 256, 3), dtype=np.single)
# cMax = 0
# for ri in range(255):
# 	r = ri / 255
# 	print(ri, r)
# 	for gi in range(255):
# 		g = gi / 255
# 		for bi in range(255):
# 			b = bi / 255
# 			color = coloraide.Color('srgb', [r, g, b]).convert('oklch')
# 			if color.c > cMax:
# 				cMax = color.c
# 			l = max(0, min(1, color.l))
# 			c = max(0, min(0.32, color.c))
# 			h = max(0, min(360, color.h))
# 			rgb2lch[ri, gi, bi] = [l, c, h]
# np.save(os.path.expanduser('~/rgb2lch.npy'), rgb2lch)
# print(cMax, "maximum chroma")

cMax = 0.322

lResolution = 501
cResolution = 323
hResolution = 721
# create a list of all possible 8bit RGB colors
# seq = np.arange(256, dtype=np.uint8)
# rSeq, gSeq, bSeq = np.meshgrid(seq, seq, seq, indexing='ij', copy=False)
# RGBcheck = np.stack((rSeq, gSeq, bSeq), axis=-1).reshape(-1, 3)
# print(RGBcheck.shape, np.where(RGBcheck == [12, 13, 15]))
RGBcheck = np.zeros((256, 256, 256), dtype=np.bool_)
RGBtotal = 256 * 256 * 256
lch2rgb = np.zeros((lResolution, cResolution, hResolution, 3), dtype=np.uint8)
lDivisor = lResolution - 1
cDivisor = (cResolution - 1) / cMax
hDivisor = (hResolution - 1) / 360
lis = list(range(lResolution))
# for li in range(lResolution):
while len(lis) > 0:
	li = lis.pop(random.randrange(len(lis)))
	l = li / lDivisor
	RGBcount = np.count_nonzero(RGBcheck)
	lCount = lResolution - (len(lis) + 1)
	if lCount > 0:
		percentRGB = (RGBcount / RGBtotal) * 100
		percentLCH = (lCount / lResolution) * 100
		predictedRGBpercent = (percentRGB / percentLCH) * 100
		print("{} i {:.3f} L {:.1f}% LCH {:.1f}% RGB {:.0f}% RGB coverage predicted".format(li, l, percentLCH, percentRGB, predictedRGBpercent))
	else:
		print("{} i {:.3f}".format(li, l))
	for ci in range(cResolution):
		c = ci / cDivisor
		# print(ci, c, "chroma")
		for hi in range(hResolution):
			h = hi / hDivisor
			color = coloraide.Color('oklch', [l, c, h]).convert('srgb')
			r = round(max(0, min(1, color.r)) * 255)
			g = round(max(0, min(1, color.g)) * 255)
			b = round(max(0, min(1, color.b)) * 255)
			lch2rgb[li, ci, hi] = [r, g, b]
			RGBcheck[r, g, b] = True
np.save(os.path.expanduser('~/lch2rgb.npy'), lch2rgb)
