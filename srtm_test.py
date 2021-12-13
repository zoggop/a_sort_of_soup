import os
from numpy import gradient
from numpy import pi
from numpy import arctan
from numpy import arctan2
from numpy import sin
from numpy import cos
from numpy import sqrt
from numpy import zeros
from numpy import uint8
import numpy as np
from PIL import Image
import math
from zipfile import ZipFile
import coloraide
from blend_modes import multiply, overlay, soft_light, hard_light, darken_only
import catacomb
from getpass import getpass
import requests
from io import BytesIO
import random
import sys
import json
from cv2 import resize, INTER_CUBIC
from perceptual_hues_lavg import perceptualHues
from skimage.restoration import denoise_nl_means, denoise_tv_chambolle

degreesPerTheta = 90 / (math.pi / 2)
maxChroma = 134
tileFilter = None # 'water' for only tiles with some water, 'land' for only tiles without water, None for no filter

CurrentGrade = None

def asInt(intStr):
	try:
		return int(intStr)
	except:
		return None

def latitudeToZ(latitude):
	theta = latitude / degreesPerTheta
	return math.sin(theta)

def uniformlyRandomIntLatLon(minLat, maxLat):
	minZ, maxZ = latitudeToZ(minLat), latitudeToZ(maxLat)
	# https://www.cs.cmu.edu/~mws/rpos.html
	z = random.randint(int(1000 * minZ), int(1000 * maxZ)) / 1000
	lat = math.asin(z) * degreesPerTheta
	lon = random.randint(-180, 180)
	return int(lat), lon

def perceptuallyUniformRandomHue():
	return random.choice(perceptualHues)

def angleDist(a, b):
	return abs(((b - a) + 180) % 360 - 180)

def lch_to_rgb(lightness, chroma, hue):
	c = coloraide.Color('lch-d65', [lightness, chroma, hue]).convert('srgb')
	if c.in_gamut():
		return c
	return None

def rgb_to_lch(red, green, blue):
	c = coloraide.Color('srgb', [red/255, green/255, blue/255]).convert('lch-d65')
	if c.in_gamut():
		return c
	return None

def highestChromaColor(lightness, hue):
	chromaStep = 10
	if maxChroma < 10:
		chromaStep = 1
	chroma = maxChroma
	iteration = 0
	while iteration < 45:
		c = lch_to_rgb(lightness, chroma, hue)
		if not c is None:
			if chromaStep == 0.01 or maxChroma == 0:
				return c
			else:
				chroma += chromaStep
				chromaStep /= 10
				chroma -= chromaStep
		chroma = max(0, chroma - chromaStep)
		iteration += 1
	print(chromaStep, lightness, chroma, hue, iteration)

def huesDeltaE(hueA, hueB):
	a = lch_to_rgb(57, 32, hueA)
	b = lch_to_rgb(57, 32, hueB)
	return a.delta_e(b, method='2000')

def gradeFunc(v):
	return CurrentGrade[v]

def colorizeWithInterpolation(bwImage, interpolation):
	global CurrentGrade
	if bwImage.mode == 'L':
		bits = 8
	elif bwImage.mode == 'I':
		bits = 16
	numColors = 2 ** bits
	divisor = numColors - 1
	redGrade = [int(interpolation(l/divisor).red * 255) for l in range(numColors)]
	greenGrade = [int(interpolation(l/divisor).green * 255) for l in range(numColors)]
	blueGrade = [int(interpolation(l/divisor).blue * 255) for l in range(numColors)]
	CurrentGrade = redGrade
	redImage = Image.eval(bwImage, gradeFunc)
	CurrentGrade = greenGrade
	greenImage = Image.eval(bwImage, gradeFunc)
	CurrentGrade = blueGrade
	blueImage = Image.eval(bwImage, gradeFunc)
	return Image.merge('RGB', (redImage, greenImage, blueImage))

def rotateImage(img, rotation):
	if rotation == 1:
		return img.transpose(Image.ROTATE_90)
	elif rotation == 2:
		return img.transpose(Image.ROTATE_180)
	elif rotation == 3:
		return img.transpose(Image.ROTATE_270)
	else:
		return img

def image_histogram_equalization(image, number_bins=65536):
	# from http://www.janeriksolem.net/histogram-equalization-with-python-and.html

	# get image histogram
	image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
	cdf = image_histogram.cumsum() # cumulative distribution function
	normal_mult = (number_bins - 1) / (cdf[-1] - cdf[0])
	cdf = (cdf - cdf[0]) * normal_mult # normalize
	# use linear interpolation of cdf to find new pixel values
	image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

	return image_equalized.reshape(image.shape)

def hillshadePreparations(array):
	x, y = gradient(array)
	slope = pi/2. - arctan(sqrt(x*x + y*y))
	aspect = arctan2(-x, y)
	return slope, aspect

def hillshade(array, azimuth, angle_altitude, slope=None, aspect=None):
	azimuth = 360.0 - azimuth
	if slope is None:
		x, y = gradient(array)
		slope = pi/2. - arctan(sqrt(x*x + y*y))
	if aspect is None:
		aspect = arctan2(-x, y)
	azimuthrad = azimuth*pi / 180.
	altituderad = angle_altitude*pi / 180.
	shaded = sin(altituderad) * sin(slope)\
	 + cos(altituderad) * cos(slope)\
	 * cos((azimuthrad - pi/2.) - aspect)
	return 255*((shaded + 1)/2)

def autocontrast(arr, maxValue):
	mult = maxValue / (arr.max() - arr.min())
	return (arr - arr.min()) * mult

def autocontrastedUint8(arr):
	return autocontrast(arr, 255).astype(np.uint8)

def autocontrastedUint16(arr):
	return autocontrast(arr, 65535).astype(np.uint16)

def skewMedianToCenter(arr):
	center = (arr.min() + arr.max()) / 2
	med = np.median(arr)
	mult = (med / center)
	return arr * (abs(arr / med) * mult)

def measureLatLonInMeters(lat1, lon1, lat2, lon2):
	R = 6378.137 # Radius of earth in KM
	dLat = lat2 * math.pi / 180 - lat1 * math.pi / 180
	dLon = lon2 * math.pi / 180 - lon1 * math.pi / 180
	a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(lat1 * math.pi / 180) * math.cos(lat2 * math.pi / 180) * math.sin(dLon/2) * math.sin(dLon/2)
	c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
	d = R * c
	return d * 1000

def getEOSDISlogin():
	comb = catacomb.Catacomb(os.path.expanduser('~/color_out_of_earth'), 'uR7UtrczNUM0FnzR8X')
	username = comb.decrypt('eosdis_username')
	if username == None:
		username = input('        EOSDIS username: ')
		password, password2 = 'unlike', 'theother'
		attempts = 0
		while password != password2:
			if attempts > 0:
				print('Passwords DO NOT match, please retry.')
			password = getpass('        EOSDIS password: ')
			password2 = getpass('Reenter EOSDIS password: ')
			attempts += 1
		comb.encrypt('eosdis_username', username)
		comb.encrypt(username, password)
	else:
		password = comb.decrypt(username)
	return username, password

def downloadResponseWithStatus(response):
	totalkB = int(int(response.headers.get('Content-Length')) / 1024)
	kB = 0
	content = b''
	for chunk in response.iter_content(chunk_size=1024*10):
		kB = min(kB + 10, totalkB)
		percent = int((kB / totalkB) * 100)
		sys.stdout.write("\r{}% ({} / {} kB)".format(percent, kB, totalkB))
		sys.stdout.flush()
		content += chunk
	print(" ")
	return content

def filterTile(locationCode, tileFilter):
	if tileFilter is None:
		return True
	previewURL = "https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11/{}.SRTMGL1.2.jpg".format(locationCode)
	print("getting preview of", locationCode)
	response = requests.get(previewURL)
	if response.status_code == 200:
		lengthBytes = int(response.headers.get('Content-Length'))
		print(lengthBytes / 1024)
		if lengthBytes < 400 * 1024:
			# this tile has water
			if tileFilter == 'water':
				return True
			elif tileFilter == 'land':
				return False
		else:
			return False
	else:
		print("couldn't get preview")
		return False

def randomSRTMlocation():
	# locCodes = {}
	# https://dwtkns.com/srtm30m/srtm30m_bounding_boxes.json
	# with open("srtm30m_bounding_boxes.json", "r") as read_file:
	#     collection = json.load(read_file)
	#     for f in collection.get('features'):
	#         locCode = f.get('properties').get('dataFile').split('.')[0]
	#         locCodes[locCode] = True
	# with open('tile_list.json', 'w') as write_file:
	#     json.dump(locCodes, write_file, indent='')

	# load tile list
	with open("tile_list.json", "r") as read_file:
		locCodes = json.load(read_file)

	test = None
	attempt = 0
	while not test:
		if attempt == 0 and len(sys.argv) > 2:
			latitude, longitude = int(sys.argv[1]), int(sys.argv[2])
		else:
			latitude, longitude = uniformlyRandomIntLatLon(-56, 59)
		if latitude >= 0:
			latPrefix = 'N'
		else:
			latPrefix = 'S'
		if longitude >= 0:
			lonPrefix = 'E'
		else:
			lonPrefix = 'W'
		latString = latPrefix + '{0:02d}'.format(abs(latitude))
		lonString = lonPrefix + '{0:03d}'.format(abs(longitude))
		locationCode = latString + lonString
		test = locCodes.get(locationCode) and filterTile(locationCode, tileFilter)
		attempt += 1
	print(locationCode)
	return locationCode, latitude, longitude

def downloadHGT(url, un, pw):
	# print(url)
	response = requests.get(url, auth = requests.auth.HTTPBasicAuth(un, pw), headers = {'user-agent': 'Firefox'})
	print("redirected")
	response2 = requests.get(response.url, auth = requests.auth.HTTPBasicAuth(un, pw), headers = {'user-agent': 'Firefox'}, stream=True)
	return downloadResponseWithStatus(response2)

zip_file_name = None
if len(sys.argv) > 1 and sys.argv[1] == 'previous' and os.path.exists(os.path.expanduser('~/color_out_of_earth/previous.txt')):
	in_txt_file = open(os.path.expanduser('~/color_out_of_earth/previous.txt'), "r")
	lines = in_txt_file.readlines()
	locationCode, latitude, longitude = lines[0].strip(), int(lines[1].strip()), int(lines[2].strip())
	print(locationCode)
	zip_file_name = os.path.expanduser('~/color_out_of_earth/{}.SRTMGL1.hgt.zip'.format(locationCode))
if zip_file_name is None:
	username, password = getEOSDISlogin()
	locationCode, latitude, longitude = randomSRTMlocation()
	zip_data = downloadHGT('http://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11/{}.SRTMGL1.hgt.zip'.format(locationCode), username, password)

# from 42 N to 43 N, 123 W to 122 W

xMeters = measureLatLonInMeters(latitude + 0.5, longitude, latitude + 0.5, longitude + 1)
yMeters = measureLatLonInMeters(latitude, longitude + 0.5, latitude + 1, longitude + 0.5)
yMult = yMeters / xMeters
print("y-axis multiplier:", yMult)

screenWidth = 1920
screenHeight = 1080
rotation = random.randint(0, 3)
if rotation == 1 or rotation == 3:
	cropWidth = screenHeight
	cropHeight = screenWidth
else:
	cropWidth = screenWidth
	cropHeight = screenHeight
print("rotation:", rotation, cropWidth, cropHeight)

with ZipFile(zip_file_name or BytesIO(zip_data), 'r') as zip:
	with zip.open('{}.hgt'.format(locationCode), mode='r') as hgtFile:
		raw = hgtFile.read()
		siz = len(raw)
		dim = int(math.sqrt(siz/2))
		arr = np.frombuffer(raw, np.dtype('>i2'), dim*dim).reshape((dim, dim))

print(arr.shape)
metersPerPixel = xMeters / arr.shape[1]
print("meters per pixel:", metersPerPixel)
print("{:,.0f} m by {:,.0f} m".format(xMeters, yMeters))
print("from {:,} m to {:,} m".format(arr.min(), arr.max()))

# Image.fromarray((arr-arr.min()).astype(np.uint16)).save(os.path.expanduser('~/color_out_of_earth/original_elevation.png'))

# calculate where a crop won't just be an image of flat water
nonZeroYs, nonZeroXs = np.nonzero(arr)
print(nonZeroXs[0], nonZeroXs[-1], nonZeroYs[0], nonZeroYs[-1])
heightBecomeCrop = round(cropHeight / yMult)
acceptableOceanHeight = int(heightBecomeCrop / 2)
acceptableOceanWidth = int(cropWidth / 2)
xMax = arr.shape[1] - cropWidth
yMax = arr.shape[0] - heightBecomeCrop
xMax = max(0, min(nonZeroXs[-1] - acceptableOceanWidth, xMax))
yMax = max(0, min(nonZeroYs[-1] - acceptableOceanHeight, yMax))
xMin = max(0, min(nonZeroXs[0] - acceptableOceanWidth, xMax))
yMin = max(0, min(nonZeroYs[0] - acceptableOceanHeight, yMax))
print(xMin, xMax, yMin, yMax)

# get a random crop
if xMax > xMin:
	x = random.randint(xMin, xMax)
else:
	x = xMin
if yMax > yMin:
	y = random.randint(yMin, yMax)
else:
	y = yMin
x2 = x + cropWidth
y2 = y + heightBecomeCrop
arr = arr[y:y2, x:x2]
print("cropped", arr.shape, arr.min(), arr.max())

# stretch and rotate elevation data
# arr = arr.astype(np.int16)
arr = arr.astype(np.single)
print("astype", arr.shape, arr.min(), arr.max())
arr = resize(arr, dsize=(cropWidth, cropHeight), interpolation=INTER_CUBIC)
print("resize", arr.shape, arr.min(), arr.max())
if rotation > 0:
	arr = np.rot90(arr, k=rotation)
print("rotation", arr.shape, arr.min(), arr.max())

# clip negative bits if this crop contains sea level
if 0 in arr and arr.min() < 0:
	arr = np.clip(arr, 0, None)
	print(np.count_nonzero(arr<0))
	print("clipped to ocean", arr.shape, arr.min(), arr.max())

# compute slope to find water (flat areas)
gy, gx = gradient(arr)
slope = sqrt((gx*gx) + (gy*gy))
# Image.fromarray(autocontrastedUint8(slope)).save(os.path.expanduser('~/color_out_of_earth/slope.png'))
possible = (slope.shape[0] * slope.shape[1])
hasWater = np.count_nonzero(slope==0) > possible * 0.1
print("slope from {} to {}, {:,} flat pixels of {:,} possible".format(slope.min(), slope.max(), np.count_nonzero(slope==0), possible))
print("water?", hasWater)

# convert equalized elevation data to 256-color grayscale image
arr_eq = image_histogram_equalization(arr, 256)
print("equalized", arr_eq.shape, arr_eq.min(), arr_eq.max())
el_img = Image.fromarray(arr_eq.astype(np.uint8))
# el_img = Image.fromarray(autocontrastedUint8(arr))
print(el_img.mode, len(el_img.getcolors()))
print(el_img.size)

# pick hues
ah = perceptuallyUniformRandomHue()
if len(sys.argv) > 1:
	# get a hue from the command line if possible
	for arg in sys.argv[1:]:
		argHue = asInt(arg)
		if not argHue is None:
			ah = argHue
bh = ah
while huesDeltaE(ah, bh) < 20 or huesDeltaE(ah, bh) > 40:
	bh = perceptuallyUniformRandomHue()
ch = bh
while huesDeltaE(ch, ah) < 20 or huesDeltaE(ch, bh) < 20 or (huesDeltaE(ch, ah) > 40 and huesDeltaE(ch, bh) > 40):
	ch = perceptuallyUniformRandomHue()
print('hues', ah, bh, ch)
# pick lightnesses
darkMidLight = [random.randint(5,25), random.randint(40, 60), random.randint(75,95)]
lOrders = [
	[0, 1, 2],
	[2, 0, 1],
	[1, 0, 2],
	[0, 2, 1]]
lOrder = random.choice(lOrders)
ls = [darkMidLight[lOrder[0]], darkMidLight[lOrder[1]], darkMidLight[lOrder[2]]]
print('lightnesses', *ls)
# create gradient
a = highestChromaColor(ls[0], ah)
b = highestChromaColor(ls[1], bh)
c = highestChromaColor(ls[2], ch)
allSteps = a.steps([b, c], steps=256, space='lch-d65')
highChromaSteps = []
for col in allSteps:
	lch = col.convert('lch-d65')
	highChromaSteps.append(highestChromaColor(lch.l, lch.h))
i = highChromaSteps[0].interpolate(highChromaSteps[1:], space='lch-d65')
# colorize elevation data
color_el_img = colorizeWithInterpolation(el_img, i)

# create hillshade
# arrForShade = denoise_nl_means(arr, patch_size=11, patch_distance=21, h=2, fast_mode=True, preserve_range=True)
arrForShade = arr / metersPerPixel # so that the height map's vertical units are the same as its horizontal units
arrForShade = denoise_tv_chambolle(arrForShade, weight=0.05)
lightHue = highChromaSteps[-1].convert('lch-d65').h
print("for shade", arrForShade.shape, arrForShade.min(), arrForShade.max())
shades = [
	[350, 70, 0.9],
	[15, 60, 0.7],
	[270, 55, 1]
]
slopeForShade, aspectForShade = hillshadePreparations(arrForShade)
hsSum = None
for shade in shades:
	hs = hillshade(arrForShade, shade[0], shade[1], slopeForShade, aspectForShade) * shade[2]
	# Image.fromarray(hs.astype(np.uint8)).save(os.path.expanduser('~/color_out_of_earth/hs-{}-{}-{}.png'.format(*shade)))
	if hsSum is None:
		hsSum = hs
	else:
		hsSum += hs
hs = autocontrast(hsSum, 255)
print(hs.min(), np.median(hs), hs.max())
hs = hs + (127 - np.median(hs)) # linearly center median
# print(np.count_nonzero(hs < 0), "pixels below 0")
if hs.min() < 0:
	# compress negative values
	widthBelow = 127 - hs.min()
	hs = np.where(hs < 128, hs - (hs.min() * ((widthBelow - (hs - hs.min())) / widthBelow)), hs)
# print(np.count_nonzero(hs == 0), "pixels at 0")
print(hs.min(), np.median(hs), hs.max())
hs_img = Image.fromarray(hs.astype(np.uint8))
print(hs_img.mode, len(hs_img.getcolors()))

# blend colorized elevation with hillshade
color_el_arr = np.array(color_el_img.convert('RGBA'))
hs_arr = np.array(hs_img.convert('RGBA'))
blended_float = overlay(color_el_arr.astype(float), hs_arr.astype(float), 1)
# blended_float = overlay(hs_arr.astype(float), color_el_arr.astype(float), 1)
blended_arr = np.uint8(blended_float)
blended_img = Image.fromarray(blended_arr)

# save images
# Image.fromarray(autocontrastedUint16(arr)).save(os.path.expanduser('~/color_out_of_earth/el16_img.png'))
# el_img.save(os.path.expanduser('~/color_out_of_earth/el_img.png'))
hs_img.save(os.path.expanduser('~/color_out_of_earth/hs_img.png'))
# color_hs_img.save(os.path.expanduser('~/color_out_of_earth/color_hs_img.png'))
color_el_img.save(os.path.expanduser('~/color_out_of_earth/color_el_img.png'))
blended_img.save(os.path.expanduser('~/color_out_of_earth/blended_img.png'))

print("saved images")

if zip_file_name is None and not zip_data is None:
	# delete previous zip file
	if os.path.exists(os.path.expanduser('~/color_out_of_earth/previous.txt')):
		in_txt_file = open(os.path.expanduser('~/color_out_of_earth/previous.txt'), "r")
		lines = in_txt_file.readlines()
		prevLocCode, prevLat, prevLon = lines[0].strip(), int(lines[1].strip()), int(lines[2].strip())
		prevZipFilepath = os.path.expanduser('~/color_out_of_earth/{}.SRTMGL1.hgt.zip'.format(prevLocCode))
		if os.path.exists(prevZipFilepath):
			os.remove(prevZipFilepath)
			# print("deleted", prevZipFilepath)
	# save zip file
	out_file = open(os.path.expanduser('~/color_out_of_earth/{}.SRTMGL1.hgt.zip'.format(locationCode)), "wb")
	out_file.write(zip_data)
	out_file.close()
	out_txt_file = open(os.path.expanduser('~/color_out_of_earth/previous.txt'), "w")
	out_txt_file.write("{}\n{}\n{}".format(locationCode, latitude, longitude))
	out_txt_file.close()