import os
import argparse
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
from blend_modes import overlay #, multiply, soft_light, hard_light, darken_only
from getpass import getpass
import requests
from io import BytesIO
import random
import sys
import json
from cv2 import resize, INTER_CUBIC
# from skimage.restoration import denoise_nl_means, denoise_tv_chambolle
import skimage.filters.rank as rank
from skimage.morphology import disk, ball
from scipy.ndimage import label, generate_binary_structure
from bs4 import BeautifulSoup

# local modules
import catacomb
from perceptual_hues_lavg import perceptualHues

CurrentGrade = None

def asInt(intStr):
	try:
		return int(intStr)
	except:
		return None

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

def highestChromaColor(lightness, hue, maxChroma=134):
	chromaStep = 10
	if maxChroma < 10:
		chromaStep = 1
	chroma = maxChroma
	iteration = 0
	while iteration < 45:
		c = lch_to_rgb(lightness, chroma, hue)
		if not c is None:
			if chromaStep == 0.01 or maxChroma == 0 or iteration == 0:
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

def slopeOfArray(array):
	gy, gx = gradient(array)
	return sqrt(gy*gy + gx*gx)

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

def autocontrastedBool(arr):
	return autocontrast(arr, 1).astype(bool)

def autocontrastedUint8(arr):
	return autocontrast(arr, 255).astype(np.uint8)

def autocontrastedUint16(arr):
	return autocontrast(arr, 65535).astype(np.uint16)

def autocontrastedSingle(arr):
	return (autocontrast(arr, 2) - 1).astype(np.single)

def decontrast(arr, minValue, maxValue):
	mult = (maxValue - minValue) / (arr.max() - arr.min())
	return minValue + (arr * mult)

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
	totalkB = None
	if response.headers and response.headers.get('Content-Length'):
		totalkB = int(int(response.headers.get('Content-Length')) / 1024)
	kB = 0
	content = b''
	for chunk in response.iter_content(chunk_size=1024*10):
		kB = kB + 10
		if not totalkB is None:
			kB = min(kB, totalkB)
			percent = int((kB / totalkB) * 100)
			sys.stdout.write("\r{}% ({} / {} kB)".format(percent, kB, totalkB))
		else:
			sys.stdout.write("{} kB".format(kB))
		sys.stdout.flush()
		content += chunk
	print(" ")
	return content

def parseLocationCode(locationCode):
	NS = locationCode[0]
	latitude = int(locationCode[1:3])
	EW = locationCode[3]
	longitude = int(locationCode[4:7])
	if NS == 'S':
		latitude = 0 - latitude
	if EW == 'W':
		longitude = 0 - longitude
	return latitude, longitude

def latLonToLocationCode(latitude, longitude):
	latitude, longitude = math.floor(latitude), math.floor(longitude)
	NS = 'N'
	if latitude < 0:
		NS = 'S'
	EW = 'E'
	if longitude < 0:
		EW = 'W'
	return '{}{:02d}{}{:03d}'.format(NS, abs(latitude), EW, abs(longitude))

def listFD(url, ext=''):
    response = requests.get(url)
    if response.status_code != 200:
    	return []
    soup = BeautifulSoup(response.text, 'html.parser')
    return [url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]

def randomASTERlocation():
	# get source of tile list if one hasn't been yet generated
	if not os.path.exists(os.path.expanduser('~/color_out_of_earth/aster_tile_list.json')):
		locCodes = {}
		for filename in listFD('https://e4ftl01.cr.usgs.gov/ASTT/ASTGTM.003/2000.03.01/', 'zip'):
			locCode = filename[-11:-4]
			locCodes[locCode] = True
		with open(os.path.expanduser('~/color_out_of_earth/aster_tile_list.json'), 'w') as write_file:
				json.dump(locCodes, write_file, indent='')
	# load tile list
	with open(os.path.expanduser('~/color_out_of_earth/aster_tile_list.json'), "r") as read_file:
		locCodes = json.load(read_file)
	locationCode = random.choice(list(locCodes))
	latitude, longitude = parseLocationCode(locationCode)
	print(locationCode)
	return locationCode, latitude, longitude

def downloadZip(url, un, pw):
	response = requests.get(url, auth = requests.auth.HTTPBasicAuth(un, pw), stream=True)
	if response.status_code == 200:
		return downloadResponseWithStatus(response2)
	elif response.url:
		print("redirected")
		response2 = requests.get(response.url, auth = requests.auth.HTTPBasicAuth(un, pw), headers = {'user-agent': 'Firefox'}, stream=True)
		if response2.status_code == 200:
			return downloadResponseWithStatus(response2)
		else:
			print(response.status_code, "could not get", url)
			return None

def parseArguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('--previous', '-p', action='store_true', help='Use previously downloaded tile.')
	parser.add_argument('-output', '-o', nargs='?', type=str, help='Path to save output image.')
	parser.add_argument('-latitude', '-lat', nargs='?', type=int, help='Integer latitude of desired tile.')
	parser.add_argument('-longitude', '-lon', nargs='?', type=int, help='Integer longitude of desired tile.')
	parser.add_argument('-lightnesses', '-ls', nargs='+', type=int, help='0-100. Up to three lightnesses, in order of elevation. The remaining lightnesses will be chosen randomly.')
	parser.add_argument('-chromas', '-cs', nargs='+', type=int, help='0-134. Up to three chromas, in order of elevation. The remaining chromas will be chosen randomly.')
	parser.add_argument('-hues', '-hs', nargs='+', type=int, help='0-359. Up to three hues, in order of elevation. The remaining hues will be chosen randomly.')
	parser.add_argument('-maxchroma', nargs='?', type=int, default=134, help='0-134. Maximum chroma of image.')
	return parser.parse_args()

args = parseArguments()

zip_data = None
locationCode = None
if args.previous and os.path.exists(os.path.expanduser('~/color_out_of_earth/previous.txt')):
	in_txt_file = open(os.path.expanduser('~/color_out_of_earth/previous.txt'), "r")
	lines = in_txt_file.readlines()
	locationCode, latitude, longitude = lines[0].strip(), int(lines[1].strip()), int(lines[2].strip())
	print(locationCode)

if locationCode is None:
	username, password = getEOSDISlogin()
	if args.latitude and args.longitude:
		latitude, longitude = args.latitude, args.longitude
		locationCode = latLonToLocationCode(latitude, longitude)
		print(locationCode)
	else:
		# locationCode, latitude, longitude = randomSRTMlocation()
		locationCode, latitude, longitude = randomASTERlocation()
	zip_data = downloadZip('https://e4ftl01.cr.usgs.gov/ASTT/ASTGTM.003/2000.03.01/ASTGTMV003_{}.zip'.format(locationCode), username, password)

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


if not zip_data is None:
	with ZipFile(BytesIO(zip_data), 'r') as zip:
		zip.extract('ASTGTMV003_{}_dem.tif'.format(locationCode), path=os.path.expanduser('~/color_out_of_earth'))
arr = np.array(Image.open(os.path.expanduser('~/color_out_of_earth/ASTGTMV003_{}_dem.tif'.format(locationCode))))

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

arr = arr.astype(np.single)
print("astype", arr.shape, arr.min(), arr.max())

# Image.fromarray(autocontrastedUint8(arr)).save(os.path.expanduser('~/color_out_of_earth/el-cropped.tif'))

# clip negative bits if this crop contains sea level
if 0 in arr and arr.min() < 0:
	arr = np.clip(arr, 0, None)
	# print(np.count_nonzero(arr<0))
	print("clipped to ocean", arr.min(), arr.max())

# find water
slope = slopeOfArray(arr)
# Image.fromarray(autocontrastedUint8(slope)).save(os.path.expanduser('~/color_out_of_earth/slope.tif'))
slopeFiltered = rank.maximum(autocontrastedUint8(slope), disk(4))
# Image.fromarray(autocontrastedUint8(slopeFiltered)).save(os.path.expanduser('~/color_out_of_earth/slope-filtered.tif'))
flat = slopeFiltered == 0
Image.fromarray(flat).save(os.path.expanduser('~/color_out_of_earth/flat.tif'))
labels, num_features = label(flat)
# print(num_features, "flat areas")
# Image.fromarray(labels).save(os.path.expanduser('~/color_out_of_earth/labels.tif'))
water = None
for label in range(1, num_features+1):
	count = np.count_nonzero(labels==label)
	if count > 20:
		thisLabel = labels == label
		if water is None:
			water = thisLabel
		else:
			water = water | thisLabel
if not water is None:
	Image.fromarray(water).save(os.path.expanduser('~/color_out_of_earth/water.tif'))
	waterElevations = arr[water]
	print("water elevations", waterElevations.min(), waterElevations.max())
	# arr = np.clip(arr, waterElevations.min(), None)
	# print("clipped to lowest water", arr.min(), arr.max())
	# Image.fromarray(autocontrastedUint8(arr)).save(os.path.expanduser('~/color_out_of_earth/el-clipped.tif'))

# stretch and rotate elevation data
arr = resize(arr, dsize=(cropWidth, cropHeight), interpolation=INTER_CUBIC)
print("resize", arr.shape, arr.min(), arr.max())
if rotation > 0:
	arr = np.rot90(arr, k=rotation)
print("rotation", arr.shape, arr.min(), arr.max())

# process elevation map for hillshading
# oldMin, oldMax = arr.min(), arr.max()
# arrForFilter = autocontrast(arr, min(65535, (oldMax - oldMin) * 15)).astype(np.uint16)
# Image.fromarray(autocontrastedUint8(arr)).save(os.path.expanduser('~/color_out_of_earth/prefilter.tif'))
# arrForShade = rank.mean_bilateral(arrForFilter, disk(16), s0=14, s1=14)
# Image.fromarray(autocontrastedUint8(arrForShade)).save(os.path.expanduser('~/color_out_of_earth/postfilter.tif'))
# print("filtered", arrForShade.min(), arrForShade.max())
# arrForShade = decontrast(arrForShade.astype(np.single), oldMin, oldMax)
# print("decontrasted", arrForShade.min(), arrForShade.max())
# arrForShade = arr
arrForShade = arr / metersPerPixel # so that the height map's vertical units are the same as its horizontal units
print("for shade", arrForShade.min(), arrForShade.max())

# create hillshade
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
hs = (0.9 * autocontrast(hsSum, 255)) + (0.1 * image_histogram_equalization(hsSum, 256))
print(hs.min(), np.median(hs), hs.max())
hs = hs + (127 - np.median(hs)) # linearly center median
# print(np.count_nonzero(hs < 0), "pixels below 0")
if hs.min() < 0:
	# compress negative values
	widthBelow = 127 - hs.min()
	hs = np.where(hs < 128, hs - (hs.min() * ((widthBelow - (hs - hs.min())) / widthBelow)), hs)
# print(np.count_nonzero(hs == 0), "pixels at 0")
print(hs.min(), np.median(hs), hs.max())
print(hs.min(), np.median(hs), hs.max())
hs_img = Image.fromarray(hs.astype(np.uint8))
print(hs_img.mode, len(hs_img.getcolors()))

# convert equalized elevation data to 256-color grayscale image
# arr_eq = image_histogram_equalization(arr, 256)
# print("equalized", arr_eq.min(), arr_eq.max())
el_img = Image.fromarray(autocontrastedUint8(arr))
# el_img = Image.fromarray(autocontrastedUint8(arr))
print(el_img.mode, len(el_img.getcolors()))
print(el_img.size)

# pick hues
ah, bh, ch = None, None, None
if args.hues:
	ah = args.hues[0]
	if len(args.hues) > 1:
		bh = args.hues[1] 
	if len(args.hues) > 2:
		ch = args.hues[2]
if ah is None:
	ah = perceptuallyUniformRandomHue()
if bh is None:
	bh = ah
	while huesDeltaE(ah, bh) < 20 or huesDeltaE(ah, bh) > 40:
		bh = perceptuallyUniformRandomHue()
if ch is None:
	ch = ah
	while huesDeltaE(ch, ah) < 20 or huesDeltaE(ch, bh) < 20 or (huesDeltaE(ch, ah) > 40 and huesDeltaE(ch, bh) > 40):
		ch = perceptuallyUniformRandomHue()
print('hues:', ah, bh, ch)

# pick lightnesses
darkMidLight = [random.randint(5,25), random.randint(40, 60), random.randint(75,95)]
lOrders = [
	[0, 1, 2],
	[2, 0, 1],
	[1, 0, 2],
	[0, 2, 1]]
lOrder = random.choice(lOrders)
if args.lightnesses:
	if len(args.lightnesses) == 1:
		lOrder = random.choice([lOrders[0], lOrders[3]])
	if len(args.lightnesses) == 2:
		lOrder = lOrders[0]
ls = [darkMidLight[lOrder[0]], darkMidLight[lOrder[1]], darkMidLight[lOrder[2]]]
if args.lightnesses:
	if len(args.lightnesses) == 1:
		ls = [args.lightnesses[0], ls[1], ls[2]]
	elif len(args.lightnesses) == 2:
		ls = [args.lightnesses[0], args.lightnesses[1], ls[2]]
	elif len(args.lightnesses) == 3:
		ls = args.lightnesses
print('lightnesses:', *ls)

# pick chromas
chromas = [None, None, None]
if args.chromas:
	cIndex = 0
	for chroma in args.chromas:
		chromas[cIndex] = chroma
		cIndex += 1	
for cIndex in range(0, 3):
	chromas[cIndex] = chromas[cIndex] or (random.randint(0,1) == 1 and random.randint(0, 134)) or args.maxchroma
print('chromas:', chromas)

# create gradient
a = highestChromaColor(ls[0], ah, chromas[0])
b = highestChromaColor(ls[1], bh, chromas[1])
c = highestChromaColor(ls[2], ch, chromas[2])
allSteps = a.steps([b, c], steps=256, space='lch-d65')
highChromaSteps = []
sIndex = 0
for col in allSteps:
	lch = col.convert('lch-d65')
	if sIndex > 127:
		mult = (sIndex - 127) / 128
		chroma = ((1-mult) * chromas[1]) + (mult * chromas[2])
	else:
		mult = sIndex / 127
		chroma = ((1-mult) * chromas[0]) + (mult * chromas[1])
	highChromaSteps.append(highestChromaColor(lch.l, lch.h, chroma))
	# print(sIndex, highChromaSteps[-1].convert('lch-d65').c)
	sIndex += 1
i = highChromaSteps[0].interpolate(highChromaSteps[1:], space='lch-d65')

# colorize elevation data
color_el_img = colorizeWithInterpolation(el_img, i)

# blend colorized elevation with hillshade
color_el_arr = np.array(color_el_img.convert('RGBA'))
hs_arr = np.array(hs_img.convert('RGBA'))
blended_float = overlay(color_el_arr.astype(float), hs_arr.astype(float), 1)
# blended_float = overlay(hs_arr.astype(float), color_el_arr.astype(float), 1)
blended_arr = np.uint8(blended_float)
blended_img = Image.fromarray(blended_arr).convert('RGB')

# save images
# Image.fromarray(autocontrastedUint16(arr)).save(os.path.expanduser('~/color_out_of_earth/el16_img.tif'))
# el_img.save(os.path.expanduser('~/color_out_of_earth/el_img.tif'))
hs_img.save(os.path.expanduser('~/color_out_of_earth/hs_img.tif'))
# color_hs_img.save(os.path.expanduser('~/color_out_of_earth/color_hs_img.tif'))
color_el_img.save(os.path.expanduser('~/color_out_of_earth/color_el_img.tif'))
blended_img.save(os.path.expanduser('~/color_out_of_earth/blended_img.png'))
if args.output:
	if os.path.exists(os.path.split(args.output)[0]):
		if os.path.splitext(args.output)[1] == '':
			blended_img.save(args.output, format='PNG')
		else:
			blended_img.save(args.output)

print("saved images")

if not zip_data is None:
	# delete previous tiff
	if os.path.exists(os.path.expanduser('~/color_out_of_earth/previous.txt')):
		in_txt_file = open(os.path.expanduser('~/color_out_of_earth/previous.txt'), "r")
		lines = in_txt_file.readlines()
		prevLocCode, prevLat, prevLon = lines[0].strip(), int(lines[1].strip()), int(lines[2].strip())
		prevTifFilepath = os.path.expanduser('~/color_out_of_earth/ASTGTMV003_{}_dem.tif'.format(prevLocCode))
		if os.path.exists(prevTifFilepath):
			os.remove(prevTifFilepath)
			# print("deleted", prevZipFilepath)
	# save previous info
	out_txt_file = open(os.path.expanduser('~/color_out_of_earth/previous.txt'), "w")
	out_txt_file.write("{}\n{}\n{}".format(locationCode, latitude, longitude))
	out_txt_file.close()