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
from PIL import Image, ImageFilter
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
from cv2 import resize, INTER_LINEAR
from bs4 import BeautifulSoup
import concurrent.futures
from screeninfo import get_monitors

# local modules
import catacomb
from perceptual_hues_lavg import perceptualHues

storageDir = os.path.expanduser('~/the_colour_out_of_earth')

CurrentGrade = None
CurrentlyDownloadingFiles = []
StatusPrintLock = False

degreesPerTheta = 90 / (math.pi / 2)

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

def uniformlyRandomLatLon():
	# https://www.cs.cmu.edu/~mws/rpos.html
	z = random.randint(-10000000, 10000000) / 10000000
	lat = math.asin(z) * degreesPerTheta
	lon = random.randint(-18000000, 18000000) / 100000
	return lat, lon

def measureLatLonInMeters(lat1, lon1, lat2, lon2):
	R = 6378.137 # Radius of earth in KM
	dLat = lat2 * math.pi / 180 - lat1 * math.pi / 180
	dLon = lon2 * math.pi / 180 - lon1 * math.pi / 180
	a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(lat1 * math.pi / 180) * math.cos(lat2 * math.pi / 180) * math.sin(dLon/2) * math.sin(dLon/2)
	c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
	d = R * c
	return d * 1000

def xyMultAtLatLon(latitude, longitude, size=1):
	halfSize = size / 2
	xMeters = measureLatLonInMeters(latitude, longitude - halfSize, latitude, longitude + halfSize)
	yMeters = measureLatLonInMeters(latitude - halfSize, longitude, latitude + halfSize, longitude)
	yMult = yMeters / xMeters
	xMult = 1
	if yMult > 2:
		xMult = 0.5
		yMult = yMult / 2
	return xMult, yMult, xMeters, yMeters

def tileListCropStretchFromLatLonCenter(latitude, longitude, width, height):
	xMult, yMult, xMeters, yMeters = xyMultAtLatLon(latitude, longitude, 1)
	cropWidth = int(width / xMult)
	cropHeight = int(height / yMult)
	lonWidth = cropWidth / 3601
	latHeight = cropHeight / 3601
	lonMin = longitude - (lonWidth / 2)
	if lonMin < -180:
		lonMin += 360
	lonMax = longitude + (lonWidth / 2)
	if lonMax > 180:
		lonMax -= 360
	latMin = latitude - (latHeight / 2)
	latMax = latitude + (latHeight / 2)
	if latMin < -90:
		latMax -= 90 + latMin
		latMin = -90
	if latMax > 90:
		latMin -= latMax - 90
		latMax = 90
	codes = {}
	for lat in range(math.floor(latMax), math.floor(latMin)-1, -1):
			for lon in range(math.floor(lonMin), math.floor(lonMax)+1):
				codes[latLonToLocationCode(lat, lon)] = True
	cropX1 = int((lonMin - math.floor(lonMin)) * 3601)
	cropX2 = cropX1 + cropWidth
	cropY1 = int((math.ceil(latMax) - latMax) * 3601)
	cropY2 = cropY1 + cropHeight
	metersPerPixel = xMeters / 3601
	return [*codes], cropX1, cropX2, cropY1, cropY2, xMult, yMult, metersPerPixel

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

def colorizeWithInterpolation(bwImage, interpolation, alpha=False):
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
	if alpha:
		alphaGrade = [int(interpolation(l/divisor).alpha * 255) for l in range(numColors)]
		CurrentGrade = alphaGrade
		alphaImage = Image.eval(bwImage, gradeFunc)
		return Image.merge('RGBA', (redImage, greenImage, blueImage, alphaImage))
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

def getEOSDISlogin():
	comb = catacomb.Catacomb(storageDir, 'uR7UtrczNUM0FnzR8X')
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

def printDownloadStatus(overwrite=True):
	global StatusPrintLock
	if StatusPrintLock:
		return
	StatusPrintLock = True
	if overwrite == True:
		for d in CurrentlyDownloadingFiles:
			sys.stdout.write("\033[F")
	for d in CurrentlyDownloadingFiles:
		sys.stdout.write("{} {} {}\n".format(d.get('code'), d.get('layer'), d.get('status') or ''))
	StatusPrintLock = False

def downloadResponseWithStatus(response, download):
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
			download['status'] = "{}% ({} / {} kB)".format(percent, kB, totalkB)
		else:
			download['status'] = "{} kB".format(kB)
		printDownloadStatus()
		content += chunk
	return content

def listFD(url, ext=''):
    response = requests.get(url)
    if response.status_code != 200:
    	return []
    soup = BeautifulSoup(response.text, 'html.parser')
    return [url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]

def loadSRTMtileList():
	# get source of tile list if one hasn't been yet generated
	if not os.path.exists(storageDir + '/srtm_tile_list.json'):
		print('downloading SRTM tile list from https://dwtkns.com/srtm30m/srtm30m_bounding_boxes.json')
		response = requests.get('https://dwtkns.com/srtm30m/srtm30m_bounding_boxes.json')
		if response.status_code == 200:
			locCodes = {}
			collection = json.loads(response.text)
			for f in collection.get('features'):
				locCode = f.get('properties').get('dataFile').split('.')[0]
				locCodes[locCode] = True
			with open(storageDir + '/srtm_tile_list.json', 'w') as write_file:
				json.dump(locCodes, write_file, indent='')
		else:
			print('could not download', response.url)
	# load tile list
	with open(storageDir + '/srtm_tile_list.json', "r") as read_file:
		locCodes = json.load(read_file)
	return locCodes

def loadASTERtileList():
	# get source of tile list if one hasn't been yet generated
	if not os.path.exists(storageDir + '/aster_tile_list.json'):
		print('downloading ASTER tile list from https://e4ftl01.cr.usgs.gov/ASTT/ASTGTM.003/2000.03.01/')
		locCodes = {}
		for filename in listFD('https://e4ftl01.cr.usgs.gov/ASTT/ASTGTM.003/2000.03.01/', 'zip'):
			locCode = filename[-11:-4]
			locCodes[locCode] = True
		with open(storageDir + '/aster_tile_list.json', 'w') as write_file:
				json.dump(locCodes, write_file, indent='')
	# load tile list
	with open(storageDir + '/aster_tile_list.json', "r") as read_file:
		locCodes = json.load(read_file)
	return locCodes

def downloadWithAuth(url, un, pw, download):
	response = requests.get(url, auth = requests.auth.HTTPBasicAuth(un, pw), stream=True)
	if response.status_code == 200:
		return downloadResponseWithStatus(response2, download)
	elif response.url:
		download['status'] = 'redirected'
		printDownloadStatus()
		attempt = 1
		response2 = None
		while response2 is None or response2.status_code != 200:
			try:
				response2 = requests.get(response.url, auth = requests.auth.HTTPBasicAuth(un, pw), headers = {'user-agent': 'Firefox'}, stream=True)
			except:
				response2 = None
			attempt += 1
		return downloadResponseWithStatus(response2, download)

def extractFile(file):
	if file.get('zipped'):
		with ZipFile(BytesIO(file.get('content')), 'r') as zip:
			with zip.open(file.get('zipped'), mode='r') as zippedFile:
				ext = os.path.splitext(file.get('zipped'))[1].lower()
				if ext == '.hgt':
					raw = zippedFile.read()
					siz = len(raw)
					dim = int(math.sqrt(siz/2))
					file['array'] = np.frombuffer(raw, np.dtype('>i2'), dim*dim).reshape((dim, dim))
				elif ext == '.tif':
					file['array'] = np.array(Image.open(BytesIO(zippedFile.read())))

def downloadOneFile(file):
	file['content'] = downloadWithAuth(file.get('url'), file.get('username'), file.get('password'), file)

def downloadFiles(files):
	global CurrentlyDownloadingFiles
	CurrentlyDownloadingFiles = files
	print(" ")
	printDownloadStatus(False)
	with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
		executor.map(downloadOneFile, files)

def downloadTiles(codes, username, password):
	downloads = []
	for code in codes:
		if SRTMlocationCodes.get(code):
			url = 'http://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11/{}.SRTMGL1.hgt.zip'.format(code)
			zipped= '{}.hgt'.format(code)
		elif ASTERlocationCodes.get(code):
			url = 'https://e4ftl01.cr.usgs.gov/ASTT/ASTGTM.003/2000.03.01/ASTGTMV003_{}.zip'.format(code)
			zipped = 'ASTGTMV003_{}_dem.tif'.format(code)
		downloads.append({'url':url, 'code':code, 'layer':'elevation', 'zipped':zipped, 'username':username, 'password':password})
		if not args.nowater:
			downloads.append({'url':'https://e4ftl01.cr.usgs.gov/ASTT/ASTWBD.001/2000.03.01/ASTWBDV001_{}.zip'.format(code), 'code':code, 'layer':'waterbody', 'zipped':'ASTWBDV001_{}_dem.tif'.format(code), 'username':username, 'password':password})
	downloadFiles(downloads)
	return downloads

def extractTiles(downloads):
	for d in downloads:
		extractFile(d)

def arrangeTiles(downloads):
	# get bounds
	latMin, latMax, lonMin, lonMax = None, None, None, None
	layers = {}
	for d in downloads:
		lat, lon = parseLocationCode(d.get('code'))
		layer = d.get('layer')
		if layers.get(layer) is None:
			layers[layer] = {'lats':{}}
		if layers.get(layer).get('lats').get(lat) is None:
			layers[layer]['lats'][lat] = {}
		layers[layer]['lats'][lat][lon] = d.get('array')
		if latMin is None or lat < latMin:
			latMin = lat
		if latMax is None or lat > latMax:
			latMax = lat
		if lonMin is None or lon < lonMin:
			lonMin = lon
		if lonMax is None or lon > lonMax:
			lonMax = lon
	outLayers = {}
	for layer in layers.keys():
		thisLayer = layers.get(layer)
		rows = None
		for lat in range(latMax, latMin-1, -1):
			row = None
			for lon in range(lonMin, lonMax+1):
				tileArr = thisLayer.get('lats').get(lat).get(lon)
				if row is None:
					row = tileArr
				else:
					row = np.concatenate((row, tileArr), axis=1)
			if rows is None:
				rows = row
			else:
				rows = np.concatenate((rows, row), axis=0)
		outLayers[layer] = rows
	return outLayers

def allFlat(arr):
	if arr is None:
		# print("array is none")
		return True
	if len(np.unique(arr)) == 1:
		# print("only 1 unique value in array")
		return True

def checkOutLocationCodes(codes):
	codesHaveSRTM, codesHaveASTER = False, False
	for code in codes:
		inSRTM = SRTMlocationCodes.get(code)
		inASTER = ASTERlocationCodes.get(code)
		if not inSRTM and not inASTER:
			return False
		if inSRTM and not codesHaveSRTM:
			codesHaveSRTM = inSRTM
		if inASTER and not inSRTM and not codesHaveASTER:
			codesHaveASTER = inASTER
	if codesHaveSRTM and codesHaveASTER:
		# SRTM tiles and ASTER tiles don't line up perfectly and show a seam when put together
		return False
	return True

def parseArguments():
	parser = argparse.ArgumentParser(description='Create a colorful image of terrain of a random location.')
	parser.add_argument('--previous', '-p', action='store_true', help='Use previously downloaded data. --dimensions, --coordinates, and --rotation will have no effect.')
	parser.add_argument('--nowater', '-w', action='store_true', help='Do not download or draw bodies of water.')
	parser.add_argument('--noshade', '-s', action='store_true', help='Do not hillshade the terrain. This leaves only gradient-mapped elevations and water bodies.')
	parser.add_argument('--output', '-o', nargs='?', type=str, metavar='FILEPATH', help='Path to save output image. If not specified, will save to ~/the_colour_out_of_earth/output.png along with elevation_gradient.tif, hillshade.tif, and water.tif')
	parser.add_argument('--coordinates', '-c', nargs=2, type=float, metavar=('LATITUDE', 'LONGITUDE'), help='Location of center of desired image in latitude longitude coordinates. If not specified, a random location will be chosen.')
	parser.add_argument('--dimensions', '-d', nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'), help='Width and height in pixels of output image. Larger images will require downloading more source tiles.')
	parser.add_argument('--rotation', '-r', nargs='?', type=int, metavar='0-3', help='How many times 90 degrees to rotate. (0: North is up. 1: East is up. 2: South is up. 3: West is up.) If not specified, this will be chosen randomly.')
	parser.add_argument('--maxchroma', nargs='?', type=float, default=134, metavar='0-134', help='Maximum chroma of image.')
	parser.add_argument('--lightnesses', nargs='+', type=int, metavar='0-100', help='Up to three lightnesses, in order of elevation. The remaining lightnesses will be chosen randomly.')
	parser.add_argument('--chromas', nargs='+', type=int, metavar='0-134', help='Up to three chromas, in order of elevation. The remaining chromas will be chosen randomly.')
	parser.add_argument('--hues', nargs='+', type=int, metavar='0-359', help='Up to three hues, in order of elevation. The remaining hues will be chosen randomly.')
	return parser.parse_args()

args = parseArguments()

if not os.path.exists(storageDir):
	os.makedirs(storageDir)

SRTMlocationCodes = loadSRTMtileList()
ASTERlocationCodes = loadASTERtileList()

# get screen dimensions
screenWidth, screenHeight = None, None
for m in get_monitors():
	if screenWidth == None or m.width > screenWidth:
		screenWidth = m.width
	if screenHeight == None or m.height > screenHeight:
		screenHeight = m.height
print("screen:", screenWidth, screenHeight)

if args.previous and os.path.exists(storageDir + '/previous.json'):
	# get previous info if asked for and exists
	with open(storageDir + '/previous.json', "r") as read_file:
		previousInfo = json.load(read_file)
		args.rotation = int(previousInfo.get('rotation'))
		args.coordinates = [float(previousInfo.get('latitude')), float(previousInfo.get('longitude'))]
		args.dimensions = [int(previousInfo.get('width')), int(previousInfo.get('height'))]

if args.dimensions:
	targetWidth, targetHeight = args.dimensions[0], args.dimensions[1]
else:
	targetWidth, targetHeight = screenWidth, screenHeight
print("output dimensions:", targetWidth, targetHeight)

# get encrypted EOSDIS login info
username, password = getEOSDISlogin()

arr, wbd_arr = None, None
downloadCropAttempt = 0
while downloadCropAttempt < 20 and allFlat(arr):
	codes = []
	attempt = 0
	# find coordinates that are within SRTM and ASTER data
	while attempt < 50 and (len(codes) == 0 or not checkOutLocationCodes(codes)):
		# rotate to get pre-rotated target dimensions
		if not args.rotation is None:
			rotation = args.rotation
		else:
			rotation = random.randint(0, 3)
		if rotation == 1 or rotation == 3:
			rotatedWidth, rotatedHeight = targetHeight, targetWidth
		else:
			rotatedWidth, rotatedHeight = targetWidth, targetHeight
		# pick a random location or use specified coordinates
		if args.coordinates and attempt == 0 and downloadCropAttempt == 0:
			latitude, longitude = args.coordinates[0], args.coordinates[1]
		else:
			latitude, longitude = uniformlyRandomLatLon()
		codes, cropX1, cropX2, cropY1, cropY2, xMult, yMult, metersPerPixel = tileListCropStretchFromLatLonCenter(latitude, longitude, rotatedWidth, rotatedHeight)
		attempt += 1
	# download, arrange, and crop tiles
	print(latitude, longitude)
	print("rotation:", rotation, rotatedWidth, rotatedHeight)
	print(codes, cropX1, cropX2, cropY1, cropY2, xMult, yMult, metersPerPixel)
	if args.previous:
		# use previously downloaded cropped images
		arr = np.array(Image.open(storageDir + '/elevation-cropped.tif'))
		if not args.nowater and os.path.exists(storageDir + '/waterbody-cropped.tif'):
			wbd_arr = np.array(Image.open(storageDir + '/waterbody-cropped.tif'))
	else:
		# download and arrange tiles into images
		tiles = downloadTiles(codes, username, password)
		extractTiles(tiles)
		layers = arrangeTiles(tiles)
		arr = layers.get('elevation')
		# Image.fromarray(arr).save(storageDir + '/elevation.tif')
		# Image.fromarray(wbd_arr).save(storageDir + '/waterbody.tif')
		print(arr.shape)
		print("from {:,} m to {:,} m".format(arr.min(), arr.max()))
		# crop
		arr = arr[cropY1:cropY2, cropX1:cropX2]
		print("cropped", arr.shape, arr.min(), arr.max())
		if not args.nowater:
			wbd_arr = layers.get('waterbody')
			wbd_arr = wbd_arr[cropY1:cropY2, cropX1:cropX2]
	downloadCropAttempt += 1

Image.fromarray(arr).save(storageDir + '/elevation-cropped.tif')
if not wbd_arr is None:
	Image.fromarray(wbd_arr).save(storageDir + '/waterbody-cropped.tif')
	if allFlat(wbd_arr):
		# ignore waterbody maps with no data
		wbd_arr = None

arr = arr.astype(np.single)
# print("astype", arr.shape, arr.min(), arr.max())

# Image.fromarray(autocontrastedUint8(arr)).save(storageDir + '/el-cropped.tif')

# clip negative bits if this crop contains sea level
if 0 in arr and arr.min() < 0:
	arr = np.clip(arr, 0, None)
	# print(np.count_nonzero(arr<0))
	print("clipped to ocean", arr.min(), arr.max())

# restrict waterbody image to only those areas with 0 slope, so it doesn't cut off the hillshade
if not wbd_arr is None:
	slope = slopeOfArray(arr)
	wbd_arr = (slope == 0) & (wbd_arr > wbd_arr.min())
	wbd_arr = autocontrastedUint8(wbd_arr.astype(np.uint8))

# stretch and rotate elevation data
arr = resize(arr, dsize=(rotatedWidth, rotatedHeight), interpolation=INTER_LINEAR)
if not wbd_arr is None:
	wbd_arr = resize(wbd_arr, dsize=(rotatedWidth, rotatedHeight), interpolation=INTER_LINEAR)
print("resized", arr.shape, arr.min(), arr.max())
if rotation > 0:
	arr = np.rot90(arr, k=rotation)
	if not wbd_arr is None:
		wbd_arr = np.rot90(wbd_arr, k=rotation)
print("rotated", arr.shape, arr.min(), arr.max())

# process elevation map for hillshading
arrForShade = arr / metersPerPixel # so that the height map's vertical units are the same as its horizontal units
print("for shade", arrForShade.min(), arrForShade.max())

if not args.noshade:
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
		# Image.fromarray(hs.astype(np.uint8)).save(storageDir + '/hs-{}-{}-{}.tif'.format(*shade))
		if hsSum is None:
			hsSum = hs
		else:
			hsSum += hs
	hs = (0.9 * autocontrast(hsSum, 255)) + (0.1 * image_histogram_equalization(hsSum, 256))
	# print(hs.min(), np.median(hs), hs.max())
	hs = hs + (127 - np.median(hs)) # linearly center median
	# print(np.count_nonzero(hs < 0), "pixels below 0")
	if hs.min() < 0:
		# compress negative values
		widthBelow = 127 - hs.min()
		hs = np.where(hs < 128, hs - (hs.min() * ((widthBelow - (hs - hs.min())) / widthBelow)), hs)
	# print(np.count_nonzero(hs == 0), "pixels at 0")
	# print(hs.min(), np.median(hs), hs.max())
	hs_img = Image.fromarray(hs.astype(np.uint8))

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
darkMidLight = [random.randint(5,30), random.randint(35, 65), random.randint(70,97)]
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
	chromas[cIndex] = chromas[cIndex] or (random.randint(0,1) == 1 and random.randint(0, args.maxchroma)) or args.maxchroma
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

# convert elevation data to 256-color grayscale image
# equalize data with a wide range
if arr.max() - arr.min() > 100:
	arr_eq = (0.5 * image_histogram_equalization(arr, 256)) + (0.5 * autocontrast(arr, 255))
else:
	arr_eq = arr
el_img = Image.fromarray(autocontrastedUint8(arr_eq))

# colorize elevation data
color_el_img = colorizeWithInterpolation(el_img, i)

# colorize water bodies
waterColor = highestChromaColor(darkMidLight[0], random.choice([bh,ch]), args.maxchroma)
waterInterpol = coloraide.Color('srgb', [0, 0, 0], 0).interpolate(waterColor)
if not wbd_arr is None:
	wbd_img = Image.fromarray(wbd_arr).filter(ImageFilter.GaussianBlur(radius=0.67))
	color_wbd_img = colorizeWithInterpolation(wbd_img, waterInterpol, True)

if args.noshade:
	blended_img = color_el_img.convert('RGBA')
else:
	# blend colorized elevation with hillshade
	color_el_arr = np.array(color_el_img.convert('RGBA'))
	hs_arr = np.array(hs_img.convert('RGBA'))
	blended_float = overlay(color_el_arr.astype(float), hs_arr.astype(float), 1)
	# blended_float = overlay(hs_arr.astype(float), color_el_arr.astype(float), 1)
	blended_arr = np.uint8(blended_float)
	blended_img = Image.fromarray(blended_arr)

if wbd_arr is None:
	blended_img = blended_img.convert('RGB')
else:
	blended_img = Image.alpha_composite(blended_img, color_wbd_img).convert('RGB')

# save images
# el_img.save(storageDir + '/el_img.tif')
if args.output:
	if os.path.exists(os.path.split(args.output)[0]):
		ext = os.path.splitext(args.output)[1].lower()
		if ext == '':
			blended_img.save(args.output, format='PNG')
		elif ext == '.jpg' or ext == '.jpeg':
			blended_img.save(args.output, quality=95)
		else:
			blended_img.save(args.output)
else:
	if not args.noshade:
		hs_img.save(storageDir + '/hillshade.tif')
	if not wbd_arr is None:
		color_wbd_img.save(storageDir + '/water.tif')
	color_el_img.save(storageDir + '/elevation_gradient.tif')
	blended_img.save(storageDir + '/output.png')

print("saved image(s)")

if not args.previous:
	# save previous info
	previousInfo = {'latitude':latitude, 'longitude':longitude, 'rotation':rotation, 'width':targetWidth, 'height':targetHeight}
	with open(storageDir + '/previous.json', 'w') as write_file:
		json.dump(previousInfo, write_file, indent='')