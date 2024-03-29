import math
import requests
import random
import json
import numpy as np
from os import path, makedirs, remove
from argparse import ArgumentParser
from PIL import Image
from zipfile import ZipFile
from coloraide import Color
from getpass import getpass
from io import BytesIO
from sys import stdout
from cv2 import resize, INTER_LINEAR, GaussianBlur, BORDER_DEFAULT, medianBlur
from concurrent.futures import ThreadPoolExecutor
from screeninfo import get_monitors
from numba import jit
from numba.types import uint8, float32, int8, boolean, int64, float64
from colour import XYZ_to_sRGB, Oklab_to_XYZ

rng = np.random.default_rng()

# local modules
import catacomb

storageDir = path.expanduser('~/a_sort_of_soup')
scriptDir = path.split(path.realpath(__file__))[0]

StatusPrintLock = False

halfPi = math.pi / 2
oneOverHalfPi = 1 / halfPi
piPer180 = math.pi / 180
degreesPerTheta = 90 / (math.pi / 2)

equatorRadius = 6378137
poleRadius = 6356752
equatorRadiusSq = equatorRadius ** 2
poleRadiusSq = poleRadius ** 2

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

def longitudeDistance(lonA, lonB):
	lonA360 = lonA + 180
	lonB360 = lonB + 180
	return abs(((lonA360 - lonB360) + 180) % 360 - 180)

def earthRadiusInMetersAtLatitude(latitude):
	# https://rechneronline.de/earth-radius/
	latRad = latitude * piPer180
	return math.sqrt(  ( ((equatorRadiusSq * math.cos(latRad)) ** 2) + ((poleRadiusSq * math.sin(latRad)) ** 2) ) / ( ((equatorRadius * math.cos(latRad)) ** 2) + ((poleRadius * math.sin(latRad)) ** 2) )  )

def measureLatLonInMeters(lat1, lon1, lat2, lon2):
	R = earthRadiusInMetersAtLatitude((lat1 + lat2) / 2)
	dLat = (lat2 * piPer180) - (lat1 * piPer180)
	dLon = longitudeDistance(lon1, lon2) * piPer180
	a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(lat1 * piPer180) * math.cos(lat2 * piPer180) * math.sin(dLon/2) * math.sin(dLon/2)
	c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
	d = R * c
	return d

def xyMultAtLatLon(latitude, longitude, size=1):
	halfSize = size / 2
	xMeters = measureLatLonInMeters(latitude, longitude - halfSize, latitude, longitude + halfSize)
	yMeters = measureLatLonInMeters(latitude - halfSize, longitude, latitude + halfSize, longitude)
	yMult = yMeters / xMeters
	xMult = 1
	if yMult > 4:
		xMult = 0.25
		yMult = yMult / 4
	if yMult > 2:
		xMult = 0.5
		yMult = yMult / 2
	return xMult, yMult, xMeters, yMeters

def angleDist(a, b):
	return abs(((b - a) + 180) % 360 - 180)

def lch_to_rgb(lightness, chroma, hue):
	c = Color('oklch', [lightness, chroma, hue]).convert('srgb')
	if c.in_gamut():
		return c
	return None

def highestChromaColor(lightness, hue, maxChroma=0.4):
	chromaStep = 0.1
	if maxChroma < 0.1:
		chromaStep = 0.01
	chroma = maxChroma
	iteration = 0
	while iteration < 45:
		c = lch_to_rgb(lightness, chroma, hue)
		if not c is None:
			if chromaStep == 0.0001 or maxChroma == 0 or iteration == 0:
				return c
			else:
				chroma += chromaStep
				chromaStep /= 10
				chroma -= chromaStep
		chroma = max(0, chroma - chromaStep)
		iteration += 1
	print(chromaStep, lightness, chroma, hue, iteration)

def huesDeltaE(hueA, hueB, lightness=0.74, chroma=0.12):
	a = lch_to_rgb(lightness, chroma, hueA)
	b = lch_to_rgb(lightness, chroma, hueB)
	return a.delta_e(b, method='2000')

def nextHueByDeltaE(hues, targetDeltaE):
	negative = random.choice([False, True])
	hue = random.choice(hues)
	fallbackDeltaE, fallbackHue = None, None
	for hueAdd in range(int(targetDeltaE * 0.75), 359):
		if negative == True:
			hueAdd = 0 - hueAdd
		newHue = (hue + hueAdd) % 360
		deltaE = 130
		for h in hues:
			deltaE = min(deltaE, huesDeltaE(h, newHue))
		if deltaE >= targetDeltaE:
			# print(deltaE, hueAdd)
			return newHue
		if fallbackDeltaE is None or deltaE > fallbackDeltaE:
			fallbackDeltaE = deltaE
			fallbackHue = newHue
	# print("fallback", fallbackDeltaE)
	return fallbackHue

# dither a 0-1 float image to a 0-255 integer image
def randomDitherImage(arr):
	r = rng.random(arr.shape)
	v255 = arr * 255
	low = np.floor(v255).astype(np.uint8)
	high = np.ceil(v255).astype(np.uint8)
	err = v255 - low
	mask = r > err
	dithered = np.zeros(arr.shape, dtype=np.uint8)
	dithered[mask] = low[mask]
	dithered[~mask] = high[~mask]
	return dithered

def arrayColorizeWithInterpolation(greyArr, interpolation, numColors=None, alpha=False, floatChannels=False, outputLCH=False):
	if numColors is None:
		if greyArr.dtype == np.uint8:
			numColors = 256
		elif greyArr.dtype == np.uint16:
			numColors = 65536
		else:
			print("input grey array for colorization is not uint8 or uint16, and numColors not specified")
			return
	else:
		greyArr = autocontrast(greyArr, numColors - 1)
		if numColors <= 256:
			greyArr = greyArr.astype(np.uint8)
		else:
			greyArr = greyArr.astype(np.uint16)
	divisor = numColors - 1
	channels = ['red', 'green', 'blue']
	if outputLCH:
		channels = ['lightness', 'chroma', 'hue']
	if alpha:
		channels.append('alpha')
	numChannels = len(channels)
	dt = np.uint8
	if floatChannels or outputLCH:
		dt = 'float32'
	lookupColor = np.zeros((numColors, numChannels), dtype=dt)
	for l in range(0, numColors):
		n = l / divisor
		if floatChannels:
			lookupColor[l] = [max(0,min(1,getattr(interpolation(n), channels[ci]))) for ci in range(numChannels)]
		elif outputLCH:
			lookupColor[l] = [getattr(interpolation(n).convert('oklch'), channels[ci]) for ci in range(numChannels)]
		else:
			lookupColor[l] = [max(0,min(255,round(getattr(interpolation(n), channels[ci]) * 255))) for ci in range(numChannels)]
	colorArr = np.zeros((*greyArr.shape, numChannels), dtype=dt)
	np.take(lookupColor, greyArr, axis=0, out=colorArr)
	return colorArr

def radialGradient(width, height, azimuth, angle):
	fWidth = width / min(width, height)
	halfFW = fWidth * 0.5
	halfExcessFW = abs(1 - fWidth) * 0.5
	centerWeight = angle / 90
	x0 = 0 - halfExcessFW
	x1 = 1 + halfExcessFW
	x0 = (x0 * (1-centerWeight)) - (halfFW * centerWeight)
	x1 = (x1 * (1-centerWeight)) + (halfFW * centerWeight)
	fHeight = height / min(width, height)
	halfFH = fHeight * 0.5
	halfExcessFH = abs(1 - fHeight) * 0.5
	y0 = 0 - halfExcessFH
	y1 = 1 + halfExcessFH
	y0 = (y0 * (1-centerWeight)) - (halfFH * centerWeight)
	y1 = (y1 * (1-centerWeight)) + (halfFH * centerWeight)
	azimuthNum = round(azimuth / 45) % 8
	xys = [
		[x0, x1, -halfFH, halfFH], # 0
		[x0, x1, y0, y1], # 45
		[-halfFW, halfFW, y0, y1], # 90
		[x1, x0, y0, y1], # 135
		[x1, x0, -halfFH, halfFH], # 180
		[x1, x0, y1, y0], # 225
		[-halfFW, halfFW, y1, y0], # 270
		[x0, x1, y1, y0]] # 315
	xy = xys[azimuthNum]
	# print(azimuth, azimuthNum, angle, centerWeight)
	# print(xy)
	X = np.linspace(xy[0], xy[1], width)[None, :]
	Y = np.linspace(xy[2], xy[3], height)[:, None]
	radgrad = np.sqrt(X**2 + Y**2)
	# radgrad = np.clip(0,1,radgrad)
	radgrad = autocontrast(radgrad, 1)
	return radgrad

def hardLightOrOverlayFloat(a, b, overlay=False):
	# generate boolean mask
	if overlay:
		mask = a >= 0.5
	else:
		mask = b >= 0.5
	ab = np.zeros_like(a) # generate an output container for the blended image 
	# now do the blending 
	ab[~mask] = (2 * a * b)[~mask] # 2ab everywhere b < 0.5
	ab[mask] = (1 - 2 * (1 - a) * (1 - b))[mask] # else this
	return ab

def OKLCH2RGB255(lch): # l 0-1, c 0-1, h 0-360
    illuminant = [0.3127, 0.329]
    c = lch[:,:,1]
    h = lch[:,:,2]
    hradians = h / (360 / (2 * math.pi))
    a = c * np.cos(hradians)
    b = c * np.sin(hradians)
    lab = np.stack((lch[:,:,0], a, b), axis=2)
    rgb = XYZ_to_sRGB(Oklab_to_XYZ(lab))
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

def lightOklch(aLCH, b, glow):
	a = aLCH[:,:,0]
	b = (b / 255).astype(np.single)
	if glow == 0:
		ab = hardLightOrOverlayFloat(a, b)
	elif glow == 1:
		ab = hardLightOrOverlayFloat(a, b, True)
	else:
		abHard = hardLightOrOverlayFloat(a, b)
		abOver = hardLightOrOverlayFloat(a, b, True)
		ab = (glow * abOver) + ((1 - glow) * abHard)
	# use the blended lightness channel
	newLCH = np.stack((ab, aLCH[:,:,1], aLCH[:,:,2]), axis=2)
	newRGB = OKLCH2RGB255(newLCH)
	return newRGB

def lightRGB(aRGB, b, glow):
	a = (aRGB / 255).astype(np.single)
	b1 = (b / 255).astype(np.single)
	b = np.stack((b1, b1, b1), axis=2)
	if glow == 0:
		ab = hardLightOrOverlayFloat(a, b)
	elif glow == 1:
		ab = hardLightOrOverlayFloat(a, b, True)
	else:
		abHard = hardLightOrOverlayFloat(a, b)
		abOver = hardLightOrOverlayFloat(a, b, True)
		ab = (glow * abOver) + ((1 - glow) * abHard)
	# use the blended lightness channel
	return (ab * 255).astype(np.uint8)

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
	gy, gx = np.gradient(array)
	return np.sqrt(gy*gy + gx*gx)

def hillshadePreparations(array):
	# x, y = np.gradient(array)
	y, x = np.gradient(array)
	slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
	aspect = np.arctan2(-x, y)
	return slope, aspect

def hillshade(array, azimuth, angle_altitude, slope=None, aspect=None):
	# azimuth = 360.0 - azimuth
	if slope is None:
		# x, y = np.gradient(array)
		y, x = np.gradient(array)
		slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
	if aspect is None:
		aspect = np.arctan2(-x, y)
	azimuthrad = azimuth*np.pi / 180.
	altituderad = angle_altitude*np.pi / 180.
	shaded = np.sin(altituderad) * np.sin(slope)\
	 + np.cos(altituderad) * np.cos(slope)\
	 * np.cos((azimuthrad - np.pi/2.) - aspect)
	return (shaded + 1)/2

def multiHillshade(shades, elevForShade):
	slopeForShade, aspectForShade = hillshadePreparations(elevForShade)
	# create layered hillshade
	hs = np.full(elevForShade.shape, 1, dtype='float32')
	for si in range(len(shades) - 1, -1, -1): # bottom layer first
		shade = shades[si]
		hsLayer = hillshade(elevForShade, shade[0], shade[1], slopeForShade, aspectForShade)
		# Image.fromarray((hsLayer*255).astype(np.uint8)).save(storageDir + '/hs-{}-{}.tif'.format(*shade))
		if len(shade) > 2:
			opacity = shade[2]
		else:
			opacity =  shade[1] / 90
		hs = (hsLayer * opacity) + (hs * (1 - opacity))
	depth = 0.25 + (args.glow * 0.5)
	hs -= (hs.min() * depth) # deepen shadows
	hs = hs * (0.5 / np.median(hs)) # move median to center value
	if args.shine > 0:
		# add shininess
		tval = ((1 - args.shine) * hs.max()) + args.shine
		hs = np.where(hs > 0.5, hs * (1 + ( ((hs - 0.5) / (hs.max() - 0.5)) * ((tval / hs.max()) - 1)) ), hs)
	# return (hs * 255).astype(np.uint8)
	return hs

def plotLine(sy, width):
	dx = width
	dy = round(sy * width)
	yi = 1
	if sy < 0:
		yi = -1
		dy = -dy
	dy2 = 2 * dy
	dyMinusdx2 = 2 * (dy - dx)
	D = dy2 - dx
	line = []
	for x in range(width):
		diff = 0
		if D > 0:
			diff = yi
			D = D + dyMinusdx2
		D = D + dy2
		line.append(diff)
	return line

def plotLightLine(azimuth, width, height):
	dx = max(-1, min(1, 1 - (2 * math.ceil(math.cos(azimuth * piPer180)))))
	dy = dx * math.tan(azimuth * piPer180)
	# print(azimuth, dx, dy)
	isYline = False
	if abs(dy) > 1:
		isYline = True
		dy = max(-1, min(1, 1 - (2 * math.ceil(math.sin(azimuth * piPer180)))))
		dx = dy * (1 / math.tan(azimuth * piPer180))
		line = plotLine(dx, height)
	else:
		line = plotLine(dy, width)
	return line, isYline, dx, dy

@jit(uint8[:,:](float32[:,:], uint8[:,:], int8[:], boolean, float64, float64, float64), nopython=True, cache=True)
def shadowLoop(elev, light, XYline, isYline, dx, dy, dz):
	height, width = elev.shape
	top = elev.max()
	if isYline:
		dy = int(dy)
		if dy == -1:
			lyStop = -1
		else:
			lyStop = height
		for y in range(height):
			for x in range(width):
				z = elev[y, x]
				lx, lz = x, z
				ly = y + dy
				li = 0
				while ly != lyStop:
					change = XYline[li]
					lx += change
					if lx < 0 or lx > width - 1:
						break
					lz += dz
					tz = elev[ly, lx]
					if lz < tz:
						light[y, x] = 0
						break
					elif lz > top:
						break
					ly += dy
					li += 1
	else:
		dx = int(dx)
		if dx == -1:
			lxStop = -1
		else:
			lxStop = width
		for y in range(height):
			for x in range(width):
				z = elev[y, x]
				ly, lz = y, z
				lx = x + dx
				li = 0
				while lx != lxStop:
					change = XYline[li]
					ly += change
					if ly < 0 or ly > height - 1:
						break
					lz += dz
					tz = elev[ly, lx]
					if lz < tz:
						light[y, x] = 0
						break
					elif lz > top:
						break
					lx += dx
					li += 1
	return light

def rayShadows(elevation, azimuth, angle):
	dz = math.tan(angle * piPer180)
	height, width = elevation.shape
	XYline, isYline, dx, dy = plotLightLine(azimuth, width, height)
	# print(dx, dy, dz, isYline)
	light = np.full((height, width), 255, dtype=np.uint8)
	light = shadowLoop(elevation, light, np.array(XYline, dtype=np.int8), isYline, dx, dy, dz)
	# fake y-edge with neighbors
	if dy < 0:
		light[0, :] = light[1, :]
	elif dy > 0:
		light[height - 1, :] = light[height - 2, :]
	# fake x-edge with neighbors
	if dx < 0:
		light[:, 0] = light[:, 1]
	elif dx > 0:
		light[:, width - 1] = light[:, width - 2]
	# Image.fromarray(light).save(storageDir + '/lightraw.tif')
	return GaussianBlur(light, (3, 3), BORDER_DEFAULT)

@jit(float32[:,:](float32[:,:], int64), nopython=True, cache=True)
def skyView(elevation, radius):
	# precompute eight lines dy, dx, and euclidean delta per segment
	linesDY = []
	linesDX = []
	linesEuc = []
	linesIterations = []
	for dy in [-1, 0, 1]:
		for dx in [-1, 0, 1]:
			if not (dx == 0 and dy == 0):
				linesDY.append(dy)
				linesDX.append(dx)
				euc = math.sqrt(dy**2 + dx**2)
				linesEuc.append(euc)
				linesIterations.append(round(radius / euc))
	numLines = len(linesDY)
	# evalulate every pixel
	height, width = elevation.shape
	view = np.zeros(elevation.shape, dtype=np.single)
	for y in range(height):
		for x in range(width):
			z = elevation[y, x]
			# find occlusions along each line
			viewSum = numLines
			for i in range(numLines):
				dy, dx = linesDY[i], linesDX[i]
				eucDelta = linesEuc[i]
				ly, lx = y, x
				dist = 0
				highZ = z
				highSlope = 0
				for r in range(linesIterations[i]):
					ly += dy
					if ly < 0 or ly >= height:
						break
					lx += dx
					if lx < 0 or lx >= width:
						break
					dist += eucDelta
					lz = elevation[ly, lx]
					if lz > highZ:
						highZ = lz
						slope = (lz - z) / dist
						if slope > highSlope:
							highSlope = slope
				if highSlope != 0:
					occlusion = math.atan(highSlope) * oneOverHalfPi
					viewSum -= occlusion
			view[y, x] = viewSum
	view /= numLines
	return view

def autocontrast(arr, maxValue):
	mult = maxValue / (arr.max() - arr.min())
	return (arr - arr.min()) * mult

def autocontrastedUint8(arr):
	return autocontrast(arr, 255).astype(np.uint8)

def getEOSDISlogin():
	if args.one_time_login:
		# get login input only for this session
		username = input('        EOSDIS username: ')
		password = getpass('        EOSDIS password: ')
		return username, password
	if args.new_login:
		# delete all stored login info
		comb = catacomb.Catacomb(storageDir, 'reset')
	else:
		comb = catacomb.Catacomb(storageDir)
	username = comb.decrypt('eosdis_username')
	if username is None:
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

def printDownloadStatus(tileDownloads, overwrite=True):
	global StatusPrintLock
	if StatusPrintLock:
		return
	StatusPrintLock = True
	if overwrite == True:
		for td in tileDownloads:
			stdout.write("\033[F")
	for td in tileDownloads:
		stdout.write("{} {} {}\n".format(td.locationCode, td.layer, td.status or ''))
	StatusPrintLock = False

def listFD(url, ext=''):
	from bs4 import BeautifulSoup
	response = requests.get(url)
	if response.status_code != 200:
		return []
	soup = BeautifulSoup(response.text, 'html.parser')
	return [url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]

def loadSRTMtileList():
	# get source of tile list if one hasn't been yet generated
	if not path.exists(storageDir + '/srtm_tile_list.json') and not path.exists(scriptDir + '/srtm_tile_list.json'):
		print('downloading SRTM tile list from https://dwtkns.com/srtm30m/srtm30m_bounding_boxes.json ...')
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
	if path.exists(storageDir + '/srtm_tile_list.json'):
		filePath = storageDir + '/srtm_tile_list.json'
	elif path.exists(scriptDir + '/srtm_tile_list.json'):
		filePath = scriptDir + '/srtm_tile_list.json'
	with open(filePath, "r") as read_file:
		locCodes = json.load(read_file)
	if not path.exists(storageDir + '/srtm_tile_list.json'):
		# copy tile list to storage directory
		with open(storageDir + '/srtm_tile_list.json', 'w') as write_file:
				json.dump(locCodes, write_file, indent='')
	return locCodes

def loadASTERtileList():
	# get source of tile list if one hasn't been yet generated
	if not path.exists(storageDir + '/aster_tile_list.json') and not path.exists(scriptDir + '/aster_tile_list.json'):
		print('downloading ASTER tile list from https://e4ftl01.cr.usgs.gov/ASTT/ASTGTM.003/2000.03.01/ ...')
		locCodes = {}
		for filename in listFD('https://e4ftl01.cr.usgs.gov/ASTT/ASTGTM.003/2000.03.01/', 'zip'):
			locCode = filename[-11:-4]
			locCodes[locCode] = True
		with open(storageDir + '/aster_tile_list.json', 'w') as write_file:
				json.dump(locCodes, write_file, indent='')
	# load tile list
	if path.exists(storageDir + '/aster_tile_list.json'):
		filePath = storageDir + '/aster_tile_list.json'
	elif path.exists(scriptDir + '/aster_tile_list.json'):
		filePath = scriptDir + '/aster_tile_list.json'
	with open(filePath, "r") as read_file:
		locCodes = json.load(read_file)
	if not path.exists(storageDir + '/aster_tile_list.json'):
		# copy tile list to storage directory
		with open(storageDir + '/aster_tile_list.json', 'w') as write_file:
				json.dump(locCodes, write_file, indent='')
	return locCodes

class TileDownload:

	zippedFilename = None
	status = None
	missing = False

	def __init__(self, baseUrl, locationCode, layer, baseZippedFilename, username, password, container):
		self.url = baseUrl.replace('^^^', locationCode)
		self.filepath = "{}/{}".format(storageDir, self.url.rsplit('/', 1)[-1])
		self.zippedFilename = baseZippedFilename.replace('^^^', locationCode)
		self.locationCode = locationCode
		self.layer = layer
		self.username = username
		self.password = password
		self.container = container

	def setStatus(self, newStatus):
		spaces = ' ' * max(0, len(self.status or '') - len(newStatus))
		self.status = newStatus + spaces
		printDownloadStatus(self.container.tileDownloads)

	def streamResponseWithStatus(self):
		if self.response is None:
			print("no response to stream")
			return
		totalkB = None
		if self.response.headers and self.response.headers.get('Content-Length'):
			totalkB = int(int(self.response.headers.get('Content-Length')) / 1024)
		kB = 0
		# content = b''
		write_file = open(self.filepath, 'wb')
		for chunk in self.response.iter_content(chunk_size=1024*10):
			kB += 10
			if not totalkB is None:
				kB = min(kB, totalkB)
				percent = int((kB / totalkB) * 100)
				self.setStatus("{}% ({:,} / {:,} kB)".format(percent, kB, totalkB))
			else:
				self.setStatus("{} kB".format(kB))
			write_file.write(chunk)
			# content += chunk
		self.fileSize = kB * 1024
		write_file.close()
		self.response = None # to reduce memory usage

	def downloadWithAuth(self):
		try:
			# response = requests.get(self.url, auth = requests.auth.HTTPBasicAuth(self.username, self.password), stream=True)
			response = requests.get(self.url)
		except requests.ConnectionError:
			self.setStatus('failed to get response')
		else:
			if response.status_code == 404:
				self.setStatus('404')
				self.missing = True
			elif response.status_code == 200:
				self.response = response
				self.streamResponseWithStatus()
			elif response.url:
				self.setStatus('redirected')
				attempt = 1
				response2 = None
				while response2 is None:
					try:
						response2 = requests.get(response.url, auth = requests.auth.HTTPBasicAuth(self.username, self.password), stream=True)
					except requests.ConnectionError:
						self.setStatus('retrying after {} attempt'.format(attempt))
						response2 = None
					attempt += 1
				if response2.status_code == 404:
					self.setStatus('404')
					self.missing = True
				else:
					self.setStatus(str(response2.status_code))
					self.response = response2
					self.streamResponseWithStatus()
			else:
				self.setStatus(str(self.status_code))
				self.missing = True
			if not self.missing:
				self.setStatus('downloaded {:,} kB'.format(int((self.fileSize or 0) / 1024)))

	def extractAndRead(self):
		if not path.exists(self.filepath):
			if self.missing:
				# create an array of zeros to fill in missing data
				dataType = bool
				if self.layer != 'waterbody':
					dataType = np.uint16
				return np.zeros((3601, 3601), dtype=dataType)
			else:
				return None
		with ZipFile(self.filepath, 'r') as zf:
			with zf.open(self.zippedFilename, mode='r') as zippedFile:
				ext = path.splitext(self.zippedFilename)[1].lower()
				raw = zippedFile.read()
				arr = None
				if ext == '.hgt':
					siz = len(raw)
					dim = int(math.sqrt(siz/2))
					arr = np.frombuffer(raw, np.dtype('>i2'), dim*dim).reshape((dim, dim))
				elif ext == '.raw':
					siz = len(raw)
					dim = int(math.sqrt(siz))
					arr = np.frombuffer(raw, np.uint8, dim*dim).reshape((dim, dim))
				elif ext == '.tif':
					arr = np.array(Image.open(BytesIO(raw)))
				if not arr is None and self.layer == 'waterbody':
					arr = arr > arr.min()
				return arr


class Compartment:

	def __init__(self, latitude, longitude, width, height):
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
		# print(lonMin, lonMax)
		# print(longitudeDistance(math.floor(lonMin), math.floor(lonMax)))
		codes = {}
		for lat in range(math.floor(latMax), math.floor(latMin)-1, -1):
				for lonAdd in range(0, longitudeDistance(math.floor(lonMin), math.floor(lonMax)) + 1):
					if lonMin < 0 and lonMax > 0:
						lon = math.floor(lonMax) + lonAdd
					else:
						lon = math.floor(lonMin) + lonAdd
					if lon > 179:
						lon -= 360
					codes[latLonToLocationCode(lat, lon)] = True
		# print(*codes)
		self.cropX1 = int((lonMin - math.floor(lonMin)) * 3601)
		self.cropX2 = self.cropX1 + cropWidth
		self.cropY1 = int((math.ceil(latMax) - latMax) * 3601)
		self.cropY2 = self.cropY1 + cropHeight
		self.metersPerPixel = xMeters / 3601
		self.metersPerPixelAfterResize = self.metersPerPixel / xMult
		self.xMult, self.yMult = xMult, yMult
		self.codes = [*codes]

	def report(self):
		print('{}\n({}, {}) ({}, {})\t{:.2f} m/px / {:.2f} m/px\t({:.3f}*x, {:.3f}*y)'.format(self.codes, self.cropX1, self.cropY1, self.cropX2, self.cropY2, self.metersPerPixel, self.metersPerPixelAfterResize, self.xMult, self.yMult))

	def downloadTilesConcurrently(self):
		print(" ")
		printDownloadStatus(self.tileDownloads, False)
		with ThreadPoolExecutor(max_workers=8) as executor:
			executor.map(downloadOneTile, self.tileDownloads)

	def download(self):
		tileDownloads = []
		for code in self.codes:
			if SRTMlocationCodes.get(code):
				url = 'https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11/^^^.SRTMGL1.hgt.zip'
				zipped = '^^^.hgt'
				wbd_url = 'https://e4ftl01.cr.usgs.gov/MEASURES/SRTMSWBD.003/2000.02.11/^^^.SRTMSWBD.raw.zip'
				wbd_zipped = '^^^.raw'
			elif ASTERlocationCodes.get(code):
				url = 'https://e4ftl01.cr.usgs.gov/ASTT/ASTGTM.003/2000.03.01/ASTGTMV003_^^^.zip'
				zipped = 'ASTGTMV003_^^^_dem.tif'
				wbd_url = 'https://e4ftl01.cr.usgs.gov/ASTT/ASTWBD.001/2000.03.01/ASTWBDV001_^^^.zip'
				wbd_zipped = 'ASTWBDV001_^^^_dem.tif'
			tileDownloads.append(TileDownload(url, code, 'elevation', zipped, username, password, self))
			tileDownloads.append(TileDownload(wbd_url, code, 'waterbody', wbd_zipped, username, password, self))
		self.tileDownloads = tileDownloads
		self.downloadTilesConcurrently()

	def arrangeTiles(self, selectedLayer):
		# get bounds
		latMin, latMax, lonMin, lonMax = None, None, None, None
		thisLayer = {}
		for td in self.tileDownloads:
			if td.layer == selectedLayer:
				lat, lon = parseLocationCode(td.locationCode)
				if thisLayer.get(lat) is None:
					thisLayer[lat] = {}
				thisLayer[lat][lon] = td.extractAndRead()
				if latMin is None or lat < latMin:
					latMin = lat
				if latMax is None or lat > latMax:
					latMax = lat
				if lonMin is None or lon < lonMin:
					lonMin = lon
				if lonMax is None or lon > lonMax:
					lonMax = lon
		# arrange tiles
		rows = None
		for lat in range(latMax, latMin-1, -1):
			row = None
			for lonAdd in range(0, longitudeDistance(lonMin, lonMax) + 1):
				if lonMin < 0 and lonMax > 0:
					lon = lonMax + lonAdd
				else:
					lon = lonMin + lonAdd
				if lon > 179:
					lon -= 360
				tileArr = thisLayer.get(lat).get(lon)
				if row is None:
					row = tileArr
				else:
					row = np.concatenate((row, tileArr), axis=1)
			if rows is None:
				rows = row
			else:
				rows = np.concatenate((rows, row), axis=0)
		return rows

	def downloadAndCrop(self, username, password):
		self.username, self.password = username, password
		# download and arrange tiles into images
		self.download()
		self.layers = {}
		for td in self.tileDownloads:
			self.layers[td.layer] = True
		for layer in self.layers.keys():
			arr = self.arrangeTiles(layer)
			print("{}: {}x{}, {:,} to {:,}".format(layer, arr.shape[0], arr.shape[1], arr.min(), arr.max()))
			arr = arr[self.cropY1:self.cropY2, self.cropX1:self.cropX2]
			print("{} cropped: {}x{}, {:,} to {:,}".format(layer, arr.shape[0], arr.shape[1], arr.min(), arr.max()))
			self.layers[layer] = arr

	def deleteDownloadedFiles(self):
		for td in self.tileDownloads:
			if path.exists(td.filepath):
				remove(td.filepath)

def downloadOneTile(tileDownload):
	tileDownload.downloadWithAuth()

def allFlat(arr):
	if arr is None:
		# print("array is none")
		return True
	if len(np.unique(arr)) == 1:
		# print("only 1 unique value in array")
		return True

def fractionAboveLevel(arr, level=None):
	if level is None:
		level = arr.min()
	fraction = np.count_nonzero(arr > level) / (arr.shape[0] * arr.shape[1])
	print(fraction, "above", level)
	return fraction

def pickHues(hueDeltaE):
	ah, bh, ch = None, None, None
	if not args.hues is None:
		if args.hues[0] > -1:
			ah = args.hues[0]
		if len(args.hues) > 1 and args.hues[1] > -1:
			bh = args.hues[1]
		if len(args.hues) > 2 and args.hues[2] > -1:
			ch = args.hues[2]
	if ah is None:
		if bh is None and ch is None:
			ah = random.randint(0, 359)
		elif bh is None:
			ah = nextHueByDeltaE([ch], hueDeltaE)
		elif ch is None:
			ah = nextHueByDeltaE([bh], hueDeltaE)
		else:
			ah = nextHueByDeltaE([bh,ch], hueDeltaE)
	if bh is None:
		if ch is None:
			bh = nextHueByDeltaE([ah], hueDeltaE)
		else:
			bh = nextHueByDeltaE([ah,ch], hueDeltaE)
	if ch is None:
		ch = nextHueByDeltaE([ah,bh], hueDeltaE)
	print('hues:', ah, bh, ch)
	return ah, bh, ch

def pickLightnesses():
	lightSpan = args.max_lightness - args.min_lightness
	lightInterval = lightSpan / 3
	darkMidLight = [
		random.uniform(args.min_lightness, args.min_lightness + lightInterval),
		random.uniform(args.min_lightness + lightInterval, args.max_lightness - lightInterval),
		random.uniform(args.max_lightness - lightInterval, args.max_lightness)]
	lOrders = [
		[0, 1, 2],
		[2, 0, 1],
		[1, 0, 2],
		[0, 2, 1]]
	lOrder = random.choice(lOrders)
	if not args.lightnesses is None:
		if len(args.lightnesses) == 1:
			lOrder = random.choice([lOrders[0], lOrders[3]])
		if len(args.lightnesses) == 2:
			lOrder = lOrders[0]
	ls = [darkMidLight[lOrder[0]], darkMidLight[lOrder[1]], darkMidLight[lOrder[2]]]
	if not args.lightnesses is None:
		if len(args.lightnesses) == 1:
			ls = [args.lightnesses[0], ls[1], ls[2]]
		elif len(args.lightnesses) == 2:
			ls = [args.lightnesses[0], args.lightnesses[1], ls[2]]
		elif len(args.lightnesses) == 3:
			ls = args.lightnesses
	print('lightnesses:', *ls)
	return ls

def pickChromas():
	chromas = []
	for cIndex in range(0, 3):
		if not args.chromas is None and len(args.chromas) > cIndex and args.chromas[cIndex] > -1:
			chromas.append(args.chromas[cIndex])
		else:
			# chromas.append((random.randint(0,1) == 1 and random.randint(0, args.max_chroma)) or args.max_chroma)
			chromas.append(random.uniform(args.min_chroma, args.max_chroma))
	print('chromas:', chromas)
	return chromas

def pickWaterColors(landColors, num=2):
	unpicked = landColors.copy()
	waterColors = []
	if not args.water_colors is None:
		# use specified water colors
		lch = []
		for component in args.water_colors:
			lch.append(component)
			if len(lch) == 3:
				rgb = lch_to_rgb(*lch)
				if rgb is None:
					rgb = highestChromaColor(lch[0], lch[2])
				waterColors.append(rgb)
				print('water color {}:'.format(len(waterColors)-1), waterColors[-1].convert('oklch'))
				lch = []
		return waterColors
	# pick water colors
	for n in range(num):
		index = random.randrange(len(unpicked))
		lcolor = unpicked.pop(index)
		allOthers = landColors.copy()
		for ci in range(len(allOthers)):
			c = allOthers[ci]
			if c == lcolor:
				allOthers.pop(ci)
				break
		color = highestChromaColor(lcolor.convert('oklch').l, random.choice(allOthers).convert('oklch').h, args.max_chroma)
		print('water color {}:'.format(n), color.convert('oklch'))
		waterColors.append(color)
	return waterColors

def saveInterpolationImage(interpolation, tifname):
	grad = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (256, 1))
	colorizedGrad = arrayColorizeWithInterpolation(grad, interpolation)
	Image.fromarray(colorizedGrad).save(storageDir + '/{}.tif'.format(tifname))

def highChromaGradient(lightnesses, chromas, hues, steps=256, colorfulness=1):
	a = highestChromaColor(lightnesses[0], hues[0], chromas[0])
	b = highestChromaColor(lightnesses[1], hues[1], chromas[1])
	c = highestChromaColor(lightnesses[2], hues[2], chromas[2])
	print(a.convert('oklch'))
	print(b.convert('oklch'))
	print(c.convert('oklch'))
	if colorfulness == 0:
		return a.interpolate([b, c], space='oklch'), a, b, c
	aChroma = (a.convert('oklch').c * (1 - colorfulness)) + (chromas[0] * colorfulness)
	bChroma = (b.convert('oklch').c * (1 - colorfulness)) + (chromas[1] * colorfulness)
	cChroma = (c.convert('oklch').c * (1 - colorfulness)) + (chromas[2] * colorfulness)
	# return a.interpolate([b, c], space='oklab'), a, b, c
	allSteps = a.steps([b, c], steps=steps, space='oklch')
	halfSteps = math.floor(steps / 2) - 1
	halfStepsHigh = halfSteps + 1
	highChromaSteps = []
	sIndex = 0
	for col in allSteps:
		lch = col.convert('oklch')
		if sIndex > halfSteps:
			mult = (sIndex - halfSteps) / halfStepsHigh
			chroma = ((1-mult) * bChroma) + (mult * cChroma)
		else:
			mult = sIndex / halfSteps
			chroma = ((1-mult) * aChroma) + (mult * bChroma)
		highChromaSteps.append(highestChromaColor(lch.l, lch.h, chroma))
		# print(sIndex, highChromaSteps[-1].convert('oklch'))
		sIndex += 1
	return highChromaSteps[0].interpolate(highChromaSteps[1:], space='oklch'), a, b, c

class TerrainCrop:

	shaded_color_elev = None
	color_map_with_waterbody = None
	rgba_waterbody = None
	light_map = None

	def __init__(self, elevation, waterbody, preRotatedWidth, preRotatedHeight, rotation, metersPerPixelAfterResize):
		self.elevation = elevation.astype(np.single)
		if waterbody is None or args.no_water or allFlat(waterbody):
			self.waterbody = None
		else:
			self.waterbody = waterbody
		self.preRotatedWidth, self.preRotatedHeight = preRotatedWidth, preRotatedHeight
		self.rotation = rotation
		self.metersPerPixelAfterResize = metersPerPixelAfterResize

	def clipToSeaLevel(self):
		# clip negative bits if this crop contains sea level
		if 0 in self.elevation and self.elevation.min() < 0:
			self.elevation = np.clip(self.elevation, 0, None)
			print("clipped to sea level", self.elevation.min(), self.elevation.max())

	def constrainWaterbodyToLowSlope(self):
		# restrict waterbody image to only those areas with low slope, so it doesn't cut off the hillshade
		if self.waterbody is None:
			return
		slope = slopeOfArray(self.elevation)
		self.waterbody = (slope < 0.8) & (self.waterbody > self.waterbody.min()).astype(np.uint8)

	def stretch(self):
		# stretch elevation and waterbody data
		self.elevation = resize(self.elevation, dsize=(self.preRotatedWidth, self.preRotatedHeight), interpolation=INTER_LINEAR)
		print("resized", self.elevation.shape, arr.min(), arr.max())
		if self.waterbody is None:
			return
		self.waterbody = resize(autocontrastedUint8(self.waterbody), dsize=(self.preRotatedWidth, self.preRotatedHeight), interpolation=INTER_LINEAR)

	def rotate(self):
		# rotate elevation and waterbody data
		if self.rotation == 0:
			return
		self.elevation = np.rot90(self.elevation, k=rotation)
		if not self.waterbody is None:
			self.waterbody = np.rot90(self.waterbody, k=rotation)
		print("rotated", self.elevation.shape, arr.min(), arr.max())

	def shade(self, azimuth=135, angle=45, ambientStrength=0.8):
		if args.no_shade:
			return
		# process elevation map for hillshading
		elevForShade = self.elevation / self.metersPerPixelAfterResize # so that the height map's vertical units are the same as its horizontal units
		print("for shade", elevForShade.min(), elevForShade.max())
		# hillshade layers: azimuth, altitude angle, opacity
		directHS = multiHillshade([[azimuth, angle, 1]], elevForShade)
		ambientHS = skyView(elevForShade, 8) * 0.5
		# Image.fromarray((directHS * 255).astype(np.uint8)).save(storageDir + '/hs-direct.tif')
		# Image.fromarray((ambientHS * 255).astype(np.uint8)).save(storageDir + '/hs-ambient.tif')
		# mix some of ambient into direct
		adjustedAmbi = ambientStrength * 0.5
		directHS = (directHS * (1 - adjustedAmbi)) + (ambientHS * adjustedAmbi)
		# Image.fromarray((directHS * 255).astype(np.uint8)).save(storageDir + '/hs-dblend.tif')
		if not args.no_shadows and ambientStrength < 1 and angle < 45:
			# draw light/shadow map
			lightMap = rayShadows(elevForShade, azimuth, angle)
			self.light_map = lightMap
			# use light/shadow map as mask for direct and ambient hillshades
			hs = ((lightMap/255) * directHS) + ((1 - (lightMap / 255)) * ambientStrength * ambientHS)
		else:
			hs = directHS
		self.hillshade = (hs * 255).astype(np.uint8)

	def colorizeElevation(self, interpolation):
		# half-equalize elevation data with a wide range
		if self.elevation.max() - self.elevation.min() > 100:
			elevForColor = (0.5 * image_histogram_equalization(self.elevation, 1024)) + (0.5 * autocontrast(self.elevation, 1023))
		else:
			elevForColor = self.elevation
		# colorize elevation data
		# self.color_elev = arrayColorizeWithInterpolation(elevForColor, interpolation, 1024)
		self.lch_color_elev = arrayColorizeWithInterpolation(elevForColor, interpolation, 1024, False, False, True)

	def overlayShade(self):
		# blend colorized elevation with hillshade using overlay
		if args.no_shade or self.hillshade is None:
			self.shaded_color_elev = OKLCH2RGB255(self.lch_color_elev)
		else:
			# self.shaded_color_elev = lightRGB(self.color_elev, self.hillshade, args.glow)
			self.shaded_color_elev = lightOklch(self.lch_color_elev, self.hillshade, args.glow)

	def superimposeWaterbody(self, waterColors):
		if args.no_water or self.waterbody is None:
			return
		bottomLayer = self.shaded_color_elev
		if bottomLayer is None:
			bottomLayer = self.color_elev
		# de-alias waterbody image and create alpha
		wbd_blur = self.waterbody # medianBlur(self.waterbody, 3)
		wbd_blur = GaussianBlur(wbd_blur, (3,3), 0.5, 0.5, BORDER_DEFAULT)
		wbd_float = wbd_blur / 255
		wbd_alpha = np.stack((wbd_float, wbd_float, wbd_float), axis=2)
		# create radial gradient image
		radgrad = radialGradient(self.waterbody.shape[1], self.waterbody.shape[0], args.azimuth, args.altitude_angle)
		# sort water colors by lightness descending
		waterColors.sort(reverse = True, key = lambda color: color.convert('oklch').l)
		waterInterpolation = waterColors[0].interpolate(waterColors[1:], space='oklab')
		steps = math.ceil(math.sqrt((min(*self.waterbody.shape) ** 2) * 2))
		wbd_radgrad = arrayColorizeWithInterpolation(radgrad, waterInterpolation, steps, False, True)
		wbd_radgrad = randomDitherImage(wbd_radgrad)
		monoColor = waterInterpolation(1)
		wbd_mono = np.zeros_like(wbd_radgrad)
		wbd_mono[:, :] = [int(monoColor.r * 255), int(monoColor.g * 255), int(monoColor.b * 255)]
		if not self.light_map is None:
			# cast shadows on water
			# lm = (self.light_map * 0.5 * (1 - args.ambient_strength)) + (args.ambient_strength * 127)
			lm = self.light_map / 255
			lm = np.stack((lm, lm, lm), axis=2)
			wbd_radgrad = ((lm * wbd_radgrad) + ((1 - lm) * wbd_mono)).astype(np.uint8)
		# superimpose radial gradient with waterbody alpha
		self.color_map_with_waterbody = ( (wbd_radgrad * wbd_alpha) + (bottomLayer * (1 - wbd_alpha)) ).astype(np.uint8)
		if args.output is None:
			# create an RGBA image of the color waterbody data to save
			self.rgba_waterbody = np.zeros((*wbd_float.shape, 4), dtype=np.uint8)
			self.rgba_waterbody = np.stack((wbd_radgrad[:,:,0], wbd_radgrad[:,:,1], wbd_radgrad[:,:,2], wbd_blur), axis=2)

	def saveImages(self):
		final_arr = self.color_map_with_waterbody
		if final_arr is None:
			final_arr = self.shaded_color_elev
		if final_arr is None:
			final_arr = self.color_elev
		final_img = Image.fromarray(final_arr)
		if not args.output is None:
			if path.exists(path.split(args.output)[0]):
				ext = path.splitext(args.output)[1].lower()
				if ext == '':
					final_img.save(args.output, format='PNG')
				elif ext == '.jpg' or ext == '.jpeg':
					final_img.save(args.output, quality=95)
				else:
					final_img.save(args.output)
				print('saved', args.output)
		else:
			if not args.no_shade:
				Image.fromarray(self.hillshade).save(storageDir + '/hillshade.tif')
				print('saved', storageDir + '/hillshade.tif')
			if not args.no_water and not self.rgba_waterbody is None:
				Image.fromarray(self.rgba_waterbody).save(storageDir + '/water.tif')
				print('saved', storageDir + '/water.tif')
			if not args.no_shadows and not self.light_map is None:
				Image.fromarray(self.light_map).save(storageDir + '/light.tif')
				print('saved', storageDir + '/light.tif')
			# Image.fromarray(self.color_elev).save(storageDir + '/elevation_gradient.tif')
			# print('saved', storageDir + '/elevation_gradient.tif')
			final_img.save(storageDir + '/output.png')
			print('saved', storageDir + '/output.png')

	def processData(self):
		self.clipToSeaLevel()
		self.constrainWaterbodyToLowSlope()
		self.stretch()
		self.rotate()
		self.lightnesses = pickLightnesses()
		self.chromas = pickChromas()
		self.hues = pickHues(args.hue_delta)
		interpolation, a, b, c = highChromaGradient(self.lightnesses, self.chromas, self.hues, 16, args.colorfulness)
		# saveInterpolationImage(interpolation, 'hcg')
		# saveInterpolationImage(a.interpolate([b, c], space='oklch'), 'oklchg')
		self.colorizeElevation(interpolation)
		self.shade(args.azimuth, args.altitude_angle, args.ambient_strength)
		self.overlayShade()
		if not args.no_water:
			self.waterColors = pickWaterColors([a, b, c])
			self.superimposeWaterbody(self.waterColors)

	def colorDict(self):
		# output dictionary of color information
		wcLCH = []
		if not args.no_water:
			for color in self.waterColors:
				lch = color.convert('oklch')
				wcLCH.append(lch.l)
				wcLCH.append(lch.c)
				wcLCH.append(lch.h)
		out = {
			'lightnesses' : self.lightnesses,
			'chromas' : self.chromas,
			'hues' : self.hues,
			'water_colors' : wcLCH,
			'colorfulness' : args.colorfulness}
		return out

	def lightDict(self):
		# output dictionary of light information
		out = {
			'glow' : args.glow,
			'shine' : args.shine,
			'azimuth' : args.azimuth,
			'altitude_angle' : args.altitude_angle,
			'ambient_strength' : args.ambient_strength}
		return out

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
	parser = ArgumentParser(description='Create a colorful image of terrain of a random location.')
	parser.add_argument('--new-login', action='store_true', default=False, help='Enter an Earthdata username & password and store it encrypted for future use. Overwrites currently stored login information if any.')
	parser.add_argument('--one-time-login', action='store_true', default=False, help='Enter an Earthdata username & password to use only for this run, and do not store it.')
	parser.add_argument('--previous', '-p', action='store_true', default=False, help='Use previously downloaded data. --dimensions, --coordinates, and --rotation will have no effect.')
	parser.add_argument('--previous-colors', action='store_true', default=False, help='Use previously used colors. Any additional color arguments will override those specific parts of the previously used colors.')
	parser.add_argument('--previous-light', action='store_true', default=False, help='Use previously used lighting. Any additional lighting arguments will override those specific parts of the previously used lighting.')
	parser.add_argument('--no-water', '-w', action='store_true', default=False, help='Do not draw bodies of water.')
	parser.add_argument('--no-shade', '-s', action='store_true', default=False, help='Do not hillshade the terrain. This leaves only gradient-mapped elevations and water bodies.')
	parser.add_argument('--no-shadows', action='store_true', default=False, help='Do not cast shadows.')
	parser.add_argument('--output', '-o', nargs='?', type=str, metavar='FILEPATH', help='Path to save output image. If not specified, will save to ~/a_sort_of_soup/output.png along with elevation_gradient.tif, hillshade.tif, and water.tif')
	parser.add_argument('--coordinates', '-c', nargs=2, type=float, metavar=('LATITUDE', 'LONGITUDE'), help='Location of center of desired image in latitude longitude coordinates. If not specified, a random location will be chosen.')
	parser.add_argument('--dimensions', '-d', nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'), help='Width and height in pixels of output image. Larger images will require downloading more source tiles. Defaults to screen dimensions.')
	parser.add_argument('--rotation', '-r', nargs='?', type=int, metavar='0-3', help='How many times 90 degrees to rotate. (0: North is up. 1: East is up. 2: South is up. 3: West is up.) If not specified, this will be chosen randomly.')
	parser.add_argument('--min-lightness', nargs='?', type=float, default=0.05, metavar='0-1', help='Unless specified by --lightnesses, lightnesses will be randomly chosen between --min-lightness and --max-lightness.')
	parser.add_argument('--max-lightness', nargs='?', type=float, default=0.95, metavar='0-1', help='Unless specified by --lightnesses, lightnesses will be randomly chosen between --min-lightness and --max-lightness.')
	parser.add_argument('--min-chroma', nargs='?', type=float, default=0, metavar='0-0.4', help='Attempt to choose colors with at least this minimum chromaticity.')
	parser.add_argument('--max-chroma', nargs='?', type=float, default=0.32, metavar='0-0.4', help='Maximum chromaticity of image.')
	parser.add_argument('--hue-delta', nargs='?', type=int, metavar='Delta-E', help='Minimum color difference between hues as calculated by CIE Delta-E 2000 at 0.74 lightness and 0.12 chromaticity. Values over 35 will usually cause Delta-E between hues to be uneven. If not specified, this will be chosen randomly from 20 through 40.')
	parser.add_argument('--lightnesses', nargs='+', type=float, metavar='0-1', help='Up to three lightnesses, in order of elevation. The remaining lightnesses will be chosen randomly.')
	parser.add_argument('--chromas', nargs='+', type=float, metavar='0-0.4', help='Up to three chromaticities, in order of elevation. The remaining chromas will be chosen randomly. To specify only the second and/or third chromaticities, enter chromaticities of -1 to have them chosen randomly.')
	parser.add_argument('--hues', nargs='+', type=int, metavar='0-359', help='Up to three hues, in order of elevation. The remaining hues will be chosen randomly. To specify only the second and/or third hue, enter hues of -1 to have them chosen randomly.')
	parser.add_argument('--water-colors', nargs='+', type=float, metavar='L C H', help='Any number of colors for the gradient to fill water bodies with, formatted in a flat list of Lightness Chroma Hue triplets.')
	parser.add_argument('--colorfulness', nargs='?', type=float, default=1, metavar='0-1', help='0 creates a gradient in OKLCH space between the three in-sRGB-gamut colors chosen. 1 finds the highest chroma colors possible at or below the *specified* three chromas. 1 by default.')
	parser.add_argument('--shine', nargs='?', type=float, default=0, metavar='0-1', help='Intensity of hillshade highlights. 0 by default.')
	parser.add_argument('--glow', nargs='?', type=float, default=0, metavar='0-1', help='Opacity of overlay and transparency of hard light blending of hillshade. 0 by default.')
	parser.add_argument('--azimuth', nargs='?', type=int, metavar='0-359', help='Azimuth of sunlight for hillshade and shadows (in 45-degree increments). If not specified, will be a random number from 0 through 180.')
	parser.add_argument('--altitude-angle', nargs='?', type=int, metavar='1-90', help='Altitude angle of sunlight for hillshade and shadows (in degrees). If not specified, this will be a random number from 7 through 45.')
	parser.add_argument('--ambient-strength', nargs='?', type=float, metavar='0-1', help='Strength of diffuse light in hillshade, and inverse of the darkness of cast shadows. If not specified, this will be a random number from 0.75 through 1.00.')
	return parser.parse_args()

args = parseArguments()

# use previous colors if asked for
if args.previous_colors and path.exists(storageDir + '/previous-colors.json'):
		with open(storageDir + '/previous-colors.json', "r") as read_file:
			prevColors = json.load(read_file)
			for k in prevColors.keys():
				if getattr(args, k) is None:
					setattr(args, k, prevColors.get(k))
# use previous lighting if asked for
if args.previous_light and path.exists(storageDir + '/previous-light.json'):
		with open(storageDir + '/previous-light.json', "r") as read_file:
			prevLight = json.load(read_file)
			for k in prevLight.keys():
				if getattr(args, k) is None:
					setattr(args, k, prevLight.get(k))

if args.hue_delta is None:
	args.hue_delta = random.randint(20, 40)
if args.azimuth is None:
	args.azimuth = random.randint(0, 180)
if args.altitude_angle is None:
	args.altitude_angle = random.randint(7, 45)
if args.ambient_strength is None:
	args.ambient_strength = 0.75 + (random.random() * random.random() * 0.25)

reportString = ''
for name in ['hue_delta', 'azimuth', 'altitude_angle', 'ambient_strength']:
	nameConv = name.replace('_', '-')
	val = getattr(args, name)
	if val > 0 and abs(val) < 1:
		reportString += ' --{} {:.2f}'.format(nameConv, val)
	else:
		reportString += ' --{} {}'.format(nameConv, val)
print(reportString)

if not path.exists(storageDir):
	makedirs(storageDir)

SRTMlocationCodes = loadSRTMtileList()
ASTERlocationCodes = loadASTERtileList()

# get screen dimensions
screenWidth, screenHeight = None, None
for m in get_monitors():
	if screenWidth == None or m.width > screenWidth:
		screenWidth = m.width
	if screenHeight == None or m.height > screenHeight:
		screenHeight = m.height

if args.previous and path.exists(storageDir + '/previous.json'):
	# get previous info if asked for and exists
	with open(storageDir + '/previous.json', "r") as read_file:
		previousInfo = json.load(read_file)
		args.rotation = int(previousInfo.get('rotation'))
		args.coordinates = [float(previousInfo.get('latitude')), float(previousInfo.get('longitude'))]
		args.dimensions = [int(previousInfo.get('width')), int(previousInfo.get('height'))]

if not args.dimensions is None:
	targetWidth, targetHeight = args.dimensions[0], args.dimensions[1]
else:
	targetWidth, targetHeight = screenWidth, screenHeight
print("output dimensions:", targetWidth, targetHeight)

# get encrypted EOSDIS login info
username, password = getEOSDISlogin()

downloadCropAttempt = 0
arr, wbd_arr = None, None
while downloadCropAttempt < 20 and (arr is None or wbd_arr is None or allFlat(arr) or fractionAboveLevel(wbd_arr) > 0.9):
	if not args.coordinates is None and downloadCropAttempt > 0:
		print("specified coordinates {:.5f} {:.5f} with output dimensions {}x{} and rotation {} contain insufficient land area".format(latitude, longitude, targetWidth, targetHeight, rotation))
		exit()
	downloadCompartment = None
	attempt = 0
	# find coordinates that are within SRTM and ASTER data
	while attempt < 50 and (downloadCompartment is None or not checkOutLocationCodes(downloadCompartment.codes)):
		if attempt > 0 and not args.coordinates is None:
			print("no tiles or insufficient tiles for specified coordinates {:.5f} {:.5f} with output dimensions {}x{} and rotation {}".format(latitude, longitude, targetWidth, targetHeight, rotation))
			exit()
		# rotate to get pre-rotated target dimensions
		if not args.rotation is None:
			rotation = args.rotation
		else:
			rotation = random.randint(0, 3)
		if rotation == 1 or rotation == 3:
			preRotatedWidth, preRotatedHeight = targetHeight, targetWidth
		else:
			preRotatedWidth, preRotatedHeight = targetWidth, targetHeight
		# pick a random location or use specified coordinates
		if not args.coordinates is None and attempt == 0 and downloadCropAttempt == 0:
			latitude, longitude = args.coordinates[0], args.coordinates[1]
		else:
			latitude, longitude = uniformlyRandomLatLon()
		downloadCompartment = Compartment(latitude, longitude, preRotatedWidth, preRotatedHeight)
		attempt += 1
	print("\n-c {:.5f} {:.5f} -r {}".format(latitude, longitude, rotation))
	downloadCompartment.report()
	if args.previous:
		# use previously downloaded cropped images
		arr = np.array(Image.open(storageDir + '/elevation-cropped.tif'))
		wbd_arr = np.array(Image.open(storageDir + '/waterbody-cropped.tif'))
	else:
		# download and arrange tiles into images
		downloadCompartment.downloadAndCrop(username, password)
		arr = downloadCompartment.layers.get('elevation')
		wbd_arr = downloadCompartment.layers.get('waterbody')
		downloadCompartment.deleteDownloadedFiles()
	downloadCropAttempt += 1

if not args.previous:
	# save images for use as previous
	Image.fromarray(arr).save(storageDir + '/elevation-cropped.tif')
	Image.fromarray(wbd_arr).save(storageDir + '/waterbody-cropped.tif')
	# save information for use as previous
	previousInfo = {'latitude':latitude, 'longitude':longitude, 'rotation':rotation, 'width':targetWidth, 'height':targetHeight}
	with open(storageDir + '/previous.json', 'w') as write_file:
		json.dump(previousInfo, write_file, indent='')

thisCrop = TerrainCrop(arr, wbd_arr, preRotatedWidth, preRotatedHeight, rotation, downloadCompartment.metersPerPixelAfterResize)
downloadCompartment = None # reduce memory usage
thisCrop.processData()
thisCrop.saveImages()

# save colors for use as previous
with open(storageDir + '/previous-colors.json', 'w') as write_file:
	json.dump(thisCrop.colorDict(), write_file, indent="\t")
# save lighting for use as previous
with open(storageDir + '/previous-light.json', 'w') as write_file:
	json.dump(thisCrop.lightDict(), write_file, indent="\t")