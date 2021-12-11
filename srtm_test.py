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
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import math
from zipfile import ZipFile
import coloraide
from blend_modes import multiply, overlay
from pw_crypt import WeAreNeagan
from getpass import getpass
import requests
from requests.auth import HTTPBasicAuth
from io import BytesIO
import random
import sys
import json
import datetime

degreesPerTheta = 90 / (math.pi / 2)
maxChroma = 134

CurrentGrade = None

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
    while iteration < 35:
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
    cdf = (number_bins - 1) * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape)

def hillshade(array, azimuth, angle_altitude):
    azimuth = 360.0 - azimuth
    x, y = gradient(array)
    slope = pi/2. - arctan(sqrt(x*x + y*y))
    aspect = arctan2(-x, y)
    azimuthrad = azimuth*pi / 180.
    altituderad = angle_altitude*pi / 180.
    shaded = sin(altituderad) * sin(slope)\
     + cos(altituderad) * cos(slope)\
     * cos((azimuthrad - pi/2.) - aspect)
    return 255*(shaded + 1)/2

def shadow(array, azimuth, angle_altitude):
    azimuth = 360.0 - azimuth

def autocontrastedUint8(arr):
    divisor = (arr.max() - arr.min()) / 255
    return ((arr - arr.min()) / divisor).astype(np.uint8)

def downloadHGT(url, un, pw):
    print(url)
    response = requests.get(url, auth = HTTPBasicAuth(un, pw), headers = {'user-agent': 'Firefox'})
    print(response.url)
    response2 = requests.get(response.url, auth = HTTPBasicAuth(un, pw), headers = {'user-agent': 'Firefox'}, stream=True)
    totalkB = int(int(response2.headers.get('Content-Length')) / 1024)
    kB = 0
    content = b''
    for chunk in response2.iter_content(chunk_size=1024*10):
        kB = min(kB + 10, totalkB)
        percent = int((kB / totalkB) * 100)
        sys.stdout.write("\r{}% ({} / {} kB)".format(percent, kB, totalkB))
        sys.stdout.flush()
        content += chunk
    print(" ")
    return content

def measureLatLonInMeters(lat1, lon1, lat2, lon2):
    R = 6378.137 # Radius of earth in KM
    dLat = lat2 * math.pi / 180 - lat1 * math.pi / 180
    dLon = lon2 * math.pi / 180 - lon1 * math.pi / 180
    a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(lat1 * math.pi / 180) * math.cos(lat2 * math.pi / 180) * math.sin(dLon/2) * math.sin(dLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c
    return d * 1000

# load tile list
locCodes = {}
# https://dwtkns.com/srtm30m/srtm30m_bounding_boxes.json
# with open("srtm30m_bounding_boxes.json", "r") as read_file:
#     collection = json.load(read_file)
#     for f in collection.get('features'):
#         locCode = f.get('properties').get('dataFile').split('.')[0]
#         locCodes[locCode] = True
# with open('tile_list.json', 'w') as write_file:
#     json.dump(locCodes, write_file, indent='')
with open("tile_list.json", "r") as read_file:
    locCodes = json.load(read_file)

os.chdir(os.path.expanduser('~/color_out_of_earth'))

# get EOSDIS login
Crypt = WeAreNeagan()
username = Crypt.decrypt('eosdis_username')
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
    Crypt.encrypt('eosdis_username', username)
    Crypt.encrypt(username, password)
else:
    username = username.decode('utf-8')
    password = Crypt.decrypt(username).decode('utf-8')

test = None
attempt = 0
while test is None:
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
    test = locCodes.get(locationCode)
    attempt += 1
print(latitude, longitude)

zip_data = downloadHGT('http://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11/{}.SRTMGL1.hgt.zip'.format(locationCode), username, password)
# out_file = open(os.path.expanduser('~/color_out_of_earth/{}.SRTMGL1.hgt.zip'.format(locationCode)), "wb")
# out_file.write(zip_data)
# out_file.close()

# from 42 N to 43 N, 123 W to 122 W

xMeters = measureLatLonInMeters(latitude + 0.5, longitude, latitude + 0.5, longitude + 1)
yMeters = measureLatLonInMeters(latitude, longitude + 0.5, latitude + 1, longitude + 0.5)
yMult = yMeters / xMeters
print(yMult, xMeters, yMeters)

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

# zip_file_name = os.path.expanduser('~/color_out_of_earth/{}.SRTMGL1.hgt.zip'.format(locationCode))
with ZipFile(BytesIO(zip_data), 'r') as zip:
    # printing all the contents of the zip file
    with zip.open('{}.hgt'.format(locationCode), mode='r') as hgtFile:
        # print(type(hgtFile.read()))
        raw = hgtFile.read()
        siz = len(raw)
        dim = int(math.sqrt(siz/2))
        arr = np.frombuffer(raw, np.dtype('>i2'), dim*dim).reshape((dim, dim))

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
print(arr.shape)

# equalize elevation data
arr_orig = arr
arr = image_histogram_equalization(arr)

# convert elevation data to 256-color grayscale image
el_img = Image.fromarray(autocontrastedUint8(arr))
print(el_img.mode, len(el_img.getcolors()))
el_img = el_img.resize((el_img.size[0], int(el_img.size[1] * yMult)), resample=Image.BICUBIC)
el_img = rotateImage(el_img, rotation)
print(el_img.size)

# colorize elevation data
ah = random.randint(0,359)
bh = ah
while angleDist(ah, bh) < 90 or angleDist(ah, bh) > 120:
    bh = random.randint(0, 359)
print("hues", ah, bh)
a = highestChromaColor(35, ah)
b = highestChromaColor(85, bh)
i = b.interpolate(a, space='lch-d65')
color_el_img = colorizeWithInterpolation(el_img, i)

# create hillshade
angle = (315 + (rotation * 90)) % 360
# hs_array = autocontrastedUint8(hillshade(arr_orig, 315, 45))
# hs_array = autocontrastedUint8(image_histogram_equalization(hillshade(arr_orig, 315, 90)))
# hs_array = autocontrastedUint8(hillshade(arr_orig, 315, 90))
hs_array = hillshade(arr_orig, angle, 30).astype(np.uint8)
hs_img = Image.fromarray(hs_array)
hs_img = hs_img.resize((hs_img.size[0], int(hs_img.size[1] * yMult)), resample=Image.BICUBIC)
hs_img = rotateImage(hs_img, rotation)
print(hs_img.mode, len(hs_img.getcolors()))

# colorize hillshade
aah = (min(ah,bh)-80)%360
bbh = (max(ah,bh)+80)%360
print("hues", aah, bbh)
aa = highestChromaColor(25, aah)
bb = highestChromaColor(75, bbh)
ii = aa.interpolate(bb, space='lch-d65')
color_hs_img = colorizeWithInterpolation(hs_img, ii)

# blend colorized elevation with hillshade
color_hs_img = color_hs_img.convert('RGBA')
color_el_arr = np.array(color_el_img.convert('RGBA'))
hs_arr = np.array(color_hs_img)
# hs_arr = np.array(hs_img.convert('RGBA'))
blended_float = multiply(color_el_arr.astype(float), hs_arr.astype(float), 0.5)
blended_arr = np.uint8(blended_float)
blended_img = Image.fromarray(blended_arr)

# save images
blended_img.save(os.path.expanduser('~/color_out_of_earth/blended_img.png'))
el_img.save(os.path.expanduser('~/color_out_of_earth/el_img.png'))
hs_img.save(os.path.expanduser('~/color_out_of_earth/hs_img.png'))
color_el_img.save(os.path.expanduser('~/color_out_of_earth/color_el_img.png'))
color_hs_img.save(os.path.expanduser('~/color_out_of_earth/color_hs_img.png'))

# blended_img.show()