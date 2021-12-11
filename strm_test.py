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
from PIL import Image, ImageOps, ImageFilter
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

degreesPerTheta = 90 / (math.pi / 2)

count = 0
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

def gradeFunc(v):
    global count
    count += 1
    try:
        return CurrentGrade[v]
    except:
        print("grade failed", v, count)

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

def autocontrastedUint8(arr):
    divisor = (arr.max() - arr.min()) / 255
    return ((arr - arr.min()) / divisor).astype(np.uint8)

def downloadHGT(url, un, pw):
    print(url)
    response = requests.get(url, auth = HTTPBasicAuth(un, pw), headers = {'user-agent': 'Firefox'})
    print(len(response.content))
    print(response.text)
    print(response.url)
    print(response.status_code)
    print(response.history)
    response2 = requests.get(response.url, auth = HTTPBasicAuth(un, pw), headers = {'user-agent': 'Firefox'})
    # print(len(response2.content))
    return response2.content

def measureLatLonInMeters(lat1, lon1, lat2, lon2):
    R = 6378.137 # Radius of earth in KM
    dLat = lat2 * math.pi / 180 - lat1 * math.pi / 180
    dLon = lon2 * math.pi / 180 - lon1 * math.pi / 180
    a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(lat1 * math.pi / 180) * math.cos(lat2 * math.pi / 180) * math.sin(dLon/2) * math.sin(dLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c
    return d * 1000

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

# https://dwtkns.com/srtm30m/srtm30m_bounding_boxes.json


test = None
attempt = 0
while test is None or test.status_code != 200:
    if attempt == 0 and len(sys.argv) > 2:
        latitude, longitude = sys.argv[1], sys.argv[2]
    else:
        latitude, longitude = uniformlyRandomIntLatLon(-56, 59)
    latitude = int(latitude)
    longitude = int(longitude)
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
    test = requests.head('https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11/{}.SRTMGL1.2.jpg'.format(locationCode))
    print(locationCode, test.status_code)
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
print(yMult)

screenWidth = 1920
screenHeight = 1080

# zip_file_name = os.path.expanduser('~/color_out_of_earth/{}.SRTMGL1.hgt.zip'.format(locationCode))
with ZipFile(BytesIO(zip_data), 'r') as zip:
    # printing all the contents of the zip file
    with zip.open('{}.hgt'.format(locationCode), mode='r') as hgtFile:
        # print(type(hgtFile.read()))
        raw = hgtFile.read()
        siz = len(raw)
        dim = int(math.sqrt(siz/2))
        arr = np.frombuffer(raw, np.dtype('>i2'), dim*dim).reshape((dim, dim))

# get a random crop
xMax = arr.shape[1] - screenWidth
yMax = arr.shape[0] - screenHeight
x = random.randint(0, xMax)
y = random.randint(0, yMax)
x2 = x + screenWidth
y2 = math.ceil(y + (screenHeight / yMult))
arr = arr[y:y2, x:x2]
print(arr.shape)

# convert elevation data to 256-color grayscale image
divisor = (arr.max() - arr.min()) / 255
arrInt8 = ((arr - arr.min()) / divisor).astype(np.uint8)
el_img = Image.fromarray(arrInt8)
print(el_img.mode, len(el_img.getcolors()))
el_img = el_img.resize((el_img.size[0], int(el_img.size[1] * yMult)), resample=Image.BICUBIC)
print(el_img.size)
el_img.save(os.path.expanduser('~/color_out_of_earth/el_img.png'))

# colorize elevation data
# a = coloraide.Color('lch-d65', [50, 50, 0]).convert('srgb')
# b = coloraide.Color('lch-d65', [50, 50, 85]).convert('srgb')
a = coloraide.Color('lch-d65', [25, 17, 0]).convert('srgb')
b = coloraide.Color('lch-d65', [75, 38, 85]).convert('srgb')
i = b.interpolate(a, space='lch-d65')
color_el_img = colorizeWithInterpolation(el_img, i)

hs_array = autocontrastedUint8(hillshade(arr,315, 45))
hs_img = Image.fromarray(hs_array)
hs_img = hs_img.resize((hs_img.size[0], int(hs_img.size[1] * yMult)), resample=Image.BICUBIC)
print(hs_img.mode, len(hs_img.getcolors()))
aa = coloraide.Color('lch-d65', [25, 17, 0]).convert('srgb')
bb = coloraide.Color('lch-d65', [75, 38, 85]).convert('srgb')
ii = aa.interpolate(bb, space='lch-d65')
color_hs_img = colorizeWithInterpolation(hs_img, ii)
color_hs_img = color_hs_img.convert('RGBA')
# color_hs_img.show()
color_el_arr = np.array(color_el_img.convert('RGBA'))
hs_arr = np.array(color_hs_img)
blended_float = overlay(color_el_arr.astype(float), hs_arr.astype(float), 1)
blended_arr = np.uint8(blended_float)
blended_img = Image.fromarray(blended_arr)
hs_img.save(os.path.expanduser('~/color_out_of_earth/hs_img.png'))
color_el_img.save(os.path.expanduser('~/color_out_of_earth/color_el_img.png'))
blended_img.save(os.path.expanduser('~/color_out_of_earth/blended_img.png'))

# blended_img.show()