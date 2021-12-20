import geocoder
import pycountry
from googletrans import Translator, constants
import sys

locationNameStructure = [
		['city', 'locality', 'municipality', 'town', 'village'],
		['county', 'admin_2', 'subregion'],
		['state', 'admin_1', 'region', 'territory'],
		['country'],
		['longlabel', 'match_addr', 'address']] # fallbacks

def nameFromData(s, languageCode=None):
	if languageCode:
		translator = Translator()
	output = ''
	if s.get('country') == None and s.get('countrycode') != None:
		s['country'] = pycountry.countries.get(alpha_3=s.get('countrycode')).name
	strings = {}
	haveAnything = False
	l = 0
	for level in locationNameStructure:
		l += 1
		if l == 5 and output != '':
			break
		for k in level:
			v = s.get(k)
			if not v is None and v != '':
				if languageCode:
					transV = translator.translate(v, dest=languageCode)
					if not transV is None and transV.text != '':
						v = transV.text
				if strings.get(v) is None:
					if haveAnything:
						output = output + ', '
					output = output + v
					strings[v] = True
					haveAnything = True
					break
	if output == '':
		return None
	else:
		return output

def nameLatitudeLongitude(latitude, longitude, languageCode='en'):
	if languageCode.lower() == 'none':
		languageCode = None
	s = {} # data dict
	latLon = [latitude, longitude]
	for provider in ['arcgis', 'osm', 'geocodefarm']:
		func = getattr(geocoder, provider)
		try:
			g = func(latLon, method='reverse')
		except:
			print("could not get location with", provider)
		finally:
			print(provider, g)
			if not g.address is None and g.address != '':
				if s.get('address') is None:
					s['address'] = g.address
			# print(provider)
			if g.raw != None:
				a = g.raw.get('address') or g.raw.get('ADDRESS')
				if a != None:
					address = {}
					for k in a.keys():
						address[k.lower()] = a.get(k)
					# print(address)
					for level in locationNameStructure:
						for k in level:
							v = address.get(k)
							if not v is None and v != '':
								if s.get(k) is None:
									s[k] = v
	name = nameFromData(s)
	if languageCode:
		transName = nameFromData(s, languageCode)
		if name != transName and transName and transName != '':
			return transName
	return name

if len(sys.argv) > 3:
	print(nameLatitudeLongitude(float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]))
elif len(sys.argv) > 2:
	print(nameLatitudeLongitude(float(sys.argv[1]), float(sys.argv[2])))