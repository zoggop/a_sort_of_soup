# A Sort of Soup
### (The Colour Out of Earth)

> You see it is so easy for you sitting in Tavistock Square to look inward; but I find it very difficult to look inward when I am also looking at the coast of Sinai; and very difficult to look at the coast of Sinai when I am also looking inward and finding the image of Virginia everywhere. So this combination makes my letter more dumb than usual.

> You manage things better. You have a more tidily sorted mind. You have a little compartment for the Press, and another little compartment for Mary Hutchinison, and another for Vita, and another for Dog Grizzle, and another for the Downs, and another for London fogs, and another for the Prince of Wales, and another for the lighthouse, no, I'm wrong, the lighthouse is allowed to play its beam over the whole lot, and their only Common Denominator is your own excitability over whichever compartment you choose to look into at the moment. But with me they all run together into a sort of soup.

> -- Vita Sackville-West to Virginia Woolf, 4 February, 1926

This is a python script that downloads a digital elevation model of a random location on earth, from which it produces a randomly-colored shaded terrain map. It gets its data from [SRTM](https://lpdaac.usgs.gov/products/srtmgl1v003/) and [ASTER](https://lpdaac.usgs.gov/products/astgtmv003/) 1 arc second DEMs and [Water](https://lpdaac.usgs.gov/products/srtmswbdv003/) [Body](https://lpdaac.usgs.gov/products/astwbdv001/) Databases. To access these data, the script needs an [Earthdata EOSDIS login](https://urs.earthdata.nasa.gov/users/new), that it will store [encrypted](https://github.com/zoggop/a-sort-of-soup/blob/main/catacomb.py).

## Requirements

- Python 3.9 (Other versions of Python 3 may work, but I haven't tested them.)

#### Python Modules:

- [coloraide](https://facelessuser.github.io/coloraide/)

- [Pillow](https://python-pillow.org/)

- [blend_modes](https://github.com/flrs/blend_modes)

- [opencv-python](https://pypi.org/project/opencv-python/)

- [beautifulsoup4](https://pypi.org/project/beautifulsoup4/)

- [screeninfo](https://pypi.org/project/screeninfo/)

## Examples

![Example Terrain Map 1](example1.jpg "a lake of lava")

![Example Terrain Map 2](example2.jpg "a triangular artifact in the ASTER DEM")

![Example Terrain Map 3](example3.jpg "God's oily fingerprints")