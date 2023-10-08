# A Sort of Soup
### or, The Colour Out of Earth

> You have a little compartment for the Press, and another little compartment for Mary Hutchinison, and another for Vita, and another for Dog Grizzle, and another for the Downs, and another for London fogs, and another for the Prince of Wales, and another for the lighthouse, no, I'm wrong, the lighthouse is allowed to play its beam over the whole lot, and their only Common Denominator is your own excitability over whichever compartment you choose to look into at the moment. But with me they all run together into a sort of soup.

> — Vita Sackville-West to Virginia Woolf, 4 February, 1926

This is a python script that downloads a digital elevation model of a random location on earth, from which it produces a randomly-colored shaded terrain map. It gets its data from [SRTM](https://lpdaac.usgs.gov/products/srtmgl1v003/) and [ASTER](https://lpdaac.usgs.gov/products/astgtmv003/) 1 arc second DEMs and [Water](https://lpdaac.usgs.gov/products/srtmswbdv003/) [Body](https://lpdaac.usgs.gov/products/astwbdv001/) Databases. To access these data, the script needs an [Earthdata EOSDIS login](https://urs.earthdata.nasa.gov/users/new).

## Requirements

- Python 3.9 (Other versions of Python 3 may work, but I haven't tested them.)

#### Python Modules:

- [coloraide](https://facelessuser.github.io/coloraide/)

- [colour](https://www.colour-science.org/)

- [Pillow](https://python-pillow.org/)

- [opencv-python](https://pypi.org/project/opencv-python/)

- [numba](https://numba.pydata.org/)

- [screeninfo](https://pypi.org/project/screeninfo/)

## Usage

```
usage: asos.py [-h] [--new-login] [--one-time-login] [--previous] [--previous-colors]
               [--previous-light] [--no-water] [--no-shade] [--no-shadows]
               [--output [FILEPATH]] [--coordinates LATITUDE LONGITUDE]
               [--dimensions WIDTH HEIGHT] [--rotation [0-3]] [--min-lightness [0-1]]
               [--max-lightness [0-1]] [--min-chroma [0-0.4]] [--max-chroma [0-0.4]]
               [--hue-delta [Delta-E]] [--lightnesses 0-1 [0-1 ...]]
               [--chromas 0-0.4 [0-0.4 ...]] [--hues 0-359 [0-359 ...]]
               [--water-colors L C H [L C H ...]] [--colorfulness [0-1]]
               [--shine [0-1]] [--glow [0-1]] [--azimuth [0-359]]
               [--altitude-angle [1-90]] [--ambient-strength [0-1]]

Create a colorful image of terrain of a random location.

optional arguments:
  -h, --help            show this help message and exit
  --new-login           Enter an Earthdata username & password and store it encrypted
                        for future use. Overwrites currently stored login information
                        if any.
  --one-time-login      Enter an Earthdata username & password to use only for this
                        run, and do not store it.
  --previous, -p        Use previously downloaded data. --dimensions, --coordinates,
                        and --rotation will have no effect.
  --previous-colors     Use previously used colors. Any additional color arguments will
                        override those specific parts of the previously used colors.
  --previous-light      Use previously used lighting. Any additional lighting arguments
                        will override those specific parts of the previously used
                        lighting.
  --no-water, -w        Do not draw bodies of water.
  --no-shade, -s        Do not hillshade the terrain. This leaves only gradient-mapped
                        elevations and water bodies.
  --no-shadows          Do not cast shadows.
  --output [FILEPATH], -o [FILEPATH]
                        Path to save output image. If not specified, will save to
                        ~/a_sort_of_soup/output.png along with elevation_gradient.tif,
                        hillshade.tif, and water.tif
  --coordinates LATITUDE LONGITUDE, -c LATITUDE LONGITUDE
                        Location of center of desired image in latitude longitude
                        coordinates. If not specified, a random location will be
                        chosen.
  --dimensions WIDTH HEIGHT, -d WIDTH HEIGHT
                        Width and height in pixels of output image. Larger images will
                        require downloading more source tiles. Defaults to screen
                        dimensions.
  --rotation [0-3], -r [0-3]
                        How many times 90 degrees to rotate. (0: North is up. 1: East
                        is up. 2: South is up. 3: West is up.) If not specified, this
                        will be chosen randomly.
  --min-lightness [0-1]
                        Unless specified by --lightnesses, lightnesses will be randomly
                        chosen between --min-lightness and --max-lightness.
  --max-lightness [0-1]
                        Unless specified by --lightnesses, lightnesses will be randomly
                        chosen between --min-lightness and --max-lightness.
  --min-chroma [0-0.4]  Attempt to choose colors with at least this minimum
                        chromaticity.
  --max-chroma [0-0.4]  Maximum chromaticity of image.
  --hue-delta [Delta-E]
                        Minimum color difference between hues as calculated by CIE
                        Delta-E 2000 at 0.74 lightness and 0.12 chromaticity. Values
                        over 35 will usually cause Delta-E between hues to be uneven.
                        If not specified, this will be chosen randomly from 20 through
                        40.
  --lightnesses 0-1 [0-1 ...]
                        Up to three lightnesses, in order of elevation. The remaining
                        lightnesses will be chosen randomly.
  --chromas 0-0.4 [0-0.4 ...]
                        Up to three chromaticities, in order of elevation. The
                        remaining chromas will be chosen randomly. To specify only the
                        second and/or third chromaticities, enter chromaticities of -1
                        to have them chosen randomly.
  --hues 0-359 [0-359 ...]
                        Up to three hues, in order of elevation. The remaining hues
                        will be chosen randomly. To specify only the second and/or
                        third hue, enter hues of -1 to have them chosen randomly.
  --water-colors L C H [L C H ...]
                        Any number of colors for the gradient to fill water bodies
                        with, formatted in a flat list of Lightness Chroma Hue
                        triplets.
  --colorfulness [0-1]  0 creates a gradient in OKLCH space between the three in-sRGB-
                        gamut colors chosen. 1 finds the highest chroma colors possible
                        at or below the *specified* three chromas. 1 by default.
  --shine [0-1]         Intensity of hillshade highlights. 0 by default.
  --glow [0-1]          Opacity of overlay and transparency of hard light blending of
                        hillshade. 0 by default.
  --azimuth [0-359]     Azimuth of sunlight for hillshade and shadows (in 45-degree
                        increments). If not specified, will be a random number from 0
                        through 180.
  --altitude-angle [1-90]
                        Altitude angle of sunlight for hillshade and shadows (in
                        degrees). If not specified, this will be a random number from 7
                        through 45.
  --ambient-strength [0-1]
                        Strength of diffuse light in hillshade, and inverse of the
                        darkness of cast shadows. If not specified, this will be a
                        random number from 0.75 through 1.00.
```

## Examples

![Example Terrain Map 1](example1.jpg)

![Example Terrain Map 2](example2.jpg)

![Example Terrain Map 3](example3.jpg)