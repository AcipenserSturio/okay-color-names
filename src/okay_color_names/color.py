import numpy as np


def rgb_to_rgbfloat(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    return [v/256 for v in rgb]


def is_valid_rgb(rgb: tuple[int, int, int]) -> bool:
    return (
        all(map(lambda x: x >= 0, rgb))
        & all(map(lambda x: x < 256, rgb))
    )


def is_valid_oklab(oklab: tuple[float, float, float]) -> list[tuple]:
    return is_valid_rgb(oklab_to_rgb(oklab))


# Generated by ChatGPT.
# I would have preferred to use existing implementations,
# but Coloria is paywalled and color-science has really opaque documentation.


def hsl_to_rgb(hsl: tuple[int, int, int]) -> tuple[int, int, int]:
    h, s, l = hsl

    # Convert H, S, L to fractions
    h = h / 360
    s = s / 100
    l = l / 100

    # Calculate chroma
    c = (1 - abs(2 * l - 1)) * s

    # Find an intermediate value
    x = c * (1 - abs((h * 6) % 2 - 1))

    # Calculate the match (lightness adjustment)
    m = l - c / 2

    # Determine RGB values based on hue range
    if 0 <= h < 1/6:
        r, g, b = c, x, 0
    elif 1/6 <= h < 1/3:
        r, g, b = x, c, 0
    elif 1/3 <= h < 1/2:
        r, g, b = 0, c, x
    elif 1/2 <= h < 2/3:
        r, g, b = 0, x, c
    elif 2/3 <= h < 5/6:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x

    # Adjust for lightness and scale to [0, 255]
    r = (r + m) * 255
    g = (g + m) * 255
    b = (b + m) * 255

    return round(r), round(g), round(b)


def rgb_to_oklab(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    # Ensure the RGB values are in the range [0, 1]
    r, g, b = [x / 255.0 for x in rgb]

    # Step 1: Convert to linear RGB
    def srgb_to_linear(c):
        if c <= 0.04045:
            return c / 12.92
        return ((c + 0.055) / 1.055) ** 2.4

    r_lin = srgb_to_linear(r)
    g_lin = srgb_to_linear(g)
    b_lin = srgb_to_linear(b)

    # Step 2: Convert linear RGB to LMS
    rgb_to_lms_matrix = np.array([
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005]
    ])

    linear_rgb = np.array([r_lin, g_lin, b_lin])
    lms = np.dot(rgb_to_lms_matrix, linear_rgb)

    # Step 3: Non-linear transform on LMS
    lms = np.cbrt(lms)

    # Step 4: Convert LMS to Oklab
    lms_to_oklab_matrix = np.array([
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660]
    ])

    oklab = np.dot(lms_to_oklab_matrix, lms)

    return tuple(oklab)


def oklab_to_rgb(oklab: tuple[float, float, float]) -> tuple[int, int, int]:
    # Step 1: Convert Oklab to LMS
    oklab_to_lms_matrix = np.array([
        [1.0000000000,  0.3963377774,  0.2158037573],
        [1.0000000000, -0.1055613458, -0.0638541728],
        [1.0000000000, -0.0894841775, -1.2914855480]
    ])

    L, a, b = oklab

    lms = np.dot(oklab_to_lms_matrix, np.array([L, a, b]))

    # Step 2: Reverse non-linear LMS transform
    lms = lms ** 3  # Cube LMS values

    # Step 3: Convert LMS to linear RGB
    lms_to_rgb_matrix = np.array([
        [4.0767416621,  -3.3077115913,  0.2309699292],
        [-1.2684380046,  2.6097574011, -0.3413193965],
        [-0.0041960863, -0.7034186147,  1.7076147010]
    ])

    linear_rgb = np.dot(lms_to_rgb_matrix, lms)

    # Step 4: Convert linear RGB to sRGB
    def linear_to_srgb(c):
        if c <= 0.0031308:
            return 12.92 * c
        return 1.055 * (c ** (1 / 2.4)) - 0.055

    r, g, b = [linear_to_srgb(c) for c in linear_rgb]

    # Step 5: Clamp RGB values to [0, 1] and convert to [0, 255]
    # r, g, b = [max(0, min(1, x)) for x in (r, g, b)]
    return (round(r * 255), round(g * 255), round(b * 255))