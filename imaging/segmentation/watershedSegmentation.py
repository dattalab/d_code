import numpy as np
import scipy.ndimage as nd
import pymorph
import mahotas

def gradientMagnitudue(image):
    sobel_x = nd.sobel(image.astype('double'), 0)
    sobel_y = nd.sobel(image.astype('double'), 1)
    return np.sqrt((sobel_x * sobel_x) + (sobel_y * sobel_y))    

def imimposemin(image, mask, connectivity):
    fm = image.copy()
    fm[mask] = -9223372036854775800
    fm[np.logical_not(mask)] = 9223372036854775800

    fp1 = image + 1

    g = np.minimum(fp1, fm)

    j = infrec(fm, g)
    return j

def infrec(f, g, Bc=None):
    if Bc is None: Bc = pymorph.secross()
    n = f.size
    return fast_conditional_dilate(f, g, Bc, n);

def fast_conditional_dilate(f, g, Bc=None, n=1):
    if Bc is None: Bc = pymorph.secross()
    f = pymorph.intersec(f,g)
    for i in xrange(n):
        prev = f
        f = pymorph.intersec(mahotas.dilate(f, Bc), g)
        if pymorph.isequal(f, prev): break
    return f

def watershedSegment(image, diskSize=20):
    gradmag = gradientMagnitudue(image)

    ## compute foreground markers

    # open image to create flat regions at cell centers
    se_disk = pymorph.sedisk(diskSize) 
    image_opened = mahotas.open(image, se_disk);

    # define foreground markers as regional maxes of cells
    # this step is slow!
    foreground_markers = mahotas.regmax(image_opened)

    ## compute background markers

    # Threshold the image, cast it to the right datatype, and then calculate the distance image
    image_black_white = image_opened > mahotas.otsu(image_opened)
    image_black_white = image_black_white.astype('uint16')

    # note the inversion here- a key difference from the matlab algorithm
    # matlab distance is to nearest non-zero pixel
    # python distance is to nearest 0 pixel
    image_distance = pymorph.to_uint16(nd.distance_transform_edt(np.logical_not(image_black_white)))
    eight_conn = pymorph.sebox()

    distance_markers = mahotas.label(mahotas.regmin(image_distance, eight_conn))[0]
    image_dist_wshed, image_dist_wshed_lines =mahotas.cwatershed(image_distance, distance_markers, eight_conn, return_lines=True)
    background_markers = image_distance_watershed_lines - image_black_white

    all_markers = np.logical_or(foreground_markers, background_markers)

    # impose a min on the gradient image.  assumes int64
    gradmag2 = imimposemin(gradmag.astype(int), all_markers, eight_conn)

    # call watershed
    segmented_cells, segmented_cell_lines = mahotas.cwatershed(gradmag2, mahotas.label(all_markers)[0], eight_conn, return_lines=True)

    # seperate watershed regions
    segmented_cells[gradientMagnitudue(segmented_cells) > 0] = 0
    return segmented_cells > 0, segmented_cells
