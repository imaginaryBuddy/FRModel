import skimage
from scipy.ndimage import distance_transform_edt
from sklearn.preprocessing import minmax_scale
from frmodel.base.D2.frame2D import Frame2D
from scipy import ndimage as ndi
import numpy as np

from skimage.segmentation import watershed
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max

from skimage import morphology
from scipy.ndimage.morphology import binary_dilation

FIG_SIZE = 10

MNL_PATH = "mnl/"

NIR_THRESHOLD = 90 / 256
def BIN_FILTER(inp: Frame2D):
    return inp.data_chn(inp.CHN.NIR).data < NIR_THRESHOLD * (2 ** 14)


BLOB_REMOVAL_PATH = MNL_PATH + "blob_removal.png"
BLOB_CONNECTIVITY = 2
BLOB_MIN_SIZE = 1000
TEXT_X = 0.5
TEXT_Y = 1.02

EDT_PATH = MNL_PATH + "edt.png"

PEAKS_PATH = MNL_PATH + "peaks.png"
PEAKS_FOOTPRINT = 200

WATERSHED_PATH = MNL_PATH + "watershed.png"

CANNY_PATH = MNL_PATH + "canny.png"
CANNY_THICKNESS = 5

def meaningless_segmentation(inp: 'Frame2D',
                             bin_filter=BIN_FILTER,
                             blob_connectivity=BLOB_CONNECTIVITY,
                             blob_min_size=BLOB_MIN_SIZE,
                             peaks_footprint=PEAKS_FOOTPRINT,
                             canny_thickness=CANNY_THICKNESS):

    # ============ BINARIZATION ============
    print("Binarizing Image...", end=" ")

    fig, ax = plt.subplots(1, 3, figsize=(FIG_SIZE, FIG_SIZE // 2), sharey=True)

    binary = np.where(bin_filter(inp), 0, 1).squeeze()
    if isinstance(inp.data, np.ma.MaskedArray):
        binary = np.logical_and(binary, ~inp.data.mask[..., 0])

    print(f"Binarized.")
    # ============ BLOB REMOVAL ============
    print("Removing Small Blobs...", end=" ")
    ax[0].imshow(binary, cmap='gray')
    ax[0].text(TEXT_X, TEXT_Y, 'ORIGINAL',
               horizontalalignment='center', transform=ax[0].transAxes)
    binary = morphology.remove_small_objects(binary.astype(bool),
                                             min_size=blob_min_size,
                                             connectivity=blob_connectivity)

    ax[1].imshow(binary, cmap='gray')
    ax[1].text(TEXT_X, TEXT_Y, 'REMOVE MEANINGLESS',
               horizontalalignment='center', transform=ax[1].transAxes)
    binary = ~morphology.remove_small_objects(~binary,
                                              min_size=blob_min_size,
                                              connectivity=blob_connectivity)

    ax[2].imshow(binary, cmap='gray')
    ax[2].text(TEXT_X, TEXT_Y, 'PATCH MEANINGFUL',
               horizontalalignment='center', transform=ax[2].transAxes)
    fig.tight_layout()
    fig.savefig(BLOB_REMOVAL_PATH)

    print(f"Removed Blobs with size < {blob_min_size}, connectivity = {blob_connectivity}.")
    # ============ DISTANCE ============
    print("Creating Distance Image...", end=" ")
    distances = distance_transform_edt(binary.astype(bool))

    fig, ax = plt.subplots(figsize=(FIG_SIZE, FIG_SIZE))

    i = ax.imshow(-distances, cmap='gray')
    fig: plt.Figure
    fig.colorbar(i, ax=ax)
    fig.tight_layout()
    fig.savefig(EDT_PATH)

    print(f"Created Distance Image.")
    # ============ PEAK FINDING ============
    print("Finding Peaks...", end=" ")
    fig, ax = plt.subplots(figsize=(FIG_SIZE, FIG_SIZE))

    peaks = peak_local_max(distances,
                           footprint=np.ones((peaks_footprint, peaks_footprint)),
                           exclude_border=0,
                           labels=binary)

    ax.imshow(-distances, cmap='gray')
    ax: plt.Axes
    ax.scatter(peaks[..., 1], peaks[..., 0], c='red', s=1)
    ax.text(x=TEXT_X, y=TEXT_Y, s=f"FOOTPRINT {peaks_footprint}", size=10,
            horizontalalignment='center', transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(PEAKS_PATH)

    print(f"Found {peaks.shape[0]} peaks with Footprint {peaks_footprint}.")
    # ============ WATERSHED ============
    print("Running Watershed...", end=" ")
    markers = np.zeros(distances.shape, dtype=bool)
    markers[tuple(peaks.T)] = True
    markers, _ = ndi.label(markers)
    water = watershed(-distances, markers, mask=binary)

    fig, ax = plt.subplots(figsize=(FIG_SIZE, FIG_SIZE))
    ax.imshow(water, cmap="magma")
    ax.scatter(peaks[..., 1], peaks[..., 0], c='red', s=1)

    fig.tight_layout()
    fig.savefig(WATERSHED_PATH)

    print(f"Created Watershed Image.")
    # ============ CANNY EDGE ============
    print("Running Canny Edge Detection...", end=" ")
    canny = skimage.feature.canny(water.astype(float))
    fig, ax = plt.subplots(figsize=(FIG_SIZE, FIG_SIZE))
    ax.axis('off')
    ax.imshow(inp.data_rgb().scale(minmax_scale).data)
    ax.imshow(binary_dilation(canny, structure=np.ones((canny_thickness, canny_thickness))),
              cmap='gray', alpha=0.5)
    fig.savefig(CANNY_PATH)

    print(f"Created Canny Edge Image.")

    buffer = np.zeros([*binary.shape, 4], dtype=np.float64)

    buffer[..., 0] = binary
    buffer[..., 1] = distances
    buffer[..., 2] = water
    buffer[..., 3] = canny

    frame = Frame2D(buffer, labels=[Frame2D.CHN.MNL.BINARY,
                                    Frame2D.CHN.MNL.DISTANCE,
                                    Frame2D.CHN.MNL.WATER,
                                    Frame2D.CHN.MNL.CANNY])

    return dict(frame=frame, peaks=peaks)
