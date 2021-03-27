import skimage
from scipy.ndimage import distance_transform_edt
from sklearn.preprocessing import minmax_scale
from frmodel.base.D2.frame2D import Frame2D
from scipy import ndimage as ndi
import numpy as np
from pathlib import Path

from skimage.segmentation import watershed
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max

from skimage import morphology
from scipy.ndimage.morphology import binary_dilation

FIG_SIZE = 10
NIR_THRESHOLD = 90 / 256

BLOB_CONNECTIVITY = 2
BLOB_MIN_SIZE = 1000
TEXT_X = 0.5
TEXT_Y = 1.02

PEAKS_FOOTPRINT = 200
CANNY_THICKNESS = 5


# noinspection PyPep8Naming
def BIN_FILTER(inp: Frame2D):
    # noinspection PyTypeChecker
    return inp.data_chn(inp.CHN.NIR).data < NIR_THRESHOLD * (2 ** 14)


def meaningless_segmentation(inp: 'Frame2D',
                             bin_filter=BIN_FILTER,
                             blob_connectivity=BLOB_CONNECTIVITY,
                             blob_min_size=BLOB_MIN_SIZE,
                             peaks_footprint=PEAKS_FOOTPRINT,
                             canny_thickness=CANNY_THICKNESS,
                             output_dir="mnl/"):
    """ Runs the Meaningless Segmentation as depicted in the journal

    The output_dir will be automatically created if it doesn't exist.
    Default: "mnl/"

    :param inp: Input Frame2D, can be MaskedData
    :param bin_filter: A function that takes in Frame2D and returns a boolean mask
    :param blob_connectivity: Connectivity of morphology.remove_small_objects
    :param blob_min_size: Min Size of morphology.remove_small_objects
    :param peaks_footprint: Footprint of Local Peak Max
    :param canny_thickness: Thickness of Canny line
    :param output_dir: Output directory, will be created if doesn't exist
    :return: Dictionary of "frame": Frame2D, "peaks": np.ndarray
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    blob_removal_path = output_dir + "blob_removal.png"
    edt_path          = output_dir + "edt.png"
    peaks_path        = output_dir + "peaks.png"
    watershed_path    = output_dir + "watershed.png"
    canny_path        = output_dir + "canny.png"

    # ============ BINARIZATION ============
    print("Binarizing Image...", end=" ")

    # noinspection PyTypeChecker
    fig, ax = plt.subplots(1, 3, figsize=(FIG_SIZE, FIG_SIZE // 2),
                           sharey=True)

    binary = np.where(bin_filter(inp), 0, 1).squeeze()
    if isinstance(inp.data, np.ma.MaskedArray):
        # noinspection PyUnresolvedReferences
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
    fig.savefig(blob_removal_path)
    fig.clf()

    print(f"Removed Blobs with size < {blob_min_size}, "
          f"connectivity = {blob_connectivity}.")
    # ============ DISTANCE ============
    print("Creating Distance Image...", end=" ")
    distances = distance_transform_edt(binary.astype(bool))

    fig, ax = plt.subplots(figsize=(FIG_SIZE, FIG_SIZE))

    i = ax.imshow(-distances, cmap='gray')
    fig: plt.Figure
    fig.colorbar(i, ax=ax)
    fig.tight_layout()
    fig.savefig(edt_path)
    fig.clf()

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
    fig.savefig(peaks_path)
    fig.clf()

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
    fig.savefig(watershed_path)
    fig.clf()

    print(f"Created Watershed Image.")
    # ============ CANNY EDGE ============
    print("Running Canny Edge Detection...", end=" ")
    canny = skimage.feature.canny(water.astype(float))
    fig, ax = plt.subplots(figsize=(FIG_SIZE, FIG_SIZE))
    ax.axis('off')
    ax.imshow(inp.data_rgb().scale(minmax_scale).data)
    ax.imshow(binary_dilation(canny, structure=np.ones((canny_thickness, canny_thickness))),
              cmap='gray', alpha=0.5)
    fig.savefig(canny_path)
    fig.clf()

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
