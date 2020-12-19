from frmodel.base.D2 import Frame2D


def channel_analysis(image_path: str,
                     verbose:bool = True,
                     image_scale: float = 0.5,
                     plot_scale: float = 0.2,
                     exclude_xy: bool = True):
    """ Quickly analyzes the image provided with multiple channels

    :param image_path: Path to image
    :param verbose: Whether to show output for GLCM loading
    :param image_scale: Scale of Image
    :param plot_scale: Scale of Plot
    :param exclude_xy: Whether to exclude XY trivial channel or not
    :return: A matplotlib fig
    """

    f = Frame2D.from_image(image_path, scale=image_scale)
    f = f.get_all_chns(exc_chns=[Frame2D.CHN.XY] if exclude_xy else [],
                       glcm=Frame2D.GLCM(verbose=verbose))
    fig = f.plot(f.labels).image(scale=plot_scale)

    return fig
