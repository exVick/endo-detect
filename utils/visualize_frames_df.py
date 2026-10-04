def _read_frame(row):
    """reads the image of one frame; dicom files are read with pydicom and
    everything else with matplotlib. for a multi-frame dicom the frame is
    picked out of the volume by its slice_rank."""
    import matplotlib.pyplot as plt

    path = str(row.FilePath)
    if path.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
        return plt.imread(path)
    import pydicom
    arr = pydicom.dcmread(path).pixel_array
    if arr.ndim == 3:
        arr = arr[int(row.slice_rank)]
    return arr



def visualize_frames(df_frames):
    """shows the frames of one patient with a dropdown over the series types.

    the input is expected to be already preselected (one patient / one accession)
    and to carry the columns:
      - FilePath             : path of the image file of the frame
      - SeriesDescription    : series type, used to group the frames
      - slice_rank           : order of the slice inside its series
      - n_slices_in_series   : total number of slices of that series

    a dropdown over the distinct SeriesDescription values and a slider over the
    slices of the selected series are displayed; nothing is returned.
    """

    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from IPython.display import display

    # -- frames are grouped by series type and ordered by their rank --
    df_frames = df_frames.sort_values(["SeriesDescription", "slice_rank"], ignore_index=True)
    series_names = list(df_frames["SeriesDescription"].unique())

    # -- widgets --
    series_dropdown = widgets.Dropdown(
        options=series_names,
        description="Series:",
        layout=widgets.Layout(width="70%"),
    )
    slice_slider = widgets.IntSlider(
        min=0, max=0, step=1, value=0,
        description="Slice:",
        layout=widgets.Layout(width="70%"),
        continuous_update=True,
    )
    out = widgets.Output()

    # -- the rows of the selected series are kept here between slider ticks --
    _cache = {"df_series": None}

    def _update_slider_range(_):
        """selects the rows of the chosen series and adjusts the slider bounds."""
        df_series = df_frames[df_frames["SeriesDescription"] == series_dropdown.value]
        _cache["df_series"] = df_series
        slice_slider.max = max(len(df_series) - 1, 0)
        slice_slider.value = 0

    def _show_slice(_):
        """displays the frame at the current slider position."""
        df_series = _cache["df_series"]
        if df_series is None or len(df_series) == 0:
            return
        idx = min(slice_slider.value, len(df_series) - 1)
        row = df_series.iloc[idx]
        frame = _read_frame(row)
        with out:
            out.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(frame, cmap="bone")
            ax.set_title(
                f"{series_dropdown.value}  |  slice {int(row.slice_rank)}"
                f"/{int(row.n_slices_in_series)}"
            )
            ax.axis("off")
            plt.tight_layout()
            plt.show()

    series_dropdown.observe(_update_slider_range, names="value")
    series_dropdown.observe(_show_slice, names="value")
    slice_slider.observe(_show_slice, names="value")

    # -- initialise --
    _update_slider_range(None)
    display(widgets.VBox([series_dropdown, slice_slider, out]))
    _show_slice(None)


def save_series_gif(df_frames, series_description, output_dir, duration=150, loop=0):
    """writes the frames of one series out as an animated gif.

    the same dataframe as for `visualize_frames` is expected (columns FilePath,
    SeriesDescription, slice_rank, n_slices_in_series); the rows of the given
    series are taken, ordered by slice_rank and written as one gif into
    `output_dir`. the animation goes forth through the slices and back again,
    so it starts and ends on the first slice and loops without a jump.

    Args:
        df_frames: frame table of one patient, already preselected.
        series_description: value of SeriesDescription whose frames are animated.
        output_dir: directory the gif is written into, created if missing.
        duration: milliseconds each frame is shown.
        loop: number of repetitions, 0 meaning endless.

    Returns:
        The path of the written gif.
    """
    import os
    import numpy as np
    from PIL import Image

    df_series = df_frames[df_frames["SeriesDescription"] == series_description]
    df_series = df_series.sort_values("slice_rank")
    acc = df_series.AccessionNumber.iloc[0]

    frames = []
    for row in df_series.itertuples():
        arr = np.asarray(_read_frame(row), dtype=float)
        if arr.ndim == 3:
            # colour images are reduced to grey so that all frames share one mode
            arr = arr.mean(axis=-1)
        # each frame is scaled to the full 8-bit range for display
        lo, hi = arr.min(), arr.max()
        arr = (arr - lo) / (hi - lo) * 255 if hi > lo else np.zeros_like(arr)
        frames.append(Image.fromarray(arr.astype(np.uint8)))

    # the size of the first frame is taken as reference, the rest is matched to it
    size = frames[0].size
    frames = [f if f.size == size else f.resize(size) for f in frames]

    # the slices are played forth and back again; the last and first slice are
    # not repeated at the turning points, so no frame is shown twice in a row
    if loop == 0:
        # endless loop: the wrap from the end back to the start closes the cycle
        frames = frames + frames[-2:0:-1]
    else:
        # finite loop: the first slice is appended so that the gif stops on it
        frames = frames + frames[-2::-1]

    os.makedirs(output_dir, exist_ok=True)
    # a filename that is safe on disk is built from the series name
    stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(series_description))
    out_path = os.path.join(output_dir, f"{acc}_{stem}.gif")
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop,
    )
    return out_path
