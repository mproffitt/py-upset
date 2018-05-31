""" function wrapper for plotting set intersections """
import numpy as np
from .visualisation import UpSetPlot
from .resources import FilterConfig
from .resources import DataExtractor

def plot(data_dict, *, unique_keys=None, sort_by='size', inters_size_bounds=(0, np.inf),
         inters_degree_bounds=(1, np.inf), additional_plots=None, query=None):
    """
    Plots a main set of graph showing intersection size, intersection matrix and the size of base sets. If given,
    additional plots are placed below the main graph.

    :param data_dict: dictionary like {data_frame_name: data_frame}

    :param unique_keys: list. Specifies the names of the columns that, together, can uniquely identify a row. If left
    empty, pyUpSet will try to use all common columns in the data frames and may possibly raise an exception (no
    common columns) or produce unexpected results (columns in different data frames with same name but different
    meanings/data).

    :param sort_by: 'size' or 'degree'. The order in which to sort the intersection bar chart and matrix in the main
    graph

    :param inters_size_bounds: tuple. Specifies the size limits of the intersections that will be displayed.
    Intersections (and relative data) whose size is outside the interval will not be plotted. Defaults to (0, np.inf).

    :param inters_degree_bounds: tuple. Specified the degree limits of the intersections that will be displayed.
    Intersections (and relative data) whose degree is outside the interval will not be plotted. Defaults to (0, np.inf).

    :param additional_plots: list of dictionaries. See below for details.

    :param query: list of tuples. See below for details.

    :return: dictionary of matplotlib objects, namely the figure and the axes.

    :raise ValueError: if no unique_keys are specified and the data frames have no common column names.

    The syntax to specify additional plots follows the signature of the corresponding matplotlib method in an Axes
    class. For each additional plot one specifies a dictionary with the kind of plot, the columns name to retrieve
    relevant data and the kwargs to pass to the plot function, as in `{'kind':'scatter', 'data':{'x':'col_1',
    'y':'col_2'}, 'kwargs':{'s':50}}`.

    Currently supported additional plots: scatter.

    It is also possible to highlight intersections. This is done through the `query` argument, where the
    intersections to highligh must be specified with the names used as keys in the data_dict.
    """
    query = [] if query is None else query
    additional_plots = [] if additional_plots is None else additional_plots
    unique_keys = list(unique_keys if unique_keys is not None else __get_all_common_columns(data_dict))

    reset = False
    for key in data_dict:
        if list(data_dict[key].columns) == unique_keys:
            reset = True

    if reset:
        for key in data_dict:
            data_dict[key].index.name = '__index' # set the name to something unlikely to clash with existing columns
        data_dict = {key: data_dict[key].reset_index() for key in data_dict}
        unique_keys = ['__index']

    filter_config = FilterConfig()
    filter_config.sort_by = sort_by
    filter_config.size_bounds = inters_size_bounds
    filter_config.degree_bounds = inters_degree_bounds

    plot_data = DataExtractor(unique_keys=unique_keys, filter_config=filter_config)

    # pylint: disable=expression-not-assigned
    # We don't require the value of this list as we're purely interested in list-comprehension
    # to append to the DataExtractor.
    [plot_data.append(name, data_dict[name]) for name in data_dict]

    upset = UpSetPlot(plot_data, additional_plots=additional_plots, highlight=query)
    results = upset.plot()
    dictionary = {
        'figure': results.figure,
        'intersection_bars': results.intersection_bars,
        'intersection_matrix': results.intersection_matrix,
        'base_set_size': results.base_set_size,
        'names': results.names
    }
    if results.additional is not None:
        dictionary['additional'] = results.additional
    return dictionary

def __get_all_common_columns(data_dict):
    """
    Computes an array of (unique) common columns to the data frames in data_dict
    :param data_dict: Dictionary of data frames
    :return: array.
    """
    if isinstance(data_dict, DataExtractor):
        return data_dict.unique_keys
    common_columns = []
    for i, k in enumerate(data_dict.keys()):
        if i == 0:
            common_columns = data_dict[k].columns
        else:
            common_columns = common_columns.intersection(data_dict[k].columns)
    if len(common_columns.values) == 0:
        raise ValueError(
            'Data frames should have homogeneous columns with the same name to use for computing intersections'
        )
    return common_columns.unique().values
